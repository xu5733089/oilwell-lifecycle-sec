"""工具注册表（方案 §7.3）与任务边界（§7.4）。

工具是智能体唯一能接触数据和数值的通道。
每个工具都是对 src/api/services.py 的薄封装 —— 大模型不能绕过它们直接读库。

任务边界在这里落成代码，不是文档里的一句话：
  · 只读数据，不写生产库
  · 只能调白名单内的工具
  · 单会话调用次数上限、单工具超时上限
  · 工具失败如实上报，不用其它数据顶替
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..api import services as S
from ..config import config
from .. import trace


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Dict]
    description: str
    parameters: Dict[str, str]
    required: List[str]

    def schema(self) -> Dict:
        """OpenAI function-calling 风格的 schema，供支持工具调用的后端使用。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string", "description": v}
                                   for k, v in self.parameters.items()},
                    "required": self.required,
                },
            },
        }


def _search_standard(query: str, top_k: int = 3, trace_id: Optional[str] = None) -> Dict:
    from .rag.retriever import Retriever
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return dict(query=query, results=_RETRIEVER.search(query, top_k=top_k),
                trace_id=trace_id)


_RETRIEVER = None

REGISTRY: Dict[str, ToolSpec] = {
    t.name: t for t in [
        ToolSpec("query_well", S.query_well, "查询单井基础信息与生产现状",
                 dict(well_code="井号", fields="需要的字段列表，可省略"), ["well_code"]),
        ToolSpec("predict_lifecycle", S.predict_lifecycle,
                 "用早期数据预测见油时间、达峰时间/产量/压力与 EUR，带 P10/P50/P90 区间",
                 dict(well_code="井号", obs_days="观测窗天数，默认 90",
                      targets="要预测的目标列表，可省略"), ["well_code"]),
        ToolSpec("find_analog_wells", S.find_analog_wells,
                 "检索早期曲线形态与静态参数最相似的老井",
                 dict(well_code="井号", top_k="返回数量，默认 5"), ["well_code"]),
        ToolSpec("fit_dca", S.fit_dca,
                 "递减曲线分析，输出递减参数、EUR 概率区间与经济极限时刻",
                 dict(well_code="井号", model="arps 模型，默认 auto 自动择优",
                      price_deck_id="价格册 id"), ["well_code"]),
        ToolSpec("estimate_reserves_volumetric", S.estimate_reserves_volumetric,
                 "容积法地质储量 + 蒙特卡洛不确定性",
                 dict(well_code="井号", mc_samples="抽样次数，默认 10000"), ["well_code"]),
        ToolSpec("cross_check_reserves", S.cross_check_reserves,
                 "动静态储量互校，输出一致性判定与归因建议",
                 dict(well_code="井号"), ["well_code"]),
        ToolSpec("sec_screen", S.sec_screen,
                 "SEC 储量预评估：类别判定、经济极限、满足性检查清单",
                 dict(well_code="井号", as_of="评估基准日", price_deck_id="价格册 id"),
                 ["well_code"]),
        ToolSpec("search_standard", _search_standard,
                 "检索 SEC 储量准则条款原文，返回条款号与正文",
                 dict(query="检索问题", top_k="返回条数，默认 3"), ["query"]),
    ]
}

READ_ONLY = set(REGISTRY)            # 目前全部工具均为只读；新增写操作须单独评审


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    status: str = "ok"               # ok / failed / denied
    result: Optional[Dict] = None
    error: Optional[str] = None
    elapsed_ms: int = 0

    def to_dict(self) -> Dict:
        d = dict(tool=self.name, args=self.args, status=self.status,
                 elapsed_ms=self.elapsed_ms)
        if self.error:
            d["error"] = self.error
        return d


class ToolRunner:
    """执行工具并留痕。超出边界直接拒绝，不给模型商量的余地。"""

    def __init__(self, trace_id: str, max_calls: Optional[int] = None):
        self.trace_id = trace_id
        self.max_calls = max_calls or config()["llm"].get("max_tool_calls", 8)
        self.calls: List[ToolCall] = []

    def run(self, name: str, **args) -> ToolCall:
        if name not in REGISTRY:
            call = ToolCall(name, args, status="denied",
                            error=f"工具 {name!r} 不在白名单内，拒绝执行")
            self.calls.append(call)
            trace.audit(self.trace_id, "agent", f"tool:{name}", args, "denied")
            return call
        if len(self.calls) >= self.max_calls:
            call = ToolCall(name, args, status="denied",
                            error=f"单会话工具调用已达上限 {self.max_calls} 次")
            self.calls.append(call)
            trace.audit(self.trace_id, "agent", f"tool:{name}", args, "denied")
            return call

        spec = REGISTRY[name]
        t0 = time.time()
        try:
            result = spec.fn(trace_id=self.trace_id, **args)
            call = ToolCall(name, args, "ok", result=result)
        except S.KernelError as exc:
            call = ToolCall(name, args, "failed", error=str(exc))
        except TypeError as exc:
            call = ToolCall(name, args, "failed", error=f"参数不合法：{exc}")
        except Exception as exc:                      # 兜底，不让异常冒到用户面前
            call = ToolCall(name, args, "failed", error=f"内部错误：{type(exc).__name__}: {exc}")
        call.elapsed_ms = int((time.time() - t0) * 1000)
        self.calls.append(call)
        trace.audit(self.trace_id, "agent", f"tool:{name}", args, call.status)
        return call

    def results(self) -> Dict[str, Dict]:
        return {c.name: c.result for c in self.calls if c.status == "ok" and c.result}

    def failures(self) -> List[ToolCall]:
        return [c for c in self.calls if c.status != "ok"]

    def trace(self) -> List[Dict]:
        return [c.to_dict() for c in self.calls]


def schemas() -> List[Dict]:
    return [t.schema() for t in REGISTRY.values()]
