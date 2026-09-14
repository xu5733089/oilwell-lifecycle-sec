"""意图路由（方案 §7.2 第 1 段）。

内网可用的是 27B 量级模型，**不用自由 ReAct**：
多轮自由工具调用容易漏参数、重复调用、调错工具，演示当场跑飞是最糟的失败模式。

这里用「关键词规则打底 + 大模型兜底」的顺序，而不是反过来：
规则命中率高、零延迟、可解释；模型只处理规则拿不准的口语化问法。
置信度低于阈值就反问澄清，绝不猜。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import config
from .llm_client import LLMClient, LLMError

INTENTS: Dict[str, Dict] = {
    "predict_lifecycle": dict(
        desc="预测新井的见油时间、达峰时间、达峰产量、达峰压力、EUR",
        kw=["预测", "达峰", "见油", "多久", "什么时候", "峰值", "生命周期", "eur", "产能"]),
    "find_analogs": dict(
        desc="找相似的类比井",
        kw=["类比", "相似", "像哪", "邻井", "对比井", "参照井"]),
    "fit_dca": dict(
        desc="递减曲线分析，拟合递减参数与 EUR",
        kw=["递减", "dca", "arps", "曲线拟合", "b值", "递减率"]),
    "estimate_reserves": dict(
        desc="容积法储量估算",
        kw=["容积法", "地质储量", "ooip", "储量估算", "储量计算", "储量多少", "储量"]),
    "cross_check": dict(
        desc="动静态储量互校与一致性诊断",
        kw=["互校", "一致", "对比储量", "采收率", "动静态", "校验"]),
    "sec_screen": dict(
        desc="SEC 储量预评估：分类、经济极限、满足性检查清单",
        kw=["sec", "已证实", "proved", "pdp", "pud", "储量评估", "合规", "准则",
            "审计", "提交", "披露"]),
    "query_well": dict(
        desc="查询单井基础信息与生产现状",
        kw=["基本信息", "查一下", "什么情况", "现状", "累产", "含水", "查询"]),
    "gen_report": dict(
        desc="生成评估报告初稿",
        kw=["报告", "出个报告", "汇报", "文档", "导出"]),
    "fallback": dict(desc="无法归类，需澄清", kw=[]),
}

# 平票时按"下游优先"取舍：一句话里同时出现"做完递减分析"和"能不能进已证实储量"，
# 用户真正要的是后者。上游分析是手段，下游结论才是目的。
PRIORITY = ["gen_report", "sec_screen", "cross_check", "estimate_reserves",
            "fit_dca", "predict_lifecycle", "find_analogs", "query_well"]

# 越权动词：平台对数据只读。带这些动词的请求先拦下来，
# 不能因为句子里恰好出现"储量"就被路由到某个分析意图 ——
# 工具层虽然是只读的拦得住，但路由层就该先说清楚"这件事我不做"。
OUT_OF_SCOPE_VERBS = ["写回", "写入", "删除", "删掉", "清空", "覆盖", "修改", "改成",
                      "更新到", "导入", "入库", "回写", "提交到系统"]


SYSTEM = """你是油井全生命周期评估平台的意图路由器。
把用户问题归到下列意图之一，只输出 JSON：{"intent": "...", "confidence": 0.0-1.0}
可选意图：
""" + "\n".join(f"- {k}: {v['desc']}" for k, v in INTENTS.items() if k != "fallback")


@dataclass
class RouteResult:
    intent: str
    confidence: float
    method: str                       # keyword | llm | fallback
    candidates: List[str]

    def to_dict(self) -> Dict:
        return dict(intent=self.intent, confidence=round(self.confidence, 3),
                    method=self.method, candidates=self.candidates)


def keyword_route(text: str) -> RouteResult:
    t = text.lower()
    if any(v in t for v in OUT_OF_SCOPE_VERBS):
        return RouteResult("fallback", 1.0, "out_of_scope", [])
    hits = {k: sum(1 for w in v["kw"] if w in t) for k, v in INTENTS.items() if v["kw"]}
    top = max(hits.values()) if hits else 0
    tied = [k for k, v in hits.items() if v == top and v > 0]
    best = min(tied, key=PRIORITY.index) if tied else "fallback"
    n = hits.get(best, 0)
    if n == 0:
        return RouteResult("fallback", 0.0, "keyword", [])
    # 命中词越多越有把握，但单词命中封顶 0.6，留给模型二次确认的余地
    conf = min(0.45 + 0.18 * n, 0.95)
    cands = sorted([k for k, v in hits.items() if v > 0], key=lambda k: -hits[k])
    return RouteResult(best, conf, "keyword", cands)


def route(text: str, client: Optional[LLMClient] = None,
          threshold: Optional[float] = None) -> RouteResult:
    threshold = threshold or config()["agent"]["router_confidence_threshold"]
    kw = keyword_route(text)
    if kw.method == "out_of_scope" or kw.confidence >= 0.6:
        return kw

    client = client or LLMClient()
    if client.backend == "mock":
        return kw if kw.confidence >= threshold else RouteResult(
            "fallback", kw.confidence, "fallback", kw.candidates)
    try:
        data = client.chat_json([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": text}])
        intent = str(data.get("intent", "fallback"))
        conf = float(data.get("confidence", 0.5))
        if intent not in INTENTS:
            intent, conf = "fallback", 0.0
        return RouteResult(intent, conf, "llm", kw.candidates)
    except (LLMError, ValueError, TypeError):
        return kw if kw.confidence >= threshold else RouteResult(
            "fallback", kw.confidence, "fallback", kw.candidates)
