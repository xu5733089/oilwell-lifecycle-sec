"""大模型客户端：OpenAI 兼容接口，三个后端可切换（方案 §2.2）。

    internal — 内网部署的 Qwen3.8-27B，生产用
    glm      — 智谱 GLM，开发期对拍用（只喂合成数据与公开数据）
    mock     — 不调任何模型的确定性后端

mock 后端不是玩具：整条智能体链路（路由 / 槽位 / 计划 / 成文 / 校验）
在没有模型的情况下也必须能跑通并被评测 —— 这既是离线开发的需要，
也是演示当天模型服务不可用时的兜底路径（方案 §12 风险预案）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from ..config import config


class LLMError(RuntimeError):
    """模型调用失败。上层须如实告知用户，不得用其它数据顶替。"""


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    usage: Dict[str, int]


class LLMClient:
    def __init__(self, backend: Optional[str] = None):
        cfg = config()["llm"]
        self.backend = backend or cfg["backend"]
        self.timeout = cfg.get("timeout_s", 60)
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 1200, json_mode: bool = False) -> LLMResponse:
        if self.backend == "mock":
            return _mock_chat(messages, json_mode)

        conf = self.cfg.get(self.backend)
        if not conf:
            raise LLMError(f"未配置后端 {self.backend!r}，可选：internal / glm / mock")
        key = os.environ.get(conf.get("api_key_env", ""), "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": conf["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = httpx.post(f"{conf['base_url'].rstrip('/')}/chat/completions",
                           json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:                       # 网络/鉴权/超时统一收口
            raise LLMError(f"{self.backend} 后端调用失败：{exc}") from exc
        return LLMResponse(text=data["choices"][0]["message"]["content"],
                           backend=self.backend, model=conf["model"],
                           usage=data.get("usage", {}))

    def chat_json(self, messages: List[Dict[str, str]], **kw) -> Dict:
        """要求模型输出 JSON。27B 级模型常带 markdown 围栏，这里统一剥掉。"""
        resp = self.chat(messages, json_mode=True, **kw)
        return parse_json(resp.text)


def parse_json(text: str) -> Dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"模型未返回合法 JSON：{text[:180]}")


# ---------------------------------------------------------------------- #
def _mock_chat(messages: List[Dict[str, str]], json_mode: bool) -> LLMResponse:
    """确定性 mock：路由与槽位走关键词，成文走模板。

    模板成文的每个数字都直接取自工具返回的 JSON，
    因此天然通过 guard 的数值一致性校验 —— 这正是"数字与文字分离"的极端形态。
    """
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if json_mode:
        return LLMResponse(text=json.dumps({"_mock": True}, ensure_ascii=False),
                           backend="mock", model="mock", usage={})
    return LLMResponse(text=f"[mock] {user[:200]}", backend="mock", model="mock", usage={})
