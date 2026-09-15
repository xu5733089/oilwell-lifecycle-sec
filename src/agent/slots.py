"""槽位抽取（方案 §7.2 第 2 段）。

结构化抽参 + JSON Schema 校验 + 缺槽反问。
**缺槽时明确问缺哪一个，绝不用默认值蒙混** —— 蒙对一次，用户就会一直信它。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import price_decks
from .llm_client import LLMClient, LLMError

# 井号形如 GL-A-0123 / SB-C-0007 / SYN0123 / X-12。
# 刻意放宽：抽得出来才能把"这口井不存在"如实告诉用户；
# 抽不出来只会退化成"请提供井号"，反而掩盖了真正的问题。
WELL_RE = re.compile(r"\b([A-Z]{1,4}(?:-[A-Z0-9]{1,6}){1,3}|SYN\d{3,5})\b", re.I)
OBS_RE = re.compile(r"(?:前|头|首)?\s*(\d{2,3})\s*(?:天|日)")
DECK_RE = re.compile(r"\b(deck_\d{4}_\d{2})\b")
DATE_RE = re.compile(r"(20\d{2})[-/年]\s*(\d{1,2})[-/月]?\s*(\d{1,2})?")
TOPK_RE = re.compile(r"(?:top\s*|前)(\d{1,2})\s*(?:口|个)?")
SCOPE_ID_RE = re.compile(r"(?<![A-Za-z0-9])(SEC_[A-Za-z0-9_]+)", re.I)
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
EXCL_RE = re.compile(r"(?:扣|近)\s*(\d{1,2})\s*年")
SCENARIO_KW = [("考核", "assessment"), ("减值", "impairment"), ("sec价", "sec"), ("sec 价", "sec")]
COMPANY_WORDS = ("全公司", "油田公司", "全油田", "公司整体", "公司层面")

# 每个意图需要哪些槽位；required 缺失就反问
SCHEMA: Dict[str, Dict[str, List[str]]] = {
    "predict_lifecycle": dict(required=["well_code"], optional=["obs_days", "targets"]),
    "find_analogs":      dict(required=["well_code"], optional=["top_k"]),
    "fit_dca":           dict(required=["well_code"], optional=["model", "price_deck_id"]),
    "estimate_reserves": dict(required=["well_code"], optional=["mc_samples"]),
    "cross_check":       dict(required=["well_code"], optional=[]),
    "sec_screen":        dict(required=["well_code"], optional=["as_of", "price_deck_id"]),
    "query_well":        dict(required=["well_code"], optional=["fields"]),
    "gen_report":        dict(required=["well_code"], optional=["as_of", "price_deck_id"]),
    "unit_composition":  dict(required=["scope"], optional=["as_of", "scenario"]),
    "unit_decline":      dict(required=["scope"], optional=["as_of", "exclude_years"]),
    "measure_effect":    dict(required=["scope"], optional=["as_of", "event_type"]),
    "new_well_identify": dict(required=["scope"], optional=["as_of"]),
    "unit_reconcile":    dict(required=["scope"], optional=["from_as_of", "to_as_of", "scenario"]),
    "unit_sensitivity":  dict(required=["scope"], optional=["as_of", "scenario"]),
    "unit_categories":   dict(required=["scope"], optional=["as_of", "from_as_of", "to_as_of", "scenario"]),
    "unit_depletion":    dict(required=["scope"], optional=["as_of"]),
    "fallback":          dict(required=[], optional=[]),
}

ASK = {
    "scope": "请提供评估对象：SEC 单元号（例如 SEC_GLA_Q1）、采油厂名称或公司名称",
    "well_code": "请提供井号（例如 GL-A-0123）",
    "as_of": "请提供评估基准日（例如 2026-12-31）",
    "price_deck_id": "请指定价格册（例如 deck_2026_12）",
}


@dataclass
class Slots:
    values: Dict[str, object] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    method: str = "rule"

    @property
    def complete(self) -> bool:
        return not self.missing

    def question(self) -> str:
        return "；".join(ASK.get(m, f"请补充 {m}") for m in self.missing)

    def to_dict(self) -> Dict:
        return dict(values=self.values, missing=self.missing, method=self.method)


def rule_extract(text: str) -> Dict[str, object]:
    v: Dict[str, object] = {}
    m = WELL_RE.search(text)
    if m:
        v["well_code"] = m.group(1).upper()
    m = OBS_RE.search(text)
    if m and 20 <= int(m.group(1)) <= 365:
        v["obs_days"] = int(m.group(1))
    m = DECK_RE.search(text)
    if m:
        v["price_deck_id"] = m.group(1)
    m = DATE_RE.search(text)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3) or 31)
        v["as_of"] = f"{y}-{mo:02d}-{min(d, 28) if mo == 2 else d:02d}"
    m = TOPK_RE.search(text)
    if m and 1 <= int(m.group(1)) <= 20:
        v["top_k"] = int(m.group(1))

    # ---- 单元级槽位 ----
    m = SCOPE_ID_RE.search(text)
    scope = m.group(1).upper() if m else _scope_by_name(text)
    if scope:
        v["scope"] = scope
    years = sorted({int(y) for y in YEAR_RE.findall(text)})
    if len(years) >= 2:
        v["from_as_of"], v["to_as_of"] = f"{years[0]}-12-31", f"{years[-1]}-12-31"
    elif len(years) == 1 and "as_of" not in v:
        v["as_of"] = f"{years[0]}-12-31"          # 评估按年度，基准日取年末
    if "as_of" in v and "price_deck_id" not in v:
        # 基准日与价格册必须配套：只改基准日不换价格册，会拿别的年份的油价算这一年的经济极限
        deck = next((k for k, d in price_decks().items() if str(d["as_of"]) == v["as_of"]), None)
        if deck:
            v["price_deck_id"] = deck
    low = text.lower()
    for kw, code in SCENARIO_KW:
        if kw in low:
            v["scenario"] = code
            break
    excl = EXCL_RE.findall(text)
    if excl:
        v["exclude_years"] = ",".join(dict.fromkeys(excl))
    for name in _measure_names():
        if name in text:
            v["event_type"] = name
            break
    return v


def _scope_by_name(text: str) -> Optional[str]:
    """按名称匹配采油厂 / SEC 单元 / 公司。名称来自单元表，不在代码里写死。"""
    try:
        from ..api import services as S
        u = S.list_units()
    except Exception:            # 库未初始化时不影响单井问答
        return None
    company = u["company"]["name"]
    if company in text or any(w in text for w in COMPANY_WORDS):
        return company
    for p in u["plants"]:
        if p["plant_name"] in text:
            return p["plant_name"]
    return None


def _measure_names() -> List[str]:
    from ..config import config
    return list(config().get("sec_unit", {}).get("measure_types", {}).values())


def extract(text: str, intent: str, client: Optional[LLMClient] = None) -> Slots:
    schema = SCHEMA.get(intent, SCHEMA["fallback"])
    values = rule_extract(text)
    method = "rule"

    need = [k for k in schema["required"] if k not in values]
    if need and client is not None and client.backend != "mock":
        prompt = (f"从用户问题中抽取字段，只输出 JSON，缺失的字段填 null。\n"
                  f"需要抽取：{schema['required'] + schema['optional']}\n"
                  f"问题：{text}")
        try:
            data = client.chat_json([{"role": "user", "content": prompt}])
            for k in schema["required"] + schema["optional"]:
                if data.get(k) not in (None, "", []) and k not in values:
                    values[k] = data[k]
            method = "rule+llm"
        except LLMError:
            pass

    missing = [k for k in schema["required"] if k not in values]
    return Slots(values=values, missing=missing, method=method)
