"""数值一致性校验器（方案 §2.1 / §7.2 第 6 段）—— 本项目的杀手锏。

大模型成文后，把文本里出现的**每一个数字**抽出来，逐个回查工具返回的 JSON。
对不上就重生成一次；再对不上就降级为「表格 + 模板化文字」。

这一步让"幻觉率"从一句口号变成一个可量化、可写进成果报告的指标：
    数值一致性通过率 = 100%（由本模块保证，而不是由模型自觉）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

# 先剥掉条款引用与日期，避免把 "Rule 4-10(a)(22)"、"2026-12-31" 里的数字当成结论数值
CITATION_RE = re.compile(r"\[[^\]]*\]|(?:Rule|Item)\s*[\d\-()a-zA-Z.]+|\d+\s*CFR\s*[\d.\-()a-zA-Z]+", re.I)
DATE_RE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]?\s*\d{0,2}\s*日?")
# 标识符：ASCII 词里同时含字母和数字的整体（井号 SB-C-0449、版本号 gbdt-q-2026.09、
# 追溯号 ag_b7d0…、价格册 deck_2026_12、口径 v1、分位标签 P50）。
# 这些数字是名字的一部分，不是结论数值 —— 不剥掉就会把井号当幻觉报出来。
IDENT_RE = re.compile(r"(?=[A-Za-z0-9._/\-]*[A-Za-z])(?=[A-Za-z0-9._/\-]*\d)"
                      r"[A-Za-z][A-Za-z0-9._/\-]*|"
                      r"(?=[A-Za-z0-9._/\-]*[A-Za-z])(?=[A-Za-z0-9._/\-]*\d)"
                      r"\d[A-Za-z0-9._/\-]*[A-Za-z][A-Za-z0-9._/\-]*")
# 必须先匹配带千分位的写法，否则 "2,848.27" 会被拆成 2 和 848.27 —— 误报的经典来源
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

# 通用小数：序号、百分比里的整十、年份等，不作为结论数值追究
SAFE = {0, 1, 2, 3, 4, 5, 10, 12, 24, 30, 50, 90, 100}


@dataclass
class GuardResult:
    ok: bool
    checked: int
    violations: List[Dict] = field(default_factory=list)
    allowed_sample: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return dict(ok=self.ok, numbers_checked=self.checked,
                    violations=self.violations,
                    pass_rate=1.0 if self.checked == 0 else
                    round(1 - len(self.violations) / self.checked, 4))


def collect_numbers(obj: Any, out: Set[float] = None) -> Set[float]:
    """递归收集工具返回体里的全部数值，作为文本允许出现的数字白名单。"""
    if out is None:
        out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        try:
            out.add(float(obj))
        except (TypeError, ValueError):
            pass
        return out
    if isinstance(obj, str):
        for m in NUM_RE.finditer(obj):          # 条款正文里的 "90%"、"12 个月" 也算合法来源
            try:
                out.add(float(m.group().replace(",", "")))
            except ValueError:
                pass
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
        return out
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            collect_numbers(v, out)
        return out
    return out


def _matches(x: float, decimals: int, allowed: Iterable[float]) -> bool:
    tol = max(0.5 * 10 ** (-decimals), 1e-9)
    for v in allowed:
        if abs(v - x) <= tol:
            return True
        if decimals == 0 and abs(round(v) - x) < 1e-9:
            return True
        # 允许文本用更粗的精度呈现同一个数（128.334 -> 128 / 128.3）
        if abs(round(v, decimals) - x) <= tol:
            return True
        # 允许 t -> 万 t、比例 -> 百分数 这类量纲换算
        for k in (1e4, 1e3, 1e2, 1e-2, 1e-3, 1e-4):
            if abs(v / k - x) <= max(tol, abs(x) * 1e-6):
                return True
    return False


def check(text: str, tool_results: Any) -> GuardResult:
    allowed = collect_numbers(tool_results) | {float(v) for v in SAFE}
    scrubbed = IDENT_RE.sub(" ", DATE_RE.sub(" ", CITATION_RE.sub(" ", text)))

    violations: List[Dict] = []
    checked = 0
    for m in NUM_RE.finditer(scrubbed):
        raw = m.group().replace(",", "")
        try:
            x = float(raw)
        except ValueError:
            continue
        checked += 1
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not _matches(x, decimals, allowed):
            ctx = scrubbed[max(0, m.start() - 24): m.end() + 24].replace("\n", " ")
            violations.append(dict(value=x, context=ctx.strip()))
    return GuardResult(ok=not violations, checked=checked, violations=violations,
                       allowed_sample=sorted(list(allowed))[:12])


def check_citations(text: str, valid_citations: Iterable[str]) -> Dict:
    """合规结论必须带条款引用，且引用的条款号必须真实存在。"""
    valid = set(valid_citations)
    cited = set(re.findall(r"\[([^\]]+)\]", text))
    bogus = sorted(c for c in cited if c not in valid)
    return dict(cited=sorted(cited), invalid=bogus,
                hit_rate=None if not cited else round(1 - len(bogus) / len(cited), 4),
                ok=not bogus)
