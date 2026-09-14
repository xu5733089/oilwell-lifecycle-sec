"""动静态储量互校诊断引擎（方案 §5.3）。

这是选题里"将单井产量与储量进行对比"的落点。
刻意不做成简单相除：当动态 EUR 与静态容积法不一致时，
逐项归因并给出可核查的调整建议，判断逻辑全在代码里 —— 可复现、可审计，
大模型只负责把结论说成人话。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _pct(x: float) -> float:
    return round(float(x) * 100.0, 3)


def implied_rf(eur_t: float, ooip_t: float) -> Optional[float]:
    if not ooip_t or ooip_t <= 0 or eur_t is None:
        return None
    return float(eur_t) / float(ooip_t)


def diagnose(eur: Dict[str, float], ooip: Dict[str, float],
             rf_reference: Dict[str, float], dca_params: Dict,
             lateral_length: Optional[float] = None) -> Dict:
    """
    eur:  {'p10','p50','p90'}  动态递减分析结果，单位 t
    ooip: {'p10','p50','p90'}  静态容积法结果，单位 t
    rf_reference: 本区块同类型井的经验采收率区间 {'p90','p50','p10'}，单位 %
    """
    rf50 = implied_rf(eur.get("p50"), ooip.get("p50"))
    findings: List[Dict] = []
    if rf50 is None:
        return dict(consistency="unknown", rf_implied_pct=None, findings=[
            dict(code="NO_DATA", severity="info", finding="缺少 EUR 或 OOIP，无法互校",
                 action="补齐静态参数或延长生产历史后重算")])

    rf_pct = _pct(rf50)
    # p10 是小值（见 block_rf_reference 的口径说明）
    lo = rf_reference.get("low_pct", rf_reference.get("p10"))
    hi = rf_reference.get("high_pct", rf_reference.get("p90"))
    b = float(dca_params.get("b", np.nan))
    d_min = float(dca_params.get("d_min", 0.0) or 0.0)

    if lo is not None and rf_pct < lo:
        consistency = "low"
        findings.append(dict(
            code="RF_TOO_LOW", severity="warn",
            finding=f"隐含采收率 {rf_pct}% 低于区块经验下限 {lo}%",
            reason="静态参数可能偏高（孔隙度/含油饱和度乐观），或该井尚未充分泄油、生产时间不足",
            action="核查静态参数来源与代表性；确认生产历史是否足以支撑外推；检查是否存在长期停井"))
    elif hi is not None and rf_pct > hi:
        consistency = "high"
        findings.append(dict(
            code="RF_TOO_HIGH", severity="warn",
            finding=f"隐含采收率 {rf_pct}% 高于区块经验上限 {hi}%",
            reason="静态参数可能偏低（有效厚度或含油饱和度取值保守），或井控面积取小，或 b 值过大致 EUR 虚高",
            action="复核 b 值与终端递减率；复核井控面积取值；检查邻井干扰是否抬高了产量归属"))
    else:
        consistency = "consistent"
        findings.append(dict(
            code="RF_IN_RANGE", severity="info",
            finding=f"隐含采收率 {rf_pct}% 落在区块经验区间 [{lo}%, {hi}%] 内",
            reason="动静态两条路线相互支持", action="无需调整"))

    # 区间不重叠：往往是层位或时间口径没拉齐，这时不出自动结论
    e_lo, e_hi = eur.get("p10"), eur.get("p90")     # p10 是低估计，见 dca.CONVENTION
    o_lo, o_hi = ooip.get("p10"), ooip.get("p90")
    if None not in (e_lo, e_hi, o_lo, o_hi) and lo is not None and hi is not None:
        if e_lo / max(o_hi, 1e-9) * 100 > hi or e_hi / max(o_lo, 1e-9) * 100 < lo:
            consistency = "conflict"
            findings.append(dict(
                code="INTERVAL_DISJOINT", severity="critical",
                finding="动态与静态的概率区间在经验采收率下无法重叠",
                reason="两条路线的输入很可能不属于同一层位或同一时间口径",
                action="拉齐层位与 as_of 日期；本井标记为「需人工确认」，不出自动结论"))

    if np.isfinite(b) and b > 1.0 and d_min <= 0:
        findings.append(dict(
            code="B_GT_1_NO_DMIN", severity="critical",
            finding=f"递减指数 b={b:.2f} > 1 且未设终端递减率",
            reason="Arps 积分不收敛，EUR 会发散",
            action="设置终端递减率（页岩油常用 6%~8%/年）或把 b 约束到 ≤1 后重算"))

    return dict(consistency=consistency, rf_implied_pct=rf_pct,
                rf_reference_pct=rf_reference, eur=eur, ooip=ooip,
                dca=dict(b=None if not np.isfinite(b) else round(b, 3),
                         d_min_per_month=round(d_min, 5)),
                findings=findings,
                needs_human=consistency in ("conflict",) or
                            any(f["severity"] == "critical" for f in findings))


def block_rf_reference(eurs: np.ndarray, ooips: np.ndarray) -> Dict[str, float]:
    """从已标注老井里统计本区块的经验采收率区间，作为互校的参照。

    口径与 dca.CONVENTION 一致：p10/p50/p90 是分位数本身，p10 是小值。
    另给 low_pct / high_pct 语义别名 —— 采收率的"P90"在储量圈指低值、
    在统计圈指高值，直接用 P-数字讲话必然有人理解反，所以对外一律用语义名。
    """
    ok = np.isfinite(eurs) & np.isfinite(ooips) & (ooips > 0)
    if ok.sum() < 8:
        return dict(p10=1.0, p50=3.0, p90=8.0, low_pct=1.0, mid_pct=3.0, high_pct=8.0,
                    n=int(ok.sum()), source="default")
    rf = eurs[ok] / ooips[ok] * 100.0
    lo = round(float(np.percentile(rf, 10)), 2)
    mid = round(float(np.percentile(rf, 50)), 2)
    hi = round(float(np.percentile(rf, 90)), 2)
    return dict(p10=lo, p50=mid, p90=hi, low_pct=lo, mid_pct=mid, high_pct=hi,
                n=int(ok.sum()), source="observed")
