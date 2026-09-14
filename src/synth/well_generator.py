"""合成井生成器（方案 §3.5）。

用"静态地质参数 -> 潜变量 -> 产量/压力曲线"的物理半经验链条造井：
模型要学的规律是真实存在且已知的，因此评测指标第一周就能跑，
不必等真实数据到位。所有产出打 data_source='SYNTHETIC'。

真值标签（t_peak / q_peak / p_peak / t_oil_break / eur）随数据一并写出，
用于校验 src/labeling 的标签提取逻辑是否正确 —— 这是本仓库的地基测试。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

RHO_OIL = 0.85          # t/m3
BO = 1.15               # 地层体积系数
HALF_SPACING_M = 150.0  # 水平井单侧泄油半宽
Q_ECON_SYNTH = 2.0      # t/d，合成数据的经济极限，仅用于生成 eur 真值
# 上升段累产系数：q(t)=q_peak·(t/t_peak)^0.62 在 [0,t_peak] 上的积分
# = q_peak·t_peak/(1+0.62) = 0.617·q_peak·t_peak。
# 曾经写成 0.31（正好差一半），导致合成数据的 EUR 真值系统性偏低。
RAMP_EXP = 0.62
RAMP_CUM_COEF = 1.0 / (1.0 + RAMP_EXP)      # = 0.617

# 措施的合成物理：增油幅度（相对措施时基础产量的倍数）、增量的月递减率、见效爬坡月数。
# 只服务于合成数据；业务侧的措施类型中文名在 conf/config.yaml 的 sec_unit.measure_types。
MEASURE_PHYSICS = {
    "perforation":  dict(p=0.25, gain=(0.6, 1.4), d=(0.02, 0.05), ramp=0.0),
    "frac":         dict(p=0.20, gain=(1.0, 2.2), d=(0.05, 0.10), ramp=0.0),
    "acid":         dict(p=0.20, gain=(0.3, 0.8), d=(0.08, 0.18), ramp=0.0),
    "workover":     dict(p=0.12, gain=(0.3, 0.7), d=(0.02, 0.05), ramp=0.0),
    "sand_control": dict(p=0.10, gain=(0.2, 0.5), d=(0.02, 0.04), ramp=0.0),
    "waterflood":   dict(p=0.13, gain=(0.2, 0.5), d=(0.01, 0.03), ramp=3.0),
}
LATE_MEASURE_PROB = 0.22     # 没有历史措施的老井，在近两年补一次措施的概率
# 扩边井相对区块中心的偏移：离老井覆盖范围边缘 >= 1500 m。
# 本评估年的扩边井放北侧、上一评估年的放南侧 —— 否则去年扩边井到今年已算老井，
# 会让同一片区今年的扩边井"周边有老井"，与真值标签对不上（业务上那片区已探明，判提采也说得通）。
EXT_Y_OFFSET = (5000.0, 6200.0)

@dataclass
class WellTruth:
    well_id: str
    q_peak: float
    t_peak: float
    p_peak: float
    t_oil_break: float
    eur: float
    ooip: float
    b: float
    di_month: float
    d_min_year: float
    cohort: str = "base"          # base / recent（上一评估年投产）
    is_extension: bool = False    # 位于已探明范围之外的扩边井


def _measure_increment(d: np.ndarray, wo_day: int, q_base_at: float, gain: float,
                       d_inc: float, ramp: float) -> np.ndarray:
    """措施增油（叠加在基础产量之上）：措施时基础产量 × 增油倍数，按自身递减率衰减。"""
    m = np.maximum(d - wo_day, 0.0) / 30.4
    shape = np.exp(-d_inc * m)
    if ramp > 0:
        shape = shape * (1.0 - np.exp(-m / ramp))
    return np.where(d >= wo_day, q_base_at * gain * shape, 0.0)


def _arps(t_month: np.ndarray, qi: float, b: float, di: float, d_min: float) -> np.ndarray:
    """修正双曲：达到终端递减率 d_min 后切为指数递减。

    b > 1 且不设 d_min 时 Arps 积分不收敛（EUR -> 无穷），
    这是储量评估最常见的错误来源，见 src/reserves/dca.py 的断言。
    """
    q = np.empty_like(t_month, dtype=float)
    if b <= 1e-6:
        return qi * np.exp(-di * t_month)
    # 双曲段瞬时递减率 D(t) = di / (1 + b*di*t)，求切换时刻
    t_switch = (di / d_min - 1.0) / (b * di) if d_min > 0 else np.inf
    hyp = t_month <= t_switch
    q[hyp] = qi / np.power(1.0 + b * di * t_month[hyp], 1.0 / b)
    if (~hyp).any():
        q_sw = qi / np.power(1.0 + b * di * t_switch, 1.0 / b)
        q[~hyp] = q_sw * np.exp(-d_min * (t_month[~hyp] - t_switch))
    return q


def _geo_field(centers, x: float, y: float) -> float:
    """平滑地质趋势场：几个"甜点中心"的高斯叠加。

    有了空间自相关，回归克里金的变差函数才有结构可拟合，
    空间邻井特征也才真的携带信息 —— 否则块金效应接近基台值，插值退化成取均值。
    """
    v = 0.0
    for cx, cy, amp, rad in centers:
        v += amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * rad ** 2)))
    return float(np.clip(v, -1.5, 1.5))


def _sample_static(rng: np.random.Generator, trend: float = 0.0) -> Dict[str, float]:
    phi = float(np.clip(rng.normal(8.0 + 1.9 * trend, 1.15), 3.5, 13.0))          # %
    so = float(np.clip(rng.normal(60.0 + 6.0 * trend, 4.5), 40.0, 78.0))          # %
    h = float(np.clip(rng.lognormal(np.log(13.0) + 0.22 * trend, 0.26), 4.0, 34.0))  # m
    perm = float(np.clip(rng.lognormal(np.log(0.35) + 0.55 * trend, 0.80), 0.008, 6.0))  # mD
    toc = float(np.clip(rng.normal(3.0 + 0.85 * trend, 0.62), 0.8, 6.0))            # %
    tvd = float(rng.uniform(1800, 2600))
    pcoef = float(np.clip(rng.normal(1.25, 0.12), 0.95, 1.65))
    sweet = float(np.clip(
        0.30 * (phi - 3.5) / 9.5 + 0.25 * (so - 40) / 38 +
        0.25 * (toc - 0.8) / 5.2 + 0.20 * (h - 4) / 30 + rng.normal(0, 0.05),
        0.0, 1.0))
    return dict(porosity_pct=phi, so_pct=so, sw_pct=100.0 - so, net_pay_m=h,
                perm_md=perm, toc_pct=toc, tvd=tvd, pressure_coef=pcoef,
                sweet_spot_idx=sweet,
                brittleness=float(np.clip(rng.normal(55, 8), 30, 80)),
                temp_c=float(60 + tvd * 0.028 + rng.normal(0, 2)))


def _sample_completion(rng: np.random.Generator, st: Dict[str, float]) -> Dict[str, float]:
    horizontal = rng.random() < 0.78
    if horizontal:
        lateral = float(rng.uniform(800, 2200))
        stages = int(np.clip(round(lateral / rng.uniform(60, 85)), 6, 40))
    else:
        lateral, stages = 0.0, int(np.clip(round(rng.uniform(1, 5)), 1, 6))
    proppant = float(stages * rng.uniform(40, 95))
    return dict(well_type="水平井" if horizontal else "直井",
                lateral_length=lateral, stage_count=stages,
                proppant_t=proppant, frac_fluid_m3=float(proppant * rng.uniform(11, 19)),
                md=float(st["tvd"] + lateral + rng.uniform(50, 250)))


def _latent(rng: np.random.Generator, st: Dict[str, float], cp: Dict[str, float]) -> Dict[str, float]:
    """静态参数 -> 曲线潜变量。这就是模型要从数据里学回来的那套映射。"""
    kh = st["perm_md"] * st["net_pay_m"]
    q_peak = (3.4 * kh ** 0.34
              * (1.0 + 0.00055 * cp["lateral_length"])
              * (st["porosity_pct"] / 8.0) ** 0.55
              * (st["so_pct"] / 60.0) ** 0.9
              * (st["pressure_coef"] / 1.25) ** 1.6
              * (1.0 + 0.25 * st["sweet_spot_idx"])
              * float(rng.lognormal(0, 0.16)))
    q_peak = float(np.clip(q_peak, 3.0, 45.0))

    t_peak = (118.0 * st["perm_md"] ** -0.17
              * (1.0 + 0.00016 * cp["lateral_length"])
              * (1.0 - 0.20 * st["sweet_spot_idx"])
              * float(rng.lognormal(0, 0.19)))
    t_peak = float(np.clip(t_peak, 55.0, 380.0))

    t_break = float(np.clip(4.0 + 26.0 * np.exp(-kh / 2.2) * rng.lognormal(0, 0.22), 2.0, 55.0))

    b = float(np.clip(rng.normal(0.85, 0.18) + 0.10 * st["sweet_spot_idx"], 0.35, 1.35))
    di = float(np.clip(0.085 + 0.16 / (1.0 + kh) + rng.normal(0, 0.015), 0.05, 0.32))  # /月
    d_min = float(np.clip(rng.normal(0.075, 0.012), 0.04, 0.12))                        # /年
    return dict(q_peak=q_peak, t_peak=t_peak, t_oil_break=t_break,
                b=b, di_month=di, d_min_year=d_min)


def _ooip_t(st: Dict[str, float], cp: Dict[str, float]) -> float:
    if cp["lateral_length"] > 0:
        area = cp["lateral_length"] * 2.0 * HALF_SPACING_M
    else:
        area = np.pi * 200.0 ** 2
    pv = area * st["net_pay_m"] * (st["porosity_pct"] / 100.0) * (st["so_pct"] / 100.0)
    return float(pv * RHO_OIL / BO)


def _daily_series(rng, lat, hist_days, first_prod: date, aux: np.random.Generator):
    """生成日度产量/压力序列，并返回真值 p_peak 与 eur。"""
    d = np.arange(1, hist_days + 1, dtype=float)
    tm = d / 30.4

    # 上升段：幂律爬坡到峰值；峰后走修正双曲
    tp_m = lat["t_peak"] / 30.4
    q = np.where(d <= lat["t_peak"],
                 lat["q_peak"] * np.power(np.maximum(d, 0.5) / lat["t_peak"], RAMP_EXP),
                 _arps(np.maximum(tm - tp_m, 0.0), lat["q_peak"], lat["b"],
                       lat["di_month"], lat["d_min_year"] / 12.0))
    # 见油前基本不出油
    ramp = 1.0 / (1.0 + np.exp(-(d - lat["t_oil_break"]) / 2.0))
    q = q * ramp

    # 措施井：30% 概率在 400 天后有一次增产作业（抽取顺序与旧版一致，主随机数流不变）；
    # 没有措施的老井再以 LATE_MEASURE_PROB 在近两年补一次（独立随机数流 aux）。
    events: List[Dict] = []
    ev_truth: List[Dict] = []
    inc = np.zeros_like(d)
    wo_day = None
    if hist_days > 500 and rng.random() < 0.30:
        wo_day = int(rng.uniform(400, min(hist_days - 90, 1500)))
        rng.uniform(0.20, 0.65), rng.uniform(45, 110)           # 旧版脉冲参数：只保留抽取
        rng.choice(["frac", "acid", "pump_change"])
    elif hist_days > 500 and aux.random() < LATE_MEASURE_PROB:
        wo_day = int(np.clip(hist_days - aux.uniform(90, 600), 400, hist_days - 90))
    if wo_day is not None:
        kinds = list(MEASURE_PHYSICS)
        kind = str(aux.choice(kinds, p=[MEASURE_PHYSICS[k]["p"] for k in kinds]))
        spec = MEASURE_PHYSICS[kind]
        gain, d_inc = float(aux.uniform(*spec["gain"])), float(aux.uniform(*spec["d"]))
        q_base_at = float(q[wo_day - 1])
        inc = _measure_increment(d, wo_day, q_base_at, gain, d_inc, spec["ramp"])
        q = q + inc
        events.append(dict(day_index=wo_day, event_type=kind, note="合成措施"))
        ev_truth.append(dict(day_index=wo_day, event_type=kind, gain=gain, d_inc_month=d_inc,
                             ramp_month=spec["ramp"], q_base_t_per_d=q_base_at))

    noise = rng.lognormal(0.0, 0.065, size=q.shape)          # 计量噪声
    q = np.maximum(q * noise, 0.0)

    # 停井段
    hours = np.full_like(d, 24.0)
    for _ in range(int(rng.integers(0, 4))):
        s = int(rng.uniform(30, max(31, hist_days - 20)))
        e = min(hist_days, s + int(rng.uniform(3, 16)))
        hours[s:e] = 0.0
    q = np.where(hours > 0, q, 0.0)

    degraded = rng.random() < 0.05      # 5% 劣质井：长时间断记录，用于验证质量门禁确实在拦人
    if degraded and hist_days > 400:
        s = int(rng.uniform(100, hist_days - 200))
        drop = np.arange(s, min(hist_days, s + int(rng.uniform(60, 200))))
    else:
        drop = np.array([], dtype=int)

    # 压力：井口压力随采出程度衰竭
    cum = np.cumsum(q)
    p_init = lat["p_init"]
    depl = np.clip(cum / max(lat["eur_ref"], 1.0), 0.0, 1.0)
    whp = p_init * (1.0 - 0.62 * depl) * rng.lognormal(0.0, 0.030, size=d.shape)
    whp = np.maximum(whp, 0.8)
    bhp = whp * rng.uniform(1.9, 2.4) + rng.normal(0, 0.2, size=d.shape)

    wc = 100.0 * (0.05 + 0.80 / (1.0 + np.exp(-(d - rng.uniform(500, 1400)) / 260.0)))
    wc = np.clip(wc + rng.normal(0, 1.5, size=d.shape), 1.0, 96.0)
    water = q * wc / np.maximum(100.0 - wc, 1.0) / RHO_OIL
    gor = np.clip(rng.normal(110, 18) + rng.normal(0, 6, size=d.shape), 20, 400)

    dates = [first_prod + timedelta(days=int(k) - 1) for k in d]
    df = pd.DataFrame({
        "dt": [x.isoformat() for x in dates],
        "day_index": d.astype(int),
        "oil_t": np.round(q, 3),
        "water_m3": np.round(water, 2),
        "gas_m3": np.round(q * gor, 1),
        "whp_mpa": np.round(whp, 3),
        "bhp_mpa": np.round(bhp, 3),
        "choke_mm": np.round(np.full_like(d, rng.uniform(4, 10)), 1),
        "hours_on": hours,
        "water_cut": np.round(wc, 2),
        "gor": np.round(gor, 1),
    })
    # 2% 随机缺失，模拟真实数据的不完整
    miss = rng.random(len(df)) < 0.02
    df.loc[miss, ["whp_mpa", "bhp_mpa"]] = np.nan
    if len(drop):
        df = df.drop(index=drop[drop < len(df)]).reset_index(drop=True)   # 整段记录丢失

    # 真值：达峰压力取真值峰日的 WHP；EUR 用长周期外推到经济极限
    peak_idx = int(np.clip(round(lat["t_peak"]) - 1, 0, len(df) - 1))
    p_peak = float(np.nanmean(whp[max(0, peak_idx - 1):peak_idx + 2]))

    # 措施增油真值：只统计真正落进记录的那部分（经过计量噪声、停井、丢记录之后）
    inc_monthly: List[Dict] = []
    if ev_truth:
        kept = np.ones(len(d), dtype=bool)
        if len(drop):
            kept[drop[drop < len(d)]] = False
        realized = np.where((hours > 0) & kept, inc * noise, 0.0)
        ev_truth[0]["realized_inc_t"] = float(realized.sum())
        ym = pd.Series(realized, index=[x.isoformat()[:7] for x in dates])
        g = ym[ym > 0].groupby(level=0).sum()
        inc_monthly = [dict(ym=k, inc_oil_t=float(v)) for k, v in g.items()]
    return df, events, p_peak, ev_truth, inc_monthly


def _eur_truth(lat: Dict[str, float]) -> float:
    """把修正双曲积分到经济极限，作为 EUR 真值（不受历史长度限制）。"""
    tm = np.arange(0, 12 * 40, 0.5) / 1.0  # 40 年，半月步长
    tm = np.arange(0, 480, 0.5)
    q = _arps(tm, lat["q_peak"], lat["b"], lat["di_month"], lat["d_min_year"] / 12.0)
    econ = q >= Q_ECON_SYNTH
    if not econ.any():
        return float(lat["q_peak"] * lat["t_peak"] * 0.5)
    q_e = q[econ]
    return float(np.trapezoid(q_e, dx=0.5) * 30.4
                 + lat["q_peak"] * lat["t_peak"] * RAMP_CUM_COEF)


def _build_well(i: int, rng: np.random.Generator, aux: np.random.Generator, cohort: str,
                blocks: List[str], layers: List[str], block_origin: Dict, field: Dict,
                end: date, extension_frac: float) -> Dict:
    """造一口井。cohort=base 时主随机数流的抽取顺序与旧版逐项一致。"""
    wid = f"SYN{i + 1:04d}"
    block = str(rng.choice(blocks))
    layer = str(rng.choice(layers))
    ox, oy = block_origin[block]
    x_off = float(ox + rng.uniform(-3500, 3500))
    y_off = float(oy + rng.uniform(-3500, 3500))
    st = _sample_static(rng, trend=_geo_field(field[block], x_off, y_off))
    cp = _sample_completion(rng, st)
    lat = _latent(rng, st, cp)
    lat["p_init"] = float(st["pressure_coef"] * st["tvd"] * 0.00981 * rng.uniform(0.42, 0.55))
    lat["eur_ref"] = _eur_truth(lat)

    if cohort == "base":
        is_new = rng.random() < 0.25
        if is_new:
            rng.uniform(0, 210)                        # 旧版投产日抽取，保留以稳定随机数流
            hist = int(rng.uniform(100, 210))
            status = "producing"
        else:
            rng.uniform(0, 3000)
            hist = int(rng.uniform(900, 2300))
            status = "producing" if rng.random() < 0.85 else "shut_in"
    else:                                              # recent：上一评估年内投产
        is_new = True
        first = date(end.year - 1, 1, 1) + timedelta(days=int(rng.uniform(0, 365)))
        hist = (end - first).days + 1
        status = "producing"

    # 日期对齐：在产井的最后一条记录落在数据截止日；关停井在截止日前若干天停产。
    # 只平移日期，不改产量序列 —— 单元级汇总需要一个公共时点。
    stop = end - timedelta(days=int(aux.uniform(60, 400))) if status == "shut_in" else end
    first_prod = stop - timedelta(days=hist - 1)

    # 扩边井：挪到区块已探明范围之外（北侧），周边没有老井
    is_ext = bool(is_new and aux.random() < extension_frac)
    if is_ext:
        x_off = float(ox + aux.uniform(-2500, 2500))
        y_off = float(oy + (1.0 if cohort == "base" else -1.0) * aux.uniform(*EXT_Y_OFFSET))

    pdf, evs, p_peak, ev_truth, inc_monthly = _daily_series(rng, lat, hist, first_prod, aux)
    pdf.insert(0, "well_id", wid)

    master = dict(
        well_id=wid, well_code_anon=f"{block}-{i + 1:04d}", block=block, layer=layer,
        well_type=cp["well_type"],
        spud_date=(first_prod - timedelta(days=int(rng.uniform(60, 160)))).isoformat(),
        completion_date=(first_prod - timedelta(days=int(rng.uniform(10, 45)))).isoformat(),
        first_prod_date=first_prod.isoformat(),
        x_off=x_off, y_off=y_off,
        tvd=st["tvd"], md=cp["md"], lateral_length=cp["lateral_length"],
        stage_count=cp["stage_count"], proppant_t=cp["proppant_t"],
        frac_fluid_m3=cp["frac_fluid_m3"], status=status, data_source="SYNTHETIC")
    static = dict(well_id=wid, layer=layer, toc_pct=st["toc_pct"],
                  porosity_pct=st["porosity_pct"], perm_md=st["perm_md"],
                  so_pct=st["so_pct"], sw_pct=st["sw_pct"],
                  net_pay_m=st["net_pay_m"], sweet_spot_idx=st["sweet_spot_idx"],
                  brittleness=st["brittleness"], pressure_coef=st["pressure_coef"],
                  temp_c=st["temp_c"])
    events = []
    for e in evs:
        e_date = first_prod + timedelta(days=e["day_index"] - 1)
        events.append(dict(well_id=wid, dt=e_date.isoformat(), **e))
    for t in ev_truth:
        t.update(well_id=wid, dt=(first_prod + timedelta(days=t["day_index"] - 1)).isoformat())
    truth = asdict(WellTruth(
        well_id=wid, q_peak=lat["q_peak"], t_peak=lat["t_peak"], p_peak=p_peak,
        t_oil_break=lat["t_oil_break"], eur=lat["eur_ref"],
        ooip=_ooip_t(st, cp), b=lat["b"], di_month=lat["di_month"],
        d_min_year=lat["d_min_year"], cohort=cohort, is_extension=is_ext))
    return dict(master=master, static=static, prod=pdf, events=events, truth=truth,
                event_truth=ev_truth,
                inc_monthly=[dict(well_id=wid, **r) for r in inc_monthly])


def generate(n_wells: int, seed: int, blocks: List[str], layers: List[str],
             data_end: str = "2026-12-31", recent_frac: float = 0.10,
             extension_frac: float = 0.25) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = date.fromisoformat(str(data_end))

    # 每个区块先摆几个"甜点中心"，构成平滑地质趋势场
    block_origin = {b: (k * 9000.0, k * 4000.0) for k, b in enumerate(blocks)}
    field = {}
    for b, (ox, oy) in block_origin.items():
        field[b] = [(ox + rng.uniform(-2600, 2600), oy + rng.uniform(-2600, 2600),
                     rng.uniform(0.7, 1.5) * rng.choice([1.0, -1.0]), rng.uniform(1100, 2400))
                    for _ in range(3)]

    wells: List[Dict] = []
    common = dict(blocks=blocks, layers=layers, block_origin=block_origin, field=field,
                  end=end, extension_frac=extension_frac)
    for i in range(n_wells):
        wells.append(_build_well(i, rng, np.random.default_rng([seed, i, 17]), "base", **common))
    # 上一评估年投产的批次用独立随机数流追加在后面，不扰动前面的井
    rng_recent = np.random.default_rng([seed, 99991])
    for j in range(int(round(n_wells * recent_frac))):
        i = n_wells + j
        wells.append(_build_well(i, rng_recent, np.random.default_rng([seed, i, 17]), "recent",
                                 **common))

    events = [e for w in wells for e in w["events"]]
    ev_cols = ["well_id", "dt", "day_index", "event_type", "note"]
    return {
        "well_master": pd.DataFrame([w["master"] for w in wells]),
        "geo_static": pd.DataFrame([w["static"] for w in wells]),
        "prod_daily": pd.concat([w["prod"] for w in wells], ignore_index=True),
        "well_event": pd.DataFrame(events, columns=ev_cols),
        "truth": pd.DataFrame([w["truth"] for w in wells]),
        # 以下两项是真值，只落文件供测试与评测，不进业务表
        "event_truth": pd.DataFrame([t for w in wells for t in w["event_truth"]]),
        "measure_inc_monthly": pd.DataFrame([r for w in wells for r in w["inc_monthly"]],
                                            columns=["well_id", "ym", "inc_oil_t"]),
    }
