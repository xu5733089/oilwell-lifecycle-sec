"""北达科他州公开产量数据适配器（NDIC 矿产资源部月度产量报表），只供递减模型回测。

    python -m src.cli eval-dca --dataset ndic [--max-wells 800]

数据：https://www.dmr.nd.gov/oilgas/mpr/YYYY_MM.xlsx（公开记录，2018-01 起），每月一个 Excel，
逐井给出油 / 水 / 气产量（桶、千立方英尺）、生产天数、层位（Pool）、经纬度。

合规与口径：
  · 下载与解析结果只落在 data/raw/ndic/（已被 .gitignore 忽略），**不进版本库**；
  · 井号经 SHA-1 不可逆哈希成 "ND-xxxxxxxx"，井名、公司名、经纬度一律不读 —— 回测产物里不出现任何真实标识；
  · 只取 Bakken / Three Forks 两套致密油层；只取 2018-02 以后首次出油的井（2018-01 已在产的井看不到投产期，排除）；
  · 产量换算：1 bbl = 0.158987 m³，Bakken 轻质原油密度取 0.82 t/m³；开井日均 = 月产油 ÷ 生产天数；
  · 报表缺失的月份（例如 2023-09）按"缺测"处理，不当作零产量；
  · 峰值取投产后前 4 个出油月中的最高开井日均，回测序列从峰值月起算；至少要有 36 个出油月。
真实井没有 EUR 真值，回测只比未来 12 / 24 个月产量。
"""
from __future__ import annotations

import hashlib
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import path

URL = "https://www.dmr.nd.gov/oilgas/mpr/{ym}.xlsx"
FIRST_YM = "2018_01"
BBL_TO_T = 0.158987 * 0.82
POOLS = {"BAKKEN": "Bakken", "THREE FORKS": "Three Forks"}
Q_ECON_T_PER_D = 1.0            # 仅用于 EUR 外推（真实井无真值，回测不比 EUR）
MIN_PRODUCING_MONTHS = 36
PEAK_WINDOW = 4
_USECOLS = ["ReportDate", "API_WELLNO", "Pool", "Oil", "Days"]


def data_dir() -> Path:
    d = path("data_dir") / "raw" / "ndic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def month_keys(first: str = FIRST_YM, last: Optional[str] = None) -> List[str]:
    last = last or pd.Timestamp.today().strftime("%Y_%m")
    return [p.strftime("%Y_%m") for p in pd.period_range(first.replace("_", "-"), last.replace("_", "-"), freq="M")]


def download(first: str = FIRST_YM, last: Optional[str] = None) -> Dict[str, int]:
    got, missing = 0, 0
    for ym in month_keys(first, last):
        f = data_dir() / f"{ym}.xlsx"
        if f.exists() and f.stat().st_size > 0:
            continue
        try:
            with urllib.request.urlopen(URL.format(ym=ym), timeout=180) as r:
                f.write_bytes(r.read())
            got += 1
        except Exception:
            missing += 1
    return dict(downloaded=got, missing=missing)


def anon_key(api_wellno) -> str:
    return "ND-" + hashlib.sha1(str(int(api_wellno)).encode()).hexdigest()[:8]


def parse_month(f: Path) -> pd.DataFrame:
    df = pd.read_excel(f, sheet_name="Oil", usecols=_USECOLS)
    df = df[df["Pool"].astype(str).str.upper().isin(POOLS)]
    return pd.DataFrame(dict(well=[anon_key(x) for x in df["API_WELLNO"]],
                             pool=df["Pool"].astype(str).str.upper().map(POOLS).to_numpy(),
                             ym=pd.to_datetime(df["ReportDate"]).dt.strftime("%Y-%m").to_numpy(),
                             oil_t=df["Oil"].fillna(0.0).to_numpy(float) * BBL_TO_T,
                             days=df["Days"].fillna(0.0).to_numpy(float)))


def load_monthly(refresh: bool = False) -> pd.DataFrame:
    """全部月份的长表（匿名井号、层位、年月、产油吨、生产天数），解析结果缓存为本地 pickle。"""
    files = sorted(data_dir().glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("data/raw/ndic 下没有月度报表，请先下载（ndic_public.download()）")
    cache = data_dir() / "_parsed.pkl"
    sig = [(f.name, f.stat().st_size) for f in files]
    if cache.exists() and not refresh:
        obj = pickle.loads(cache.read_bytes())
        if obj.get("sig") == sig:
            return obj["df"]
    df = pd.concat([parse_month(f) for f in files], ignore_index=True)
    cache.write_bytes(pickle.dumps(dict(sig=sig, df=df)))
    return df


def build_series(df: pd.DataFrame, available_yms: List[str]) -> List[Dict]:
    """逐井月度序列：峰值月起算；缺失报表的月份留空（不当零），生产天数为 0 的月份只计产量不计开井日均。"""
    first_month = min(available_yms)
    all_months = pd.period_range(first_month, max(available_yms), freq="M").strftime("%Y-%m").tolist()
    pos = {m: i for i, m in enumerate(all_months)}
    have = np.zeros(len(all_months), bool)
    for m in available_yms:
        have[pos[m]] = True
    wells = []
    for (well, pool), g in df.groupby(["well", "pool"]):
        g = g.groupby("ym", as_index=False)[["oil_t", "days"]].sum()
        g = g[g["oil_t"] > 0]
        if g.empty or g["ym"].min() <= first_month:
            continue
        idx = g["ym"].map(pos).to_numpy(int)
        oil = np.zeros(len(all_months)); days = np.zeros(len(all_months))
        oil[idx], days[idx] = g["oil_t"].to_numpy(), g["days"].to_numpy()
        start = int(idx.min())
        on = (days > 0) & (oil > 0)
        if int(on[start:].sum()) < MIN_PRODUCING_MONTHS:
            continue
        rate = np.where(on, oil / np.maximum(days, 1.0), np.nan)
        prod_months = np.flatnonzero(on[start:])[:PEAK_WINDOW] + start
        peak = int(prod_months[np.nanargmax(rate[prod_months])])
        k = np.arange(peak, len(all_months))
        ok = on[k] & have[k]
        wells.append(dict(well_id=well, well_code=well, block=pool, order_key=all_months[start],
                          first_prod_ym=all_months[start], months=(k[ok] - peak + 0.5).astype(float),
                          rates=rate[k][ok], volumes=oil[peak:], cum_before_peak_t=float(oil[start:peak].sum())))
    return wells


def hindcast_wells(max_wells: Optional[int] = 800, seed: int = 20260916) -> List[Dict]:
    df = load_monthly()
    yms = sorted(df["ym"].unique().tolist())
    wells = build_series(df, yms)
    if max_wells and len(wells) > max_wells:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(len(wells), size=max_wells, replace=False).tolist())
        wells = [w for i, w in enumerate(wells) if i in keep]
    return wells
