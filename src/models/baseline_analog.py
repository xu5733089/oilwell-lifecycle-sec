"""基线 A：邻井类比 + Arps（方案 §4.2）。

这是工程师现在的做法，也是"提升了多少"的分母。
没有基线，任何精度数字都没有说服力 —— 所以基线是 P0，不是可选项。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AnalogBaseline:
    targets: List[str]
    k: int = 5
    ref: pd.DataFrame = field(default_factory=pd.DataFrame)   # 训练井：坐标+层位+标签

    def fit(self, master: pd.DataFrame, labels: pd.DataFrame) -> "AnalogBaseline":
        self.ref = master[["well_id", "layer", "x_off", "y_off"]].merge(labels, on="well_id")
        return self

    def predict(self, master: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        rows = {t: [] for t in self.targets}
        for _, w in master.iterrows():
            pool = self.ref[(self.ref["layer"] == w["layer"]) & (self.ref["well_id"] != w["well_id"])]
            if len(pool) < 3:
                pool = self.ref[self.ref["well_id"] != w["well_id"]]
            if pool.empty:
                for t in self.targets:
                    rows[t].append((np.nan, np.nan, np.nan))
                continue
            d = np.hypot(pool["x_off"] - w["x_off"], pool["y_off"] - w["y_off"]).to_numpy()
            nb = pool.iloc[np.argsort(d)[:self.k]]
            for t in self.targets:
                v = nb[t].dropna() if t in nb else pd.Series(dtype=float)
                if v.empty:
                    rows[t].append((np.nan, np.nan, np.nan))
                else:
                    rows[t].append((float(v.quantile(.10)), float(v.median()),
                                    float(v.quantile(.90))))
        return {t: pd.DataFrame(rows[t], columns=["p10", "p50", "p90"],
                                index=master.index) for t in self.targets}
