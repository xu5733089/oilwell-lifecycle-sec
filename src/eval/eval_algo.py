"""算法层评测：切分方式对比（方案 §4.7）。

把同一套数据、同一个模型在三种切分下并排跑出来，
是回答"你怎么保证不是过拟合"最有力的一张表。

**结论由数据算出来，不预先写死。** 随机切分是否虚高、虚高多少，
取决于数据里到底有多少可泄漏的结构；在按井独立抽样的合成数据上，
泄漏本就弱于真实井网（真实区块有大量加密井，邻井相关性强得多）。
把预设结论写进代码，等于让报告去迁就一句口号。

    python -m src.cli eval-algo
"""
from __future__ import annotations

import json
from typing import Dict, List

from ..config import path
from ..pipeline import run_training

SPLITS = ("time", "group", "random")
NOTE = {
    "time": "按投产年份外推：训练早期井、测试近期井。最贴近真实使用场景，本项目的正式口径。",
    "group": "按区块分组：整个区块留出。检验模型能否推广到没见过的区块。",
    "random": "随机切分：**仅作对照**。同平台邻井被拆到两边，指标虚高，不得对外引用。",
}


def run(obs_days: int | None = None) -> Dict:
    rows: List[Dict] = []
    for split in SPLITS:
        meta = run_training(obs_days=obs_days, split_method=split)
        for tgt, r in meta["report"].items():
            rows.append(dict(split=split, target=tgt,
                             mae=round(r["model"]["mae"], 4),
                             medae=round(r["model"]["medae"], 4),
                             mape=round(r["model"]["mape"], 2),
                             coverage=round(r["model"]["coverage"], 4),
                             baseline_mae=round(r["baseline"]["mae"], 4),
                             gain_pct=r["mae_gain_vs_baseline_pct"],
                             n_test=r["model"]["n"]))

    # 随机切分相对时间切分的"虚高幅度"：这个数越大，越说明泄漏是真实存在的
    inflation: Dict[str, float] = {}
    for tgt in {r["target"] for r in rows}:
        t = next((r["mae"] for r in rows if r["split"] == "time" and r["target"] == tgt), None)
        rd = next((r["mae"] for r in rows if r["split"] == "random" and r["target"] == tgt), None)
        if t and rd:
            inflation[tgt] = round(100 * (1 - rd / t), 1)

    n_opt = sum(1 for v in inflation.values() if v > 0)
    mean_opt = round(sum(inflation.values()) / len(inflation), 1) if inflation else 0.0
    conclusion = (
        f"随机切分在 {n_opt}/{len(inflation)} 个目标上给出更低的 MAE，"
        f"平均乐观偏差 {mean_opt:+.1f}%。"
        "本数据集按井独立抽样（仅静态参数带空间趋势），可泄漏的结构有限；"
        "真实井网中加密井密集、邻井相关性强，该偏差通常明显更大。"
        "无论偏差大小，对外报告一律使用时间外推切分的指标 —— "
        "因为上线时要预测的就是未来的新井，随机切分不对应任何真实使用场景。")
    out = dict(rows=rows, notes=NOTE, random_split_optimism_pct=inflation,
               n_targets_optimistic=n_opt, mean_optimism_pct=mean_opt,
               conclusion=conclusion)
    p = path("artifacts_dir") / "eval_algo.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out["report_path"] = str(p)
    return out
