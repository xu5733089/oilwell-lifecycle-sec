"""训练与评测主流水线（方案 §4.7 验证协议）。

切分方式刻意默认 time（按投产年份外推），不用随机切分：
随机切分会让指标虚高，也不符合真实使用场景 —— 我们要预测的是未来的新井。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import db
from .config import config, label_def_version
from .features.build import build, encode, feature_columns
from .labeling.labels import extract_all
from .models.analog_retrieval import AnalogIndex, _resample_curve
from .models.baseline_analog import AnalogBaseline
from .models.conformal import Conformal, calibrate
from .models.gbdt_quantile import QuantileModel
from .models import registry
from .quality.gate import check_wells, summary as quality_summary


def load_tables() -> Dict[str, pd.DataFrame]:
    return {
        "master": db.read_df("SELECT * FROM well_master"),
        "static": db.read_df("SELECT * FROM geo_static"),
        "prod": db.read_df("SELECT * FROM prod_daily"),
        "events": db.read_df("SELECT * FROM well_event"),
    }


MIN_LIFECYCLE_DAYS = 420        # 训练/评测样本的最短生产历史


def eligible_wells(prod: pd.DataFrame, labels: pd.DataFrame,
                   min_days: int = MIN_LIFECYCLE_DAYS) -> List[str]:
    """训练与回测只用"已走完主要生产周期、标签可信"的老井。

    投产不足半年的新井，其"达峰"标签是假的（峰还没到），
    把它们放进评测集会让指标失去意义 —— 它们是上线后的预测对象，不是评测样本。
    这与申报书承诺的回测方式一致：用老井，只喂前 3 个月数据，与实际值比对。
    """
    hist = prod.groupby("well_id")["day_index"].max()
    long_enough = set(hist[hist >= min_days].index)
    ok = set(labels.loc[labels["label_quality"] == "ok", "well_id"])
    return sorted(long_enough & ok)


def split_wells(master: pd.DataFrame, labels: pd.DataFrame, method: str,
                calib_ratio: float, test_ratio: float, seed: int = 0,
                eligible: List[str] | None = None
                ) -> Tuple[List[str], List[str], List[str]]:
    ok = (labels.loc[labels["label_quality"] != "rejected", "well_id"]
          if eligible is None else pd.Series(eligible))
    m = master[master["well_id"].isin(set(ok))].copy()
    if method == "time":
        m = m.sort_values("first_prod_date")
    elif method == "group":                      # 按区块分组，防同平台井互相泄漏
        m = m.sort_values(["block", "first_prod_date"])
        blocks = list(dict.fromkeys(m["block"]))
        hold = blocks[-1:]
        test = m[m["block"].isin(hold)]["well_id"].tolist()
        rest = m[~m["block"].isin(hold)]
        n_cal = int(len(rest) * calib_ratio)
        return (rest["well_id"].tolist()[:-n_cal or None],
                rest["well_id"].tolist()[-n_cal:] if n_cal else [], test)
    else:
        m = m.sample(frac=1.0, random_state=seed)
    n = len(m)
    n_test = int(n * test_ratio)
    n_cal = int(n * calib_ratio)
    ids = m["well_id"].tolist()
    return ids[: n - n_cal - n_test], ids[n - n_cal - n_test: n - n_test], ids[n - n_test:]


def metrics_for(y: pd.Series, pred: pd.DataFrame) -> Dict[str, float]:
    ok = y.notna().to_numpy() & pred["p50"].notna().to_numpy()
    nan = float("nan")
    if ok.sum() == 0:
        return dict(n=0, mae=nan, medae=nan, mape=nan, coverage=nan, interval_width=nan)
    yt = y.to_numpy(float)[ok]
    p50 = pred["p50"].to_numpy(float)[ok]
    lo, hi = pred["p10"].to_numpy(float)[ok], pred["p90"].to_numpy(float)[ok]
    err = np.abs(p50 - yt)
    return dict(
        n=int(ok.sum()),
        mae=float(err.mean()),
        medae=float(np.median(err)),
        mape=float(np.mean(err / np.maximum(np.abs(yt), 1e-6)) * 100),
        coverage=float(((yt >= lo) & (yt <= hi)).mean()),
        interval_width=float(np.mean(hi - lo)),
    )


def run_training(obs_days: int | None = None, split_method: str | None = None,
                 seed: int = 42) -> Dict:
    cfg = config()
    mcfg = cfg["model"]
    obs_days = obs_days or mcfg["obs_days"]
    split_method = split_method or mcfg["split"]["method"]

    t = load_tables()
    qrep = check_wells(t["prod"], t["master"], t["static"])
    good = set(qrep.loc[qrep["passed"], "well_id"])
    prod = t["prod"][t["prod"]["well_id"].isin(good)]
    master = t["master"][t["master"]["well_id"].isin(good)].reset_index(drop=True)

    labels = extract_all(prod, t["events"])
    db.replace_rows(labels, "lifecycle_label", ["well_id", "label_def_version"])

    elig = eligible_wells(prod, labels)
    tr_ids, cal_ids, te_ids = split_wells(master, labels, split_method,
                                          mcfg["split"]["calib_ratio"],
                                          mcfg["split"]["test_ratio"], seed,
                                          eligible=elig)
    train_labels = labels[labels["well_id"].isin(tr_ids)]

    # 空间邻井特征只从训练井取邻居，杜绝通过邻居标签泄漏答案
    X = encode(build(prod, master, t["static"], obs_days, ref_labels=train_labels))
    X = X[X["well_id"].isin(set(tr_ids) | set(cal_ids) | set(te_ids))].reset_index(drop=True)
    feat_cols = feature_columns(X)
    Y = X[["well_id"]].merge(labels, on="well_id", how="left").set_index(X.index)

    part = {"train": tr_ids, "calib": cal_ids, "test": te_ids}
    idx = {k: X.index[X["well_id"].isin(v)] for k, v in part.items()}
    targets = mcfg["targets"]

    model = QuantileModel(targets=targets, quantiles=mcfg["quantiles"],
                          feature_cols=feat_cols, seed=seed)
    model.fit(X.loc[idx["train"]], Y.loc[idx["train"]])

    # 保形校准
    conf = Conformal(alpha=mcfg["conformal_alpha"])
    if len(idx["calib"]):
        pc = model.predict(X.loc[idx["calib"]])
        for tgt, dfp in pc.items():
            conf.fit_target(tgt, Y.loc[idx["calib"], tgt].to_numpy(float),
                            dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))

    # 基线 A：邻井类比
    base = AnalogBaseline(targets=targets).fit(
        master[master["well_id"].isin(tr_ids)], train_labels)

    # 测试集评估
    Xte, Yte = X.loc[idx["test"]], Y.loc[idx["test"]]
    pred = model.predict(Xte)
    pred_cal = {}
    for tgt, dfp in pred.items():
        lo, hi = conf.apply(tgt, dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))
        pred_cal[tgt] = pd.DataFrame({"p10": lo, "p50": dfp["p50"].to_numpy(float),
                                      "p90": hi}, index=dfp.index)
    base_pred = base.predict(master.set_index("well_id").loc[Xte["well_id"]].reset_index())
    for k in base_pred:
        base_pred[k].index = Xte.index

    report = {}
    for tgt in targets:
        if tgt not in pred_cal or tgt not in Yte:
            continue
        m_raw = metrics_for(Yte[tgt], pred[tgt])
        m_cal = metrics_for(Yte[tgt], pred_cal[tgt])
        m_base = metrics_for(Yte[tgt], base_pred[tgt]) if tgt in base_pred else dict(n=0)
        gain = (100 * (1 - m_cal["mae"] / m_base["mae"])
                if m_base["n"] and m_cal["n"] and np.isfinite(m_base["mae"]) else None)
        report[tgt] = dict(model=m_cal, model_uncalibrated=m_raw, baseline=m_base,
                           mae_gain_vs_baseline_pct=None if gain is None else round(gain, 1),
                           conformal_delta=round(conf.delta.get(tgt, 0.0), 4))

    analog = AnalogIndex(feature_cols=feat_cols, obs_days=obs_days).fit(
        X[X["well_id"].isin(tr_ids)], prod[prod["well_id"].isin(tr_ids)], train_labels)

    version = registry.make_version()
    bundle = dict(model=model, conformal=conf, baseline=base, analog=analog,
                  feature_cols=feat_cols, obs_days=obs_days,
                  train_labels=train_labels, train_ids=tr_ids,
                  coverage={t: r["model"]["coverage"] for t, r in report.items()},
                  meta=dict(model_version=version, label_def_version=label_def_version(),
                            backend=model.backend, split_method=split_method,
                            n_train=len(tr_ids), n_calib=len(cal_ids), n_test=len(te_ids),
                            n_eligible=len(elig), n_wells_total=len(master),
                            min_lifecycle_days=MIN_LIFECYCLE_DAYS,
                            quality=quality_summary(qrep), report=report,
                            data_source=str(master["data_source"].mode().iloc[0])))
    registry.save(bundle, version)
    return bundle["meta"]
