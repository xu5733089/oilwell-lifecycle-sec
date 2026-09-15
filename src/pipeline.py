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
from .models.attribution import global_shap
from .models.conformal import Conformal, calibrate
from .models.ensemble import blend, choose_weights
from .models.gbdt_quantile import QuantileModel
from .models.seq_model import SeqQuantileModel, sequence_tensor
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


def _calibrated(preds: Dict[str, pd.DataFrame], conf: Conformal) -> Dict[str, pd.DataFrame]:
    out = {}
    for tgt, dfp in preds.items():
        lo, hi = conf.apply(tgt, dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))
        out[tgt] = pd.DataFrame({"p10": lo, "p50": dfp["p50"].to_numpy(float), "p90": hi}, index=dfp.index)
    return out


def _brief(m: Dict) -> Dict:
    return {k: (round(float(m[k]), 4) if isinstance(m.get(k), float) and np.isfinite(m[k]) else m.get(k))
            for k in ("n", "mae", "coverage", "interval_width")}


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

    # 序列模型：直接读前 obs_days 天的逐日曲线，与梯度提升用同一批训练井
    scfg = mcfg.get("seq") or {}
    seq_model, seq_all = None, None
    if scfg.get("enabled", True) and len(idx["train"]) >= 40:
        seq_all = sequence_tensor(prod, X["well_id"].tolist(), obs_days)
        seq_model = SeqQuantileModel(targets=targets, tab_cols=feat_cols, seed=seed,
                                     n_seeds=int(scfg.get("n_seeds", 3)), hidden=int(scfg.get("hidden", 16)),
                                     max_epochs=int(scfg.get("max_epochs", 200)))
        seq_model.fit(seq_all[idx["train"]], X.loc[idx["train"]], Y.loc[idx["train"]])

    def family(rows):
        pg = model.predict(X.loc[rows])
        ps = seq_model.predict(seq_all[rows], X.loc[rows], index=rows) if seq_model is not None else None
        return pg, ps

    # 校准集按投产时间交错对半：A 半选融合权重，B 半做保形校准 —— 选权重与校准不用同一批井
    cal = idx["calib"]
    if seq_model is not None and len(cal) >= 20:
        cal_a, cal_b = cal[::2], cal[1::2]
        pg_a, ps_a = family(cal_a)
        blend_info = choose_weights(Y.loc[cal_a], pg_a, ps_a, targets)
    else:
        cal_a, cal_b, blend_info = cal[:0], cal, {}
    weights = {t: float(blend_info.get(t, {}).get("weight", 0.0)) for t in targets}

    # 保形校准：融合模型是线上口径；两个单模型各自校准一份，只为测试集上公平对比
    alpha = mcfg["conformal_alpha"]
    conf = Conformal(alpha=alpha)
    conf_family = {"gbdt": Conformal(alpha=alpha), "seq": Conformal(alpha=alpha)}
    if len(cal_b):
        pg_b, ps_b = family(cal_b)
        for c, preds_b in ((conf, blend(pg_b, ps_b, weights)), (conf_family["gbdt"], pg_b), (conf_family["seq"], ps_b)):
            for tgt, dfp in (preds_b or {}).items():
                c.fit_target(tgt, Y.loc[cal_b, tgt].to_numpy(float),
                             dfp["p10"].to_numpy(float), dfp["p90"].to_numpy(float))

    # 基线 A：邻井类比
    base = AnalogBaseline(targets=targets).fit(
        master[master["well_id"].isin(tr_ids)], train_labels)

    # 测试集评估
    Xte, Yte = X.loc[idx["test"]], Y.loc[idx["test"]]
    pg_t, ps_t = family(idx["test"])
    pred = blend(pg_t, ps_t, weights)
    pred_cal = _calibrated(pred, conf)
    fam_cal = {"gbdt": _calibrated(pg_t, conf_family["gbdt"])}
    if ps_t is not None:
        fam_cal["seq"] = _calibrated(ps_t, conf_family["seq"])
        fam_cal["blend"] = pred_cal
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
                           conformal_delta=round(conf.delta.get(tgt, 0.0), 4),
                           families={k: _brief(metrics_for(Yte[tgt], v[tgt])) for k, v in fam_cal.items() if tgt in v},
                           seq_weight=weights.get(tgt, 0.0))

    analog = AnalogIndex(feature_cols=feat_cols, obs_days=obs_days).fit(
        X[X["well_id"].isin(tr_ids)], prod[prod["well_id"].isin(tr_ids)], train_labels)

    version = registry.make_version()
    bundle = dict(model=model, conformal=conf, baseline=base, analog=analog,
                  seq_model=seq_model, blend_weights=weights, blend_selection=blend_info,
                  conformal_family=conf_family, global_shap=global_shap(model, X),
                  feature_cols=feat_cols, obs_days=obs_days,
                  train_labels=train_labels, train_ids=tr_ids,
                  coverage={t: r["model"]["coverage"] for t, r in report.items()},
                  meta=dict(model_version=version, label_def_version=label_def_version(),
                            backend=model.backend + ("+tcn" if seq_model is not None else ""),
                            split_method=split_method,
                            seq=None if seq_model is None else dict(
                                history=seq_model.history, n_seeds=seq_model.n_seeds, channels=len(seq_all[0]),
                                n_calib_blend=int(len(cal_a)), n_calib_conformal=int(len(cal_b)),
                                weights=weights),
                            n_train=len(tr_ids), n_calib=len(cal_ids), n_test=len(te_ids),
                            n_eligible=len(elig), n_wells_total=len(master),
                            min_lifecycle_days=MIN_LIFECYCLE_DAYS,
                            quality=quality_summary(qrep), report=report,
                            data_source=str(master["data_source"].mode().iloc[0])))
    registry.save(bundle, version)
    return bundle["meta"]
