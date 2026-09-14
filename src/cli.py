"""命令行入口。

    python -m src.cli init            # 建库 + 生成合成井 + 入库
    python -m src.cli train           # 训练 + 保形校准 + 出算法指标
    python -m src.cli demo            # 端到端演示：新井预测 -> 储量 -> 互校 -> SEC
    python -m src.cli ask "GL-A-0001 什么时候达峰"
    python -m src.cli eval-algo       # 三种切分方式对比（证明不能用随机切分）
    python -m src.cli eval-agent      # 跑评测集，出智能体指标
    python -m src.cli serve           # 起 HTTP 服务
    python -m src.cli report GL-A-0001
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import db


def _p(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_init(args) -> None:
    from .ingest.synth_adapter import ingest
    db.init_schema()
    counts = ingest(n_wells=args.n_wells, seed=args.seed)
    print("已入库：", {k: v for k, v in counts.items() if k != "truth_file"})
    print("合成真值：", counts["truth_file"])


def cmd_train(args) -> None:
    from .pipeline import run_training
    from .api import services
    meta = run_training(obs_days=args.obs_days, split_method=args.split)
    services.reset_cache()
    print(f"模型版本 {meta['model_version']}｜后端 {meta['backend']}｜"
          f"切分 {meta['split_method']}｜训练/校准/测试 = "
          f"{meta['n_train']}/{meta['n_calib']}/{meta['n_test']}")
    print(f"{'目标':14s}{'模型MAE':>10s}{'基线MAE':>10s}{'提升':>8s}{'覆盖率':>8s}{'校准前':>8s}")
    for tgt, r in meta["report"].items():
        g = r["mae_gain_vs_baseline_pct"]
        print(f"{tgt:14s}{r['model']['mae']:10.2f}{r['baseline']['mae']:10.2f}"
              f"{(str(g) + '%'):>8s}{r['model']['coverage']:8.2f}"
              f"{r['model_uncalibrated']['coverage']:8.2f}")
    print("数据质量门禁：", meta["quality"])


def cmd_demo(args) -> None:
    from .api import services as S
    from . import db
    m = S._tables()["master"]
    new = m[m["first_prod_date"] > "2026-01-01"]
    new_code = args.new_well or (new.iloc[0]["well_code_anon"] if len(new) else None)

    # 老井演示样本挑「仍在经济生产」的那口：SEC 主线要走通 PDP 分支才有看头，
    # 挑到一口已跌破经济极限的井，演示只会得到 NOT_PROVED。
    rate = db.read_df(
        "SELECT well_id, AVG(oil_t) r FROM (SELECT well_id, oil_t, "
        "ROW_NUMBER() OVER (PARTITION BY well_id ORDER BY day_index DESC) rn "
        "FROM prod_daily WHERE hours_on>0) WHERE rn<=14 GROUP BY well_id")
    old = (m[(m["first_prod_date"] < "2023-01-01") & (m["status"] == "producing")]
           .merge(rate, on="well_id").sort_values("r", ascending=False))
    old_code = args.old_well or (old.iloc[0]["well_code_anon"] if len(old) else None)

    print("=" * 72)
    print(f"演示主线一：新井 {new_code} —— 只用早期数据预测全生命周期")
    print("=" * 72)
    r = S.predict_lifecycle(new_code)
    name = dict(t_oil_break="见油时间", t_peak="达峰时间", q_peak="达峰产量",
                p_peak="达峰压力", eur="EUR")
    for k, v in r["results"].items():
        print(f"  {name.get(k, k):8s} P50={v['p50']:>10.2f} {v['unit']:<5s}"
              f" 区间 [{v['p10']:.2f}, {v['p90']:.2f}]")
    print("  主要影响特征：", ", ".join(
        f"{t['feature']}({t['contrib']:+g})" for t in r["explain"]["top_features"][:4]))
    a = S.find_analog_wells(new_code, top_k=3)
    for h in a["analogs"]:
        print(f"  类比井 {h['well_code']} 相似度 {h['similarity']}，"
              f"实际达峰 {h['t_peak_actual']} d / {h['q_peak_actual']} t/d")

    print()
    print("=" * 72)
    print(f"演示主线二：老井 {old_code} —— 递减分析 -> 储量互校 -> SEC 预评估")
    print("=" * 72)
    d = S.fit_dca(old_code)
    print(f"  递减模型 {d['fit']['model']}  b={d['fit']['b']}  "
          f"Di={d['fit']['di_per_month']}/月  R²={d['fit']['r2']}")
    print(f"  EUR P50={d['eur']['p50']:.0f} t，区间 [{d['eur']['p10']:.0f}, {d['eur']['p90']:.0f}]，"
          f"已累产 {d['cum_to_date_t']:.0f} t")
    print(f"  经济极限 {d['economics']['q_econ_t_per_d']} t/d，"
          f"第 {d['economics']['t_econ_month']:.0f} 个月到达")
    c = S.cross_check_reserves(old_code)
    print(f"  互校判定 {c['consistency']}，隐含采收率 {c['rf_implied_pct']}%，"
          f"区块经验区间 {c['rf_reference_pct']}")
    for f in c["findings"]:
        print(f"    [{f['severity']}] {f['finding']}")
    s = S.sec_screen(old_code)
    print(f"  SEC 类别 {s['category']}｜已证实储量 {s['proved_reserves_t']:.0f} t｜口径 {s['basis']}")
    print(f"  满足性检查 {s['summary']['n_pass']}/{s['summary']['n_items']} 项通过"
          f"，{s['summary']['n_needs_human']} 项需人工确认")

    print()
    print("=" * 72)
    print("演示主线三：自然语言提问 —— 大模型只编排与成文，数字全部来自工具")
    print("=" * 72)
    from .agent.orchestrator import Agent
    ans = Agent().answer(f"{old_code} 的 SEC 储量预评估结果是什么")
    print(f"  意图 {ans.intent}（{ans.route['method']}，置信度 {ans.route['confidence']}）")
    print("  工具调用轨迹：", " -> ".join(
        f"{t['tool']}({t['status']},{t['elapsed_ms']}ms)" for t in ans.tool_trace))
    print(f"  数值一致性：{'通过' if ans.guard['ok'] else '未通过'}"
          f"（校验 {ans.guard['numbers_checked']} 个数字，"
          f"通过率 {ans.guard['pass_rate']}）")
    print(f"  条款引用：{ans.citations.get('cited')}，无效引用 {ans.citations.get('invalid')}")
    print("-" * 72)
    print(ans.text)


def cmd_ask(args) -> None:
    from .agent.orchestrator import Agent
    ans = Agent().answer(args.question)
    if args.json:
        _p(ans.to_dict())
        return
    print(ans.text)
    print("\n--- 调用轨迹 ---")
    for t in ans.tool_trace:
        print(f"  {t['tool']:32s} {t['status']:8s} {t['elapsed_ms']:>5d}ms "
              f"{t.get('error', '')}")
    print(f"数值一致性 {'通过' if ans.guard['ok'] else '未通过'}｜"
          f"trace_id={ans.trace_id}｜耗时 {ans.elapsed_ms}ms")


def cmd_eval_agent(args) -> None:
    from .eval.eval_agent import run
    out = run(target_n=args.n, verbose=args.verbose)
    m, t = out["metrics"], out["targets"]
    print(f"{'指标':26s}{'实测':>12s}{'目标':>12s}{'达标':>6s}")
    for k, target in t.items():
        v = m.get(k)
        ok = out["metrics"]["pass_targets"].get(k)
        vs = "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        print(f"{k:26s}{vs:>12s}{str(target):>12s}{('是' if ok else '否' if ok is not None else '—'):>6s}")
    print("\n分类别：")
    for cat, d in out["by_category"].items():
        print(f"  {cat:14s} n={d['n']:3d} 意图准确率={d['intent_accuracy']:.2f} "
              f"数值一致性={d['numeric_consistency']:.2f}")
    print("\n报告：", out["report_path"])


def cmd_eval_algo(args) -> None:
    from .eval.eval_algo import run
    out = run()
    print(f"{'切分':8s}{'目标':14s}{'MAE':>10s}{'覆盖率':>8s}{'相对基线':>9s}")
    for r in out["rows"]:
        print(f"{r['split']:8s}{r['target']:14s}{r['mae']:10.2f}{r['coverage']:8.2f}"
              f"{(str(r['gain_pct']) + '%'):>9s}")
    print("\n随机切分相对时间切分的乐观偏差（越大说明泄漏越明显）：")
    for k, v in out["random_split_optimism_pct"].items():
        print(f"  {k:14s}{v:+.1f}%")
    print("\n" + out["conclusion"])
    print("报告：", out["report_path"])


def cmd_report(args) -> None:
    from .report.docx_report import build_report
    p = build_report(args.well_code, as_of=args.as_of)
    print("报告已生成：", p)


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="oilwell-sec", description="油井全生命周期与 SEC 储量预评估平台")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="建库并生成合成井数据")
    p.add_argument("--n-wells", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("train", help="训练模型并出算法指标")
    p.add_argument("--obs-days", type=int, default=None)
    p.add_argument("--split", choices=["time", "group", "random"], default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("demo", help="端到端演示")
    p.add_argument("--new-well", default=None)
    p.add_argument("--old-well", default=None)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("ask", help="自然语言提问")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval-agent", help="跑智能体评测集")
    p.add_argument("-n", type=int, default=150)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_eval_agent)

    p = sub.add_parser("eval-algo", help="三种切分方式对比，证明不能用随机切分")
    p.set_defaults(func=cmd_eval_algo)

    p = sub.add_parser("report", help="生成单井评估报告初稿")
    p.add_argument("well_code")
    p.add_argument("--as-of", default="2026-12-31")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="起 HTTP 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
