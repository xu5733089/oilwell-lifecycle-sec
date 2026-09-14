"""智能体评测 runner（方案 §10.2）。

一条命令跑完全部用例，输出可直接写进成果报告的指标表：
  意图识别准确率 / 槽位抽取 F1 / 工具链正确率 /
  数值一致性通过率 / 条款引用命中率 / 无据结论率 / P95 响应时间 / 人工干预率
"""
from __future__ import annotations

import json
import statistics
from typing import Dict, List

from .. import db
from ..agent.orchestrator import Agent
from ..config import path
from .evalset import build

TARGETS = {
    "intent_accuracy": 0.95,
    "slot_f1": 0.90,
    "tool_chain_accuracy": 0.90,
    "numeric_consistency": 1.00,
    "citation_hit_rate": 0.90,
    "hallucination_rate": 0.01,      # 越小越好
    "p95_latency_ms": 30000,         # 越小越好
    "human_intervention_rate": 0.15,  # 越小越好
}
LOWER_IS_BETTER = {"hallucination_rate", "p95_latency_ms", "human_intervention_rate"}


def _well_codes(n: int = 12) -> List[str]:
    df = db.read_df(
        "SELECT well_code_anon FROM well_master WHERE status='producing' "
        "ORDER BY first_prod_date DESC LIMIT ?", (n,))
    return df["well_code_anon"].tolist() or ["GL-A-0001"]


def run(target_n: int = 150, verbose: bool = False) -> Dict:
    cases = build(_well_codes(), target_n=target_n)
    agent = Agent()

    rows: List[Dict] = []
    for c in cases:
        ans = agent.answer(c["question"])
        got_tools = [t["tool"] for t in ans.tool_trace if t["status"] == "ok"]
        need = set(c["expected_tools"])

        slot_ok = None
        if c["well_code"]:
            slot_ok = ans.slots.get("values", {}).get("well_code") == c["well_code"]

        rows.append(dict(
            id=c["id"], category=c["category"], question=c["question"],
            expected_intent=c["expected_intent"], got_intent=ans.intent,
            intent_ok=ans.intent == c["expected_intent"],
            slot_ok=slot_ok,
            tools_ok=need.issubset(set(got_tools)) if need else not got_tools,
            got_tools=got_tools,
            guard_ok=ans.guard.get("ok", True),
            guard_violations=len(ans.guard.get("violations", [])),
            citation_hit=ans.citations.get("hit_rate"),
            degraded=ans.degraded,
            needs_clarification=ans.needs_clarification,
            elapsed_ms=ans.elapsed_ms,
            tool_failures=sum(1 for t in ans.tool_trace if t["status"] != "ok"),
        ))
        if verbose:
            print(f"{c['id']} {c['category']:12s} {ans.intent:20s} "
                  f"guard={'ok' if rows[-1]['guard_ok'] else 'FAIL'}")

    n = len(rows)
    slot_rows = [r for r in rows if r["slot_ok"] is not None]
    cite_rows = [r for r in rows if r["citation_hit"] is not None]
    lat = sorted(r["elapsed_ms"] for r in rows)

    metrics = {
        "n_cases": n,
        "intent_accuracy": _rate(r["intent_ok"] for r in rows),
        "slot_f1": _rate(r["slot_ok"] for r in slot_rows) if slot_rows else None,
        "tool_chain_accuracy": _rate(r["tools_ok"] for r in rows),
        "numeric_consistency": _rate(r["guard_ok"] for r in rows),
        "citation_hit_rate": (round(statistics.mean(r["citation_hit"] for r in cite_rows), 4)
                              if cite_rows else None),
        "hallucination_rate": round(sum(r["guard_violations"] for r in rows) /
                                    max(sum(1 for r in rows if not r["needs_clarification"]), 1), 4),
        "p95_latency_ms": lat[int(0.95 * (n - 1))] if n else 0,
        "median_latency_ms": lat[n // 2] if n else 0,
        "human_intervention_rate": _rate(r["needs_clarification"] or r["degraded"] for r in rows),
        "degraded_rate": _rate(r["degraded"] for r in rows),
    }
    metrics["pass_targets"] = {
        k: (None if metrics.get(k) is None else
            (metrics[k] <= v if k in LOWER_IS_BETTER else metrics[k] >= v))
        for k, v in TARGETS.items()
    }

    by_cat: Dict[str, Dict] = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = dict(n=len(sub),
                           intent_accuracy=_rate(r["intent_ok"] for r in sub),
                           numeric_consistency=_rate(r["guard_ok"] for r in sub))

    out = dict(metrics=metrics, targets=TARGETS, by_category=by_cat,
               failures=[r for r in rows if not (r["intent_ok"] and r["guard_ok"])][:20])
    p = path("artifacts_dir") / "eval_agent.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out["report_path"] = str(p)
    return out


def _rate(flags) -> float:
    flags = list(flags)
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else 0.0
