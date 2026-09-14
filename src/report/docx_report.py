"""评估报告初稿生成（方案 §6.4 / §8）。

报告的每个数字都取自 services 返回的 JSON —— 与界面、与智能体回答同源。
首页固定标注"预评估"免责声明；检查清单里凡有"需人工确认"项，
报告页眉会标红提示，未处理完不得作为结论使用。

依赖 python-docx；未安装时自动降级为 markdown，保证流程不中断。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..api import services as S
from ..config import path
from ..sec.checklist import DISCLAIMER

STATUS_CN = {"pass": "通过", "fail": "不通过", "needs_human": "需人工确认"}


def gather(well_code: str, as_of: str = "2026-12-31",
           price_deck_id: str = S.DEFAULT_PRICE_DECK) -> Dict:
    data: Dict = {"well": S.query_well(well_code)}
    for key, fn, kw in [
        ("predict", S.predict_lifecycle, {}),
        ("analogs", S.find_analog_wells, {"top_k": 3}),
        ("dca", S.fit_dca, {"price_deck_id": price_deck_id}),
        ("volumetric", S.estimate_reserves_volumetric, {}),
        ("crosscheck", S.cross_check_reserves, {}),
        ("sec", S.sec_screen, {"as_of": as_of, "price_deck_id": price_deck_id}),
    ]:
        try:
            data[key] = fn(well_code, **kw)
        except S.KernelError as exc:
            data[key] = {"_error": str(exc)}      # 失败如实记录，不留白也不编造
    return data


def _md(data: Dict) -> str:
    w, sec = data["well"], data.get("sec", {})
    L: List[str] = [
        f"# {w['well_code']} 储量与全生命周期预评估报告（初稿）", "",
        f"> {DISCLAIMER}", "",
        f"数据来源 `{w['data_source']}`｜模型版本 `{w['model_version']}`"
        f"｜口径版本 `{w['label_def_version']}`｜追溯号 `{w['trace_id']}`", "",
        "## 一、单井概况", "",
        f"- 区块层位：{w['block']} / {w['layer']}｜井型：{w['well_type']}｜状态：{w['status']}",
        f"- 投产日期：{w['first_prod_date']}｜生产 {w['prod_days']} 天｜累产油 {w['cum_oil_t']} t",
        f"- 近 14 天日产 {w['current_rate_t_per_d']} t/d｜井口压力 {w['latest_whp_mpa']} MPa"
        f"｜含水 {w['water_cut_pct']}%", "",
    ]

    p = data.get("predict", {})
    if "results" in p:
        L += ["## 二、全生命周期预测", "",
              f"观测窗 {p['obs_days']} 天。P10/P50/P90 为分位数本身（p10 为数值小者）。", "",
              "| 指标 | P10 | P50 | P90 | 单位 |", "| --- | ---: | ---: | ---: | --- |"]
        cn = dict(t_oil_break="见油时间", t_peak="达峰时间", q_peak="达峰产量",
                  p_peak="达峰压力", eur="EUR")
        for k, v in p["results"].items():
            L.append(f"| {cn.get(k, k)} | {v['p10']} | {v['p50']} | {v['p90']} | {v['unit']} |")
        L += ["", f"主要影响特征（{p['explain']['method']}）：" + "，".join(
            f"{t['feature']} {t['contrib']:+g}" for t in p["explain"]["top_features"][:5]), ""]

    a = data.get("analogs", {})
    if "analogs" in a:
        L += ["### 类比井", "", "| 井号 | 相似度 | 实际达峰(d) | 峰值(t/d) | EUR(t) |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for h in a["analogs"]:
            L.append(f"| {h['well_code']} | {h['similarity']} | {h['t_peak_actual']} "
                     f"| {h['q_peak_actual']} | {h['eur_actual']} |")
        L.append("")

    d = data.get("dca", {})
    if "fit" in d:
        f, e = d["fit"], d["economics"]
        L += ["## 三、递减分析与 EUR", "",
              f"- 模型 {f['model']}｜b={f['b']}｜Di={f['di_per_month']}/月"
              f"｜终端递减 {f['d_min_per_month']}/月｜R²={f['r2']}｜样本 {f['n_points']} 点",
              f"- EUR：P10 {d['eur']['p10']} / P50 {d['eur']['p50']} / P90 {d['eur']['p90']} t"
              f"（自助法 {d['eur']['n_bootstrap']} 次）",
              f"- 经济极限 {e['q_econ_t_per_d']} t/d，第 {e['t_econ_month']} 个月到达"
              f"（价格册 {e['price_deck_id']}，12 月均价 {e['avg_12m_price_usd_bbl']} USD/bbl）",
              f"- 约束：{d['guard']}", ""]

    v = data.get("volumetric", {})
    if "ooip_t" in v:
        o = v["ooip_t"]
        L += ["## 四、容积法储量", "",
              f"- OOIP：P10 {o['p10']} / P50 {o['p50']} / P90 {o['p90']} t（{v['method']}）", ""]

    c = data.get("crosscheck", {})
    if "consistency" in c:
        L += ["## 五、动静态互校", "",
              f"- 判定：**{c['consistency']}**｜隐含采收率 {c['rf_implied_pct']}%"
              f"｜区块经验区间 {c['rf_reference_pct']}", ""]
        for fnd in c["findings"]:
            L.append(f"- [{fnd['severity']}] {fnd['finding']}"
                     + (f"　建议：{fnd['action']}" if fnd.get("action") else ""))
        L.append("")

    if "category" in sec:
        sm = sec["summary"]
        L += ["## 六、SEC 储量预评估", "",
              f"- 基准日 {sec['as_of']}｜类别 **{sec['category']}**｜{sec['category_rationale']}",
              f"- 已证实储量 {sec['proved_reserves_t']} t｜口径 {sec['basis']}",
              f"- 满足性检查：{sm['n_pass']}/{sm['n_items']} 项通过，"
              f"{sm['n_needs_human']} 项需人工确认", "",
              "| 状态 | 检查项 | 依据 | 条款 |", "| --- | --- | --- | --- |"]
        for it in sec["checklist"]:
            L.append(f"| {STATUS_CN[it['status']]} | {it['item']} | {it['evidence']} "
                     f"| {it['citation'] or '—'} |")
        L += ["", f"> {sm['disclaimer']}", ""]

    errs = [f"{k}：{v['_error']}" for k, v in data.items() if isinstance(v, dict) and "_error" in v]
    if errs:
        L += ["## 附：未能完成的分析", ""] + [f"- {e}" for e in errs] + [""]
    return "\n".join(L)


def build_report(well_code: str, as_of: str = "2026-12-31",
                 price_deck_id: str = S.DEFAULT_PRICE_DECK,
                 out_dir: Optional[Path] = None) -> str:
    data = gather(well_code, as_of, price_deck_id)
    md = _md(data)
    out = Path(out_dir or path("artifacts_dir"))
    md_path = out / f"report_{well_code}_{as_of}.md"
    md_path.write_text(md, encoding="utf-8")

    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        return str(md_path)          # 未装 python-docx 时以 markdown 交付，流程不中断

    doc = Document()
    doc.add_heading(f"{well_code} 储量与全生命周期预评估报告（初稿）", level=0)
    warn = doc.add_paragraph(DISCLAIMER)
    warn.runs[0].bold = True
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("> ") or s.startswith("| ---"):
            continue
        if s.startswith("# "):
            continue
        if s.startswith("## "):
            doc.add_heading(s[3:], level=1)
        elif s.startswith("### "):
            doc.add_heading(s[4:], level=2)
        elif s.startswith("|"):
            doc.add_paragraph(" ".join(x.strip() for x in s.strip("|").split("|")),
                              style="List Bullet")
        elif s.startswith("- "):
            doc.add_paragraph(s[2:], style="List Bullet")
        else:
            doc.add_paragraph(s)
    for p in doc.paragraphs:
        for r in p.runs:
            r.font.size = r.font.size or Pt(10.5)
    docx_path = out / f"report_{well_code}_{as_of}.docx"
    doc.save(docx_path)
    return str(docx_path)
