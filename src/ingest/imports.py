"""业务数据导入：产量与工作量计划、单元资产账面价值、部署井位（PUD）。

真实数据接入的"最后一公里"：业务部门手里是 Excel / CSV 台账，不是数据库。流程固定为
    上传 → 预检（逐行逐列给出错误与提醒，不写库）→ 确认导入（按主键替换、记批次）→ 可撤销最近一次。

· 表头中英文均可（字段名或中文列名），列顺序不限；
· CSV 自动识别 UTF-8 / GBK，Excel 需要 openpyxl；
· 有任何错误就不允许导入，提醒不阻断；
· 每批记下写入的主键与被替换的旧行，撤销时原样恢复；
· 导入后清空服务缓存，受影响的评估自动重算（评估快照按指纹失效）。
这是业务数据写入，只走 HTTP 与界面，不注册为智能体工具；每次操作记审计日志。
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Dict, List, Optional

import pandas as pd

from .. import db, trace

MAX_BYTES = 5 * 1024 * 1024
MAX_ISSUES = 200


def _col(name: str, label: str, kind: str, required: bool = True, **kw) -> Dict:
    return dict(name=name, label=label, kind=kind, required=required, **kw)


TYPES: Dict[str, Dict] = {
    "plan": dict(
        table="unit_plan_monthly", title="产量与工作量计划", keys=["unit_id", "ym"],
        description="按单元、按月的新井、措施、老井产量计划，用于运行监控的计划与实际对标和计划完成率指标",
        columns=[_col("unit_id", "单元号", "unit"), _col("ym", "年月", "ym"),
                 _col("plan_new_wells", "计划新井数", "int", min=0),
                 _col("plan_new_oil_t", "计划新井产油(t)", "float", min=0),
                 _col("plan_measure_wells", "计划措施井次", "int", min=0),
                 _col("plan_measure_inc_t", "计划措施增油(t)", "float", min=0),
                 _col("plan_old_oil_t", "计划老井产油(t)", "float", min=0)]),
    "asset_book": dict(
        table="unit_asset_book", title="单元资产账面价值", keys=["unit_id", "as_of"],
        description="各评估期的期初资产净值与本期资本化投入（万元），用于产量法折耗与减值测试；期初净值可留空，由上期期末滚动",
        columns=[_col("unit_id", "单元号", "unit"), _col("as_of", "评估基准日", "eval_date"),
                 _col("opening_nbv_wan", "期初资产净值(万元)", "float", required=False, min=0),
                 _col("capex_additions_wan", "本期资本化投入(万元)", "float", min=0)]),
    "location": dict(
        table="unit_location", title="部署井位（PUD）", keys=["location_id"],
        description="开发方案中的部署井位：坐标、计划钻井年月、首次入账日期、是否已钻，用于 PUD 入账、五年规则与转化跟踪",
        columns=[_col("location_id", "井位编号", "str"), _col("unit_id", "单元号", "unit"),
                 _col("x_off", "东西坐标(m)", "float"), _col("y_off", "南北坐标(m)", "float"),
                 _col("planned_drill_ym", "计划钻井年月", "ym"),
                 _col("first_booked_as_of", "首次入账日期", "date", required=False),
                 _col("status", "状态", "enum", choices={"planned": "planned", "计划": "planned", "未钻": "planned",
                                                         "drilled": "drilled", "已钻": "drilled",
                                                         "cancelled": "cancelled", "取消": "cancelled", "已取消": "cancelled"}),
                 _col("drilled_well_id", "已钻井号", "well", required=False),
                 _col("capex_wan", "钻完井投资(万元)", "float", required=False, min=0)]),
}


def _norm(h: str) -> str:
    return re.sub(r"[\s_（）()\-]", "", str(h)).lower()


def _services():
    from ..api import services as S
    return S


def types() -> Dict:
    return dict(types=[dict(key=k, title=t["title"], description=t["description"], keys=t["keys"],
                            columns=[dict(name=c["name"], label=c["label"], required=c["required"]) for c in t["columns"]])
                       for k, t in TYPES.items()])


def _spec(import_type: str) -> Dict:
    if import_type not in TYPES:
        raise _services().KernelError(f"未知导入类型 {import_type!r}，可选：{'、'.join(TYPES)}")
    return TYPES[import_type]


# --------------------------------------------------------------------------- #
def _read(filename: str, content_base64: str) -> pd.DataFrame:
    S = _services()
    try:
        raw = base64.b64decode(content_base64 or "", validate=False)
    except Exception:
        raise S.KernelError("文件内容无法解码") from None
    if not raw:
        raise S.KernelError("文件为空")
    if len(raw) > MAX_BYTES:
        raise S.KernelError(f"文件超过 {MAX_BYTES // 1024 // 1024} MB")
    name = str(filename or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        try:
            df = pd.read_excel(io.BytesIO(raw), dtype=str)
        except ImportError:
            raise S.KernelError("读取 Excel 需要 openpyxl，请安装或另存为 CSV 后上传") from None
        except Exception as exc:
            raise S.KernelError(f"Excel 读取失败：{exc}") from None
    elif name.endswith(".xls"):
        raise S.KernelError("不支持旧版 .xls，请另存为 .xlsx 或 CSV")
    else:
        text = None
        for enc in ("utf-8-sig", "gbk"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise S.KernelError("CSV 编码无法识别，请另存为 UTF-8 或 GBK")
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except Exception as exc:
            raise S.KernelError(f"CSV 解析失败：{exc}") from None
    df = df.fillna("")
    df = df[~(df.astype(str).apply(lambda r: "".join(r).strip(), axis=1) == "")]
    return df.reset_index(drop=True)


def _parse_ym(v: str) -> Optional[str]:
    v = v.strip()
    m = re.fullmatch(r"(20\d{2})[-/.年]?\s*(\d{1,2})月?(?:[-/.]\d{1,2}(?:\s.*)?)?", v)
    if not m:
        return None
    mo = int(m.group(2))
    return f"{m.group(1)}-{mo:02d}" if 1 <= mo <= 12 else None


def _parse_date(v: str) -> Optional[str]:
    try:
        ts = pd.to_datetime(v.strip().replace("年", "-").replace("月", "-").replace("日", ""), errors="raise")
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None


def validate(import_type: str, df: pd.DataFrame) -> Dict:
    S = _services()
    spec = _spec(import_type)
    errors: List[Dict] = []
    warnings: List[Dict] = []

    def issue(bucket: List[Dict], row: Optional[int], column: Optional[str], msg: str, value: str = "") -> None:
        if len(bucket) < MAX_ISSUES:
            bucket.append(dict(row=row, column=column, value=value, message=msg))

    lookup = {}
    for c in spec["columns"]:
        for alias in (c["name"], c["label"], re.sub(r"\(.*\)$", "", c["label"])):
            lookup[_norm(alias)] = c["name"]
    mapping = {}
    for h in df.columns:
        key = lookup.get(_norm(h))
        if key and key not in mapping.values():
            mapping[h] = key
        elif not key:
            issue(warnings, None, str(h), "无法识别的列，已忽略")
    have = set(mapping.values())
    for c in spec["columns"]:
        if c["required"] and c["name"] not in have:
            issue(errors, None, c["label"], f"缺少必填列（可用列名：{c['name']} 或 {c['label']}）")
    if errors:
        return dict(errors=errors, warnings=warnings, rows=[], n_rows=len(df), mapping=mapping)

    units = set(S._units()[0]["unit_id"])
    dates = [str(d) for d in S._su()["evaluation_dates"]]
    master = S._tables()["master"]
    well_ids = dict(zip(master["well_code_anon"].str.upper(), master["well_id"]))
    well_ids.update({w.upper(): w for w in master["well_id"]})
    labels = {c["name"]: c["label"] for c in spec["columns"]}
    rows: List[Dict] = []
    seen: Dict[tuple, int] = {}
    for i, rec in enumerate(df.rename(columns=mapping).to_dict("records")):
        line = i + 2                                     # 表头占第 1 行
        out: Dict = {}
        ok = True
        for c in spec["columns"]:
            v = str(rec.get(c["name"], "") or "").strip()
            if v == "":
                if c["required"]:
                    issue(errors, line, c["label"], "必填项为空"); ok = False
                out[c["name"]] = None
                continue
            kind = c["kind"]
            val = None
            if kind == "unit":
                val = v.upper()
                if val not in units:
                    issue(errors, line, c["label"], "单元号不存在", v); ok = False
            elif kind == "ym":
                val = _parse_ym(v)
                if val is None:
                    issue(errors, line, c["label"], "年月格式应为 YYYY-MM", v); ok = False
            elif kind in ("date", "eval_date"):
                val = _parse_date(v)
                if val is None:
                    issue(errors, line, c["label"], "日期格式应为 YYYY-MM-DD", v); ok = False
                elif kind == "eval_date" and val not in dates:
                    issue(errors, line, c["label"], f"须为评估基准日之一：{'、'.join(dates)}", v); ok = False
            elif kind in ("int", "float"):
                try:
                    num = float(v.replace(",", ""))
                    if kind == "int" and abs(num - round(num)) > 1e-9:
                        raise ValueError
                    val = int(round(num)) if kind == "int" else num
                    if "min" in c and val < c["min"]:
                        issue(errors, line, c["label"], f"不能小于 {c['min']}", v); ok = False
                except ValueError:
                    issue(errors, line, c["label"], "应为整数" if kind == "int" else "应为数值", v); ok = False
            elif kind == "enum":
                val = c["choices"].get(v.lower(), c["choices"].get(v))
                if val is None:
                    issue(errors, line, c["label"], "可选值：" + "、".join(sorted(set(c["choices"].values()))), v); ok = False
            elif kind == "well":
                val = well_ids.get(v.upper())
                if val is None:
                    issue(errors, line, c["label"], "井号不存在", v); ok = False
            else:
                val = v
            out[c["name"]] = val
        if not ok:
            continue
        key = tuple(out[k] for k in spec["keys"])
        if key in seen:
            issue(errors, line, "、".join(labels[k] for k in spec["keys"]), f"与第 {seen[key]} 行主键重复", " / ".join(map(str, key)))
            continue
        seen[key] = line
        if import_type == "location":
            if out["status"] == "drilled" and not out["drilled_well_id"]:
                issue(warnings, line, labels["drilled_well_id"], "状态为已钻但未填井号：无法按投产日期判断转化时点，各期均视为已钻")
            if out["drilled_well_id"] and out["status"] != "drilled":
                issue(warnings, line, labels["status"], "已填井号但状态不是已钻：按该井投产日期判断各期是否已钻")
        out["_line"] = line
        rows.append(out)

    if import_type == "plan" and rows:
        _plan_warnings(rows, warnings, issue)
    return dict(errors=errors, warnings=warnings, rows=rows, n_rows=len(df), mapping=mapping)


def _plan_warnings(rows: List[Dict], warnings: List[Dict], issue) -> None:
    """计划与实际偏差过大的提醒（不阻断）：同单元同月实际产量已知时，计划合计偏离超过 50%。"""
    S = _services()
    mon = S._monthly()
    uw = S._units()[1]
    act = mon.merge(uw, on="well_id").groupby(["unit_id", "ym"])["oil_t"].sum()
    for r in rows:
        a = act.get((r["unit_id"], r["ym"]))
        plan = sum(r[k] or 0.0 for k in ("plan_new_oil_t", "plan_measure_inc_t", "plan_old_oil_t"))
        if a and a > 0 and abs(plan - a) / a > 0.5:
            issue(warnings, r["_line"], "计划产油合计", f"与当月实际产量 {a:,.0f} t 偏差 {100 * (plan - a) / a:+.0f}%，请核对单位与月份",
                  f"{plan:,.0f}")


def _existing(spec: Dict, rows: List[Dict]) -> pd.DataFrame:
    cur = db.read_df(f"SELECT * FROM {spec['table']}")
    if cur.empty or not rows:
        return cur.iloc[0:0]
    keys = {tuple(str(r[k]) for k in spec["keys"]) for r in rows}
    mask = cur[spec["keys"]].astype(str).apply(tuple, axis=1).isin(keys)
    return cur[mask]


def preview(import_type: str, filename: str, content_base64: str, trace_id: Optional[str] = None) -> Dict:
    S = _services()
    S._snapshot_ready()
    spec = _spec(import_type)
    res = validate(import_type, _read(filename, content_base64))
    rows = res["rows"]
    replaced = _existing(spec, rows)
    summary = dict(n_rows=res["n_rows"], n_valid=len(rows), n_errors=len(res["errors"]), n_warnings=len(res["warnings"]),
                   n_replace=int(len(replaced)), n_insert=len(rows) - int(len(replaced)),
                   n_units=len({r["unit_id"] for r in rows if r.get("unit_id")}))
    for k in ("ym", "as_of", "planned_drill_ym"):
        vals = sorted({r[k] for r in rows if r.get(k)})
        if vals:
            summary[f"{k}_range"] = [vals[0], vals[-1]]
    return dict(**S._env(trace_id), import_type=import_type, title=spec["title"], filename=filename,
                can_commit=not res["errors"] and bool(rows), summary=summary,
                columns=[dict(name=c["name"], label=c["label"]) for c in spec["columns"]],
                errors=res["errors"], warnings=res["warnings"],
                sample=[{k: v for k, v in r.items() if k != "_line"} for r in rows[:20]],
                mapping=res["mapping"])


def commit(import_type: str, filename: str, content_base64: str, trace_id: Optional[str] = None) -> Dict:
    S = _services()
    S._snapshot_ready()
    spec = _spec(import_type)
    res = validate(import_type, _read(filename, content_base64))
    if res["errors"]:
        raise S.KernelError(f"预检发现 {len(res['errors'])} 处错误，未导入；请修正后重新上传")
    if not res["rows"]:
        raise S.KernelError("没有可导入的数据行")
    rows = [{k: v for k, v in r.items() if k != "_line"} for r in res["rows"]]
    replaced = _existing(spec, rows)
    new = pd.DataFrame(rows)
    new["data_source"] = "IMPORT"
    tid = trace_id or trace.new_trace_id("imp")
    batch_id = "B" + tid.split("_", 1)[-1][:12]
    keys = [[r[k] for k in spec["keys"]] for r in rows]
    where = " AND ".join(f"{k} = ?" for k in spec["keys"])
    with db.connect() as conn:
        conn.executemany(f"DELETE FROM {spec['table']} WHERE {where}", keys)
        new.to_sql(spec["table"], conn, if_exists="append", index=False)
        conn.execute("INSERT INTO import_batch (batch_id, import_type, filename, n_rows, summary_json, keys_json, replaced_json, "
                     "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)",
                     (batch_id, import_type, filename, len(rows),
                      json.dumps(dict(n_replace=int(len(replaced)), n_insert=len(rows) - int(len(replaced)),
                                      n_warnings=len(res["warnings"])), ensure_ascii=False),
                      json.dumps(keys, ensure_ascii=False, default=str),
                      replaced.astype(object).where(replaced.notna(), None).to_json(orient="records", force_ascii=False),
                      trace.now_iso()))
    S.reset_cache()
    trace.audit(tid, "user", f"import:{import_type}", dict(batch_id=batch_id, filename=filename, n_rows=len(rows)))
    return dict(**S._env(tid), batch_id=batch_id, import_type=import_type, n_rows=len(rows),
                n_replace=int(len(replaced)), n_insert=len(rows) - int(len(replaced)), n_warnings=len(res["warnings"]))


def undo(batch_id: str, trace_id: Optional[str] = None) -> Dict:
    S = _services()
    S._snapshot_ready()
    b = db.read_df("SELECT * FROM import_batch WHERE batch_id = ?", (batch_id,))
    if b.empty:
        raise S.KernelError(f"导入批次 {batch_id} 不存在")
    b = b.iloc[0]
    if b["status"] != "active":
        raise S.KernelError(f"批次 {batch_id} 已撤销")
    latest = db.read_df("SELECT batch_id FROM import_batch WHERE import_type = ? AND status = 'active' "
                        "ORDER BY created_at DESC, rowid DESC LIMIT 1", (b["import_type"],))
    if latest["batch_id"].iloc[0] != batch_id:
        raise S.KernelError("只能撤销该类型最近一次仍有效的导入，请先撤销更晚的批次")
    spec = TYPES[b["import_type"]]
    keys = json.loads(b["keys_json"])
    old = pd.DataFrame(json.loads(b["replaced_json"] or "[]"))
    where = " AND ".join(f"{k} = ?" for k in spec["keys"])
    tid = trace_id or trace.new_trace_id("imp")
    with db.connect() as conn:
        conn.executemany(f"DELETE FROM {spec['table']} WHERE {where}", keys)
        if not old.empty:
            old.to_sql(spec["table"], conn, if_exists="append", index=False)
        conn.execute("UPDATE import_batch SET status = 'reverted', reverted_at = ? WHERE batch_id = ?",
                     (trace.now_iso(), batch_id))
    S.reset_cache()
    trace.audit(tid, "user", f"import_undo:{b['import_type']}", dict(batch_id=batch_id))
    return dict(**S._env(tid), batch_id=batch_id, restored=int(len(old)), removed=len(keys))


def history(limit: int = 20, trace_id: Optional[str] = None) -> Dict:
    S = _services()
    S._snapshot_ready()
    df = db.read_df("SELECT batch_id, import_type, filename, n_rows, summary_json, status, created_at, reverted_at "
                    "FROM import_batch ORDER BY created_at DESC, rowid DESC LIMIT ?", (int(limit),))
    active_latest = {}
    for r in df.itertuples(index=False):
        if r.status == "active" and r.import_type not in active_latest:
            active_latest[r.import_type] = r.batch_id
    batches = [dict(batch_id=r.batch_id, import_type=r.import_type, title=TYPES.get(r.import_type, {}).get("title", r.import_type),
                    filename=r.filename, n_rows=int(r.n_rows or 0), summary=json.loads(r.summary_json or "{}"),
                    status=r.status, created_at=r.created_at, reverted_at=r.reverted_at,
                    can_undo=active_latest.get(r.import_type) == r.batch_id)
               for r in df.itertuples(index=False)]
    return dict(**S._env(trace_id), batches=batches)


def template(import_type: str) -> bytes:
    """模板：中文表头 + 取自当前库的两行示例（UTF-8 带 BOM，Excel 直接打开不乱码）。"""
    S = _services()
    S._snapshot_ready()
    spec = _spec(import_type)
    cur = db.read_df(f"SELECT * FROM {spec['table']} LIMIT 2")
    rename = {c["name"]: c["label"] for c in spec["columns"]}
    if cur.empty:
        cur = pd.DataFrame(columns=[c["name"] for c in spec["columns"]])
    if import_type == "location" and "drilled_well_id" in cur:
        codes = S._codes()
        cur["drilled_well_id"] = cur["drilled_well_id"].map(lambda w: codes.get(w, w) if w else w)
    out = cur[[c["name"] for c in spec["columns"]]].rename(columns=rename)
    return out.to_csv(index=False).encode("utf-8-sig")
