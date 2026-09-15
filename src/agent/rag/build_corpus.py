"""准则条文库构建：把 eCFR 官方 XML 切成段落级条款语料。

    python -m src.cli build-standards            # 用 data/standards/raw 下的 XML 构建
    python -m src.cli build-standards --fetch    # 先从 eCFR 拉取指定版本日期的原文

设计要点：
  · 官方英文原文逐段保留，**条款号精确到段落层级**（(a)(31)(ii) 而不是笼统的 Rule 4-10）；
  · eCFR 的段落标号有四到五级（(a) → (1) → (i) → (A) → (1)），同一段里还可能连着下一级标号，
    这里用层级栈解析，罗马数字 (i)/(v)/(x) 与字母 (i) 按上下文区分；
  · 中文"要点"来自 data/standards/notes_zh.yaml，明确标注为**非官方说明**，不冒充译文；
  · 生成物是纯 markdown，检索器按 `## [条款号] 标题` 切分，条款号即引用号。
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ...config import path

ECFR_DATE = "2025-01-01"
ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-17.xml?part={part}&section={section}"
# (eCFR 段号, 引用前缀, 所属法规, 输出文件)
SOURCES: List[Tuple[str, str, str, str]] = [
    ("210.4-10", "Rule 4-10", "Regulation S-X", "reg_sx_rule_4-10.md"),
    ("229.1201", "Item 1201", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1202", "Item 1202", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1203", "Item 1203", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1204", "Item 1204", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1205", "Item 1205", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1206", "Item 1206", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1207", "Item 1207", "Regulation S-K", "reg_sk_items_1201-1208.md"),
    ("229.1208", "Item 1208", "Regulation S-K", "reg_sk_items_1201-1208.md"),
]
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
         "xvi", "xvii", "xviii", "xix", "xx"}
MARK_RE = re.compile(r"^\((\d{1,2}|[a-z]{1,5}|[A-Z])\)\s*")
INLINE_RE = re.compile(r"^(?P<head>(?:(?!\((?:\d{1,2}|[ivx]{1,5}|[A-Z])\)).){2,140}?\.)\s+\((?P<tok>\d{1,2}|[ivx]{1,5}|[A-Z])\)\s+(?P<rest>.*)$", re.S)
INSTR_RE = re.compile(r"^(?P<label>Instructions?(?:\s+\d+)?)\s+to\s+(?:paragraph\s+(?P<para>(?:\([^)]+\))+)|Item\s+\d+)\s*[:.]\s*(?P<body>.*)$", re.S)


def raw_dir() -> Path:
    return path("standards_dir") / "raw"


def fetch(date: str = ECFR_DATE, folder: Optional[Path] = None) -> List[Path]:
    """从 eCFR 拉取原文。eCFR 要求请求声明可接受压缩，否则返回 406。"""
    folder = folder or raw_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for section, *_ in SOURCES:
        part = section.split(".")[0]
        req = urllib.request.Request(ECFR_URL.format(date=date, part=part, section=section),
                                     headers={"Accept": "application/xml", "Accept-Encoding": "gzip",
                                              "User-Agent": "oilwell-lifecycle-sec/standards-builder"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
        p = folder / f"{section}.xml"
        p.write_bytes(data)
        out.append(p)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _level(tok: str, stack: List[Tuple[int, str]]) -> int:
    if tok.isdigit():
        # (A) 之下的 (1)(2) 是第五级；但 (D) 之后出现的 (17) 恰好接续第二级的 (16)，应回到第二级
        lv2 = next((t for lv, t in reversed(stack) if lv == 2), None)
        lv5 = next((t for lv, t in reversed(stack) if lv == 5), None)
        if lv2 is not None and int(tok) == int(lv2) + 1 and not (lv5 is not None and int(tok) == int(lv5) + 1):
            return 2
        return 5 if stack and stack[-1][0] >= 4 else 2
    if tok.isupper():
        return 4
    if tok in ROMAN:
        top = next((t for lv, t in reversed(stack) if lv == 1), None)
        if len(tok) == 1 and top is not None and len(top) == 1 and ord(tok) == ord(top) + 1:
            return 1                       # 字母序号恰好轮到 i / v / x
        return 3
    return 1


def parse_section(xml_path: Path, prefix: str, regulation: str) -> Dict:
    root = ET.parse(xml_path).getroot()
    section = root.get("N")
    head = _norm("".join(root.find("HEAD").itertext()))
    items: List[Dict] = []
    stack: List[Tuple[int, str]] = []

    def cite(tokens: Tuple[str, ...]) -> str:
        return prefix + "".join(f"({t})" for t in tokens)

    def add(tokens: Tuple[str, ...], text: str, title: Optional[str] = None, kind: str = "paragraph") -> None:
        items.append(dict(citation=cite(tokens), cfr=f"17 CFR {section}" + "".join(f"({t})" for t in tokens),
                          tokens=list(tokens), level=len(tokens), title=title, text=text, kind=kind,
                          parent=cite(tokens[:-1]) if tokens else None))

    for el in root.iter():
        if el.tag not in ("P", "FP"):
            continue
        txt = _norm("".join(el.itertext()))
        if not txt:
            continue
        m = INSTR_RE.match(txt)
        if m:
            toks = tuple(re.findall(r"\(([^)]+)\)", m.group("para") or ""))
            label = m.group("label")
            items.append(dict(citation=f"{cite(toks)} {label}", cfr=f"17 CFR {section}"
                              + "".join(f"({t})" for t in toks) + f", {label}", tokens=list(toks),
                              level=len(toks) + 1, title=label, text=m.group("body"), kind="instruction",
                              parent=cite(toks)))
            continue
        mk = MARK_RE.match(txt)
        if not mk:
            if not items:
                add((), txt, title=re.sub(r"^§\s*[\d.\-]+\s*(\(Item \d+\)\s*)?", "", head).rstrip("."),
                    kind="preamble")
            else:
                items[-1]["text"] += "\n\n" + txt          # 续段：表名、a./b. 分项、条款尾句
            continue
        tok = mk.group(1)
        lv = _level(tok, stack)
        while stack and stack[-1][0] >= lv:
            stack.pop()
        stack.append((lv, tok))
        rest = txt[mk.end():]
        inline = INLINE_RE.match(rest)
        if inline:
            add(tuple(t for _, t in stack), inline.group("head"), title=inline.group("head").rstrip("."))
            tok2 = inline.group("tok")
            lv2 = _level(tok2, stack)
            while stack and stack[-1][0] >= lv2:
                stack.pop()
            stack.append((lv2, tok2))
            add(tuple(t for _, t in stack), inline.group("rest"))
            continue
        title = None
        hm = re.match(r"^((?:(?!\.\s).){2,90}?)\.\s", rest)
        if hm and len(stack) <= 2 and hm.group(1)[0].isupper() and len(hm.group(1).split()) <= 10 \
                and not re.search(r"\b(is|are|means|shall|must)\b", hm.group(1)):
            title = hm.group(1)
        add(tuple(t for _, t in stack), rest, title=title)

    short_head = re.sub(r"^§\s*[\d.\-]+\s*(\(Item \d+\)\s*)?", "", head).rstrip(".")
    titles: Dict[str, str] = {}
    for it in items:                               # 无标题的分项继承最近上级的标题
        if it["title"]:
            titles[it["citation"]] = it["title"]
        else:
            p = it["parent"]
            while p and p not in titles and p != prefix:
                p = p[: p.rfind("(")] if "(" in p else None
            it["title"] = titles.get(p, short_head)
    return dict(section=section, prefix=prefix, regulation=regulation, head=head, items=items)


def load_notes() -> Dict:
    p = path("standards_dir") / "notes_zh.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(date: str = ECFR_DATE, do_fetch: bool = False) -> Dict[str, int]:
    if do_fetch:
        fetch(date)
    notes = load_notes()
    zh_titles, zh_notes = notes.get("titles", {}), notes.get("notes", {})
    files: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    for section, prefix, regulation, out in SOURCES:
        xml = raw_dir() / f"{section}.xml"
        if not xml.exists():
            raise FileNotFoundError(f"缺少原文 {xml}，请先运行 python -m src.cli build-standards --fetch")
        sec = parse_section(xml, prefix, regulation)
        lines = files.setdefault(out, [])
        for it in sec["items"]:
            c = it["citation"]
            zt, p = "", c.split(" Instruction")[0]
            while p and not zt:
                zt = zh_titles.get(p, "")
                p = p[: p.rfind("(")] if p.endswith(")") else ""
            meta = dict(cfr=it["cfr"], parent=it["parent"], kind=it["kind"], level=it["level"],
                        regulation=regulation, section_head=sec["head"], zh_title=zt,
                        source=f"eCFR {date}")
            lines.append(f"## [{c}] {it['title']}" + (f" · {zt}" if zt else ""))
            lines.append(f"<!-- meta: {json.dumps(meta, ensure_ascii=False)} -->")
            lines.append(it["text"].strip())
            if c in zh_notes:
                lines.append("")
                lines.append("要点（非官方中文说明）：" + _norm(zh_notes[c]))
            lines.append("")
        counts[prefix] = len(sec["items"])

    d = path("standards_dir")
    for old in d.glob("*.md"):                     # 生成物整体替换，避免残留旧语料
        old.unlink()
    header = ["# SEC 油气储量规则：官方原文条款库", "",
              f"> 来源：eCFR（Electronic Code of Federal Regulations）版本日期 {date}，美国联邦法规，公有领域文本。",
              "> 引用格式：[Rule 4-10(a)(22)(v)] = 17 CFR 210.4-10(a)(22)(v)；[Item 1203(b)] = 17 CFR 229.1203(b)。",
              "> 英文为官方原文；\"要点\"为平台整理的非官方中文说明，合规判断以原文为准。本文件由 build_corpus.py 生成，请勿手改。", ""]
    for name, lines in files.items():
        (d / name).write_text("\n".join(header + lines), encoding="utf-8")
    return counts
