"""准则条文库检索（方案 §7.5）。

语料是 eCFR 官方英文原文（17 CFR 210.4-10、229.1201–1208），由 build_corpus.py 切到段落级，
每段一个 `## [条款号] 标题` —— 条款号精确到 (a)(31)(ii) 这一层，就是引用号。

三个刻意的设计选择：

1) **按条款切分，不按固定长度切。**
   准则类文本的语义单元就是条款。固定长度切会把一条规则劈成两半，检索召回质量差很多。

2) **BM25 + 向量混合，RRF 融合。**
   条款检索里关键词（"five years"、"first-day-of-the-month"）和语义都重要。
   向量侧用 TF-IDF + 余弦（sklearn 自带，内网可装）；换 bge 之类的嵌入模型只替换 _dense。

3) **中文问、英文原文答：术语表做查询扩展。**
   用户问"五年规则"，原文写的是 "scheduled to be drilled within five years"。
   notes_zh.yaml 的术语表把中文检索词映射到原文关键词，再叠加条款的中文标题与非官方要点，
   中文查询也能落到正确的英文条款上。查询里直接写条款号（"4-10(a)(31)"、"Item 1203"）时按条款号精确命中。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from ...config import path

CLAUSE_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.M)
META_RE = re.compile(r"<!--\s*meta:\s*(\{.*?\})\s*-->", re.S)
CITE_IN_QUERY = re.compile(r"(?:rule\s*)?(4-10|item\s*12\d\d|12\d\d)((?:\([0-9a-zA-Z]+\))*)", re.I)
RRF_K = 60
STOP = {"the", "of", "and", "or", "to", "a", "an", "in", "for", "by", "on", "with", "as", "be", "is",
        "are", "that", "this", "which", "such", "any", "at", "from", "its", "it", "shall", "may"}


@dataclass
class Chunk:
    citation: str
    title: str
    text: str
    source: str
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        body, _, note = self.text.strip().partition("要点（非官方中文说明）：")
        return dict(citation=self.citation, title=self.title, text=body.strip(), source=self.source,
                    note_zh=note.strip() or None, cfr=self.meta.get("cfr"), parent=self.meta.get("parent"),
                    zh_title=self.meta.get("zh_title") or None, regulation=self.meta.get("regulation"),
                    edition=self.meta.get("source"), kind=self.meta.get("kind"))


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _tokenize(s: str) -> List[str]:
    """中英混排的粗分词：英文按词（去复数、去停用词）、条款号整体保留、中文按字 + 二元组。"""
    s = s.lower()
    cites = re.findall(r"\d+-\d+(?:\([0-9a-z]+\))+|\(\w{1,5}\)", s)
    en = [_stem(w) for w in re.findall(r"[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)?%?", s) if w not in STOP]
    zh = re.findall(r"[一-鿿]", s)
    bi = ["".join(p) for p in zip(zh, zh[1:])]
    return cites + en + zh + bi


def _clause_key(c: str) -> str:
    return re.sub(r"\s+", "", c.lower()).replace("rule", "").replace("item", "")


def load_chunks(standards_dir: Optional[Path] = None) -> List[Chunk]:
    d = standards_dir or path("standards_dir")
    chunks: List[Chunk] = []
    for f in sorted(Path(d).glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        marks = list(CLAUSE_RE.finditer(raw))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            body = raw[m.end():end]
            meta: Dict = {}
            mm = META_RE.search(body)
            if mm:
                try:
                    meta = json.loads(mm.group(1))
                except ValueError:
                    meta = {}
                body = body[:mm.start()] + body[mm.end():]
            title = m.group(2).strip()
            zh = meta.get("zh_title")
            if zh and title.endswith(f" · {zh}"):              # 标题行里的中文名只供检索，展示时单独给 zh_title
                title = title[: -len(f" · {zh}")]
            chunks.append(Chunk(citation=m.group(1).strip(), title=title,
                                text=body, source=f.name, meta=meta))
    return chunks


def load_glossary(standards_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    p = Path(standards_dir or path("standards_dir")) / "notes_zh.yaml"
    if not p.exists():
        return {}
    g = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("glossary") or {}
    return {str(k): [str(x) for x in v] for k, v in g.items()}


class Retriever:
    def __init__(self, chunks: Optional[List[Chunk]] = None, glossary: Optional[Dict[str, List[str]]] = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.glossary = glossary if glossary is not None else load_glossary()
        self._by_key = {_clause_key(c.citation): i for i, c in enumerate(self.chunks)}
        docs = [f"{c.citation} {c.title} {c.meta.get('zh_title', '')} {c.text}" for c in self.chunks]
        self.tokens = [_tokenize(d) for d in docs]
        self.tf = []
        self.df: Dict[str, int] = {}
        for toks in self.tokens:
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = float(np.mean([len(t) for t in self.tokens])) if self.tokens else 1.0
        self.vec = TfidfVectorizer(analyzer=_tokenize)
        self.M = self.vec.fit_transform(docs) if docs else None

    # ---------------- 查询扩展 ----------------
    def expand(self, query: str) -> str:
        extra = [" ".join(v) for k, v in sorted(self.glossary.items(), key=lambda kv: -len(kv[0]))
                 if k.lower() in query.lower()]
        return query if not extra else f"{query} {' '.join(extra)}"

    # ---------------- BM25 ----------------
    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = _tokenize(query)
        n = len(self.chunks)
        scores = np.zeros(n)
        for i, tf in enumerate(self.tf):
            dl = len(self.tokens[i]) or 1
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
            scores[i] = s
        return scores

    # ---------------- 向量 ----------------
    def _dense(self, query: str) -> np.ndarray:
        if self.M is None:
            return np.zeros(len(self.chunks))
        qv = self.vec.transform([query])
        num = (self.M @ qv.T).toarray().ravel()
        den = (np.sqrt(self.M.multiply(self.M).sum(1)).A.ravel() *
               np.sqrt(qv.multiply(qv).sum()) + 1e-9)
        return num / den

    def _explicit(self, query: str) -> List[int]:
        """查询里直接写了条款号：精确命中该条款（找不到就退到最近的上级条款）。"""
        out = []
        for m in CITE_IN_QUERY.finditer(query):
            head = m.group(1).lower().replace("item", "").strip()
            key = _clause_key(head + (m.group(2) or ""))
            while key:
                if key in self._by_key:
                    out.append(self._by_key[key])
                    break
                key = key[: key.rfind("(")] if key.endswith(")") else ""
            if not key and head != "4-10":          # "Item 1203" 这类没有根段落的：取其第一段
                first = next((i for i, c in enumerate(self.chunks) if _clause_key(c.citation).startswith(head)), None)
                if first is not None:
                    out.append(first)
        return list(dict.fromkeys(out))

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks:
            return []
        q = self.expand(query)
        ranks = []
        for scores in (self._bm25(q), self._dense(q)):
            order = np.argsort(-scores)
            r = np.empty(len(scores), dtype=int)
            r[order] = np.arange(len(scores))
            ranks.append(r)
        rrf = sum(1.0 / (RRF_K + r + 1) for r in ranks)
        pinned = self._explicit(query)
        rest = [i for i in np.argsort(-rrf) if i not in pinned]
        out = []
        for i in (pinned + rest)[:top_k]:
            d = self.chunks[i].to_dict()
            d["score"] = round(float(rrf[i]), 5)
            d["match"] = "citation" if i in pinned else "hybrid"
            out.append(d)
        return out

    def citations(self) -> List[str]:
        return [c.citation for c in self.chunks]

    # ---------------- 条款浏览 ----------------
    def get(self, citation: str) -> Optional[Dict]:
        i = self._by_key.get(_clause_key(citation))
        if i is None:
            return None
        c = self.chunks[i]
        d = c.to_dict()
        d["children"] = [dict(citation=x.citation, title=x.title) for x in self.chunks
                         if x.meta.get("parent") == c.citation]
        chain, p = [], c.meta.get("parent")
        while p:
            j = self._by_key.get(_clause_key(p))
            if j is None:
                break
            chain.append(dict(citation=self.chunks[j].citation, title=self.chunks[j].title))
            p = self.chunks[j].meta.get("parent")
        d["ancestors"] = list(reversed(chain))
        return d

    def toc(self) -> List[Dict]:
        return [dict(citation=c.citation, title=c.title, zh_title=c.meta.get("zh_title") or None,
                     level=c.meta.get("level", 0), parent=c.meta.get("parent"), kind=c.meta.get("kind"),
                     regulation=c.meta.get("regulation"), has_note="要点（非官方中文说明）" in c.text)
                for c in self.chunks]
