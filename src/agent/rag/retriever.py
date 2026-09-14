"""准则知识库检索（方案 §7.5）。

两个刻意的设计选择：

1) **按条款切分，不按固定长度切。**
   准则类文本的语义单元就是条款。固定长度切会把一条规则劈成两半，
   检索召回质量差很多。这里以 markdown 的 `## [条款号] 标题` 为切分点，
   条款号作为元数据保留，供强制引用使用。

2) **BM25 + 向量混合，RRF 融合。**
   条款检索里关键词（"五年"、"首日价格"）和语义都重要。
   向量侧用 TF-IDF + 余弦（sklearn 自带，内网可装）；
   要换成 bge 之类的嵌入模型，只替换 _dense_scores 即可。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ...config import path

CLAUSE_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.M)
RRF_K = 60


@dataclass
class Chunk:
    citation: str
    title: str
    text: str
    source: str

    def to_dict(self) -> Dict[str, str]:
        return dict(citation=self.citation, title=self.title,
                    text=self.text.strip(), source=self.source)


def _tokenize(s: str) -> List[str]:
    """中英混排的粗分词：英文按词、中文按字 + 二元组。无需额外分词依赖。"""
    s = s.lower()
    en = re.findall(r"[a-z0-9()\-\.]+", s)
    zh = re.findall(r"[一-鿿]", s)
    bi = ["".join(p) for p in zip(zh, zh[1:])]
    return en + zh + bi


def load_chunks(standards_dir: Optional[Path] = None) -> List[Chunk]:
    d = standards_dir or path("standards_dir")
    chunks: List[Chunk] = []
    for f in sorted(Path(d).glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        marks = list(CLAUSE_RE.finditer(raw))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            chunks.append(Chunk(citation=m.group(1).strip(), title=m.group(2).strip(),
                                text=raw[m.end():end], source=f.name))
    return chunks


class Retriever:
    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        docs = [f"{c.citation} {c.title} {c.text}" for c in self.chunks]
        self.tokens = [_tokenize(d) for d in docs]
        self.df: Dict[str, int] = {}
        for toks in self.tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = float(np.mean([len(t) for t in self.tokens])) if self.tokens else 1.0
        self.vec = TfidfVectorizer(analyzer=lambda s: _tokenize(s))
        self.M = self.vec.fit_transform(docs) if docs else None

    # ---------------- BM25 ----------------
    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = _tokenize(query)
        n = len(self.chunks)
        scores = np.zeros(n)
        for i, toks in enumerate(self.tokens):
            dl = len(toks) or 1
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
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

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks:
            return []
        ranks = []
        for scores in (self._bm25(query), self._dense(query)):
            order = np.argsort(-scores)
            r = np.empty(len(scores), dtype=int)
            r[order] = np.arange(len(scores))
            ranks.append(r)
        rrf = sum(1.0 / (RRF_K + r + 1) for r in ranks)
        idx = np.argsort(-rrf)[:top_k]
        out = []
        for i in idx:
            d = self.chunks[i].to_dict()
            d["score"] = round(float(rrf[i]), 5)
            out.append(d)
        return out

    def citations(self) -> List[str]:
        return [c.citation for c in self.chunks]
