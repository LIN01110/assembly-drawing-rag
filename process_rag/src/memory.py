"""记忆库：历史工艺卡检索。

三种方法论（消融臂）：
  cbr    — 案例推理：按 材料类别 + 公差等级带 + 零件类型关键词 结构化过滤
  vector — 文本相似度（TF-IDF 字符 n-gram，零外部模型依赖）全库排序
  hybrid — CBR 过滤圈定候选集 → TF-IDF 排序（默认，可解释 + 泛化）
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

CARDS_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "process_cards.json"

_materials_cache: dict | None = None


def load_cards() -> list[dict]:
    return json.loads(CARDS_PATH.read_text(encoding="utf-8"))


def _material_category(grade: str) -> str:
    global _materials_cache
    if _materials_cache is None:
        from src import db
        conn = db.get_conn()
        _materials_cache = {r["grade"]: r["category"] for r in conn.execute("SELECT grade, category FROM materials")}
        conn.close()
    return _materials_cache.get(grade, "")


def card_text(card: dict) -> str:
    return f"{card['part_type']} {card['material']} {card['features']} {' '.join(card['steps'])} {card['notes']}"


def part_text(part: dict) -> str:
    return " ".join(str(part.get(k, "")) for k in ("part_type", "material", "features", "tech_requirements"))


class MemoryStore:
    """历史工艺卡记忆库。"""

    def __init__(self):
        self.cards = load_cards()
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self._matrix = self._vectorizer.fit_transform([card_text(c) for c in self.cards])

    def _tfidf_rank(self, part: dict, candidates: list[tuple[int, dict]], top_k: int) -> list[tuple[dict, float]]:
        query = self._vectorizer.transform([part_text(part)])
        scored = []
        for idx, card in candidates:
            score = float((query @ self._matrix[idx].T).toarray()[0][0])
            scored.append((card, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _cbr_filter(self, part: dict) -> list[tuple[int, dict]]:
        """结构化过滤：材料同类 + IT 等级带（±2）+ 零件类型词命中加分。"""
        grade = part.get("material", "")
        category = _material_category(grade)
        it = part.get("tolerance_it")
        ptype = part.get("part_type", "")
        out = []
        for idx, card in enumerate(self.cards):
            score = 0
            if card["material"] == grade:
                score += 2
            elif _material_category(card["material"]) == category and category:
                score += 1
            if it is not None and abs(card["tolerance_it"] - int(it)) <= 2:
                score += 1
            if ptype and any(kw in card["part_type"] or card["part_type"] in ptype
                             for kw in re.split(r"[/、\s]+", ptype) if kw):
                score += 2
            if score >= 2:  # 至少材料同类或（同牌号+类型）
                out.append((idx, card, score))
        out.sort(key=lambda x: -x[2])
        return [(idx, card) for idx, card, _ in out]

    def recall(self, part: dict, method: str = "hybrid", top_k: int = 3) -> list[tuple[dict, float]]:
        """召回相似历史工艺卡。method ∈ {cbr, vector, hybrid}。"""
        if method == "vector":
            return self._tfidf_rank(part, list(enumerate(self.cards)), top_k)
        if method == "cbr":
            return [(c, 1.0) for _, c in self._cbr_filter(part)[:top_k]]
        # hybrid：CBR 候选集 → TF-IDF 排序；CBR 空集时回退全库
        candidates = self._cbr_filter(part) or list(enumerate(self.cards))
        return self._tfidf_rank(part, candidates, top_k)


class GBStore:
    """GB 标准条款库（TF-IDF 检索）。"""

    GB_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "gb_standards.json"

    def __init__(self):
        self.clauses = json.loads(self.GB_PATH.read_text(encoding="utf-8"))
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self._matrix = self._vectorizer.fit_transform(
            [f"{c['code']} {c['topic']} {c['text']}" for c in self.clauses])

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        q = self._vectorizer.transform([query])
        scores = (q @ self._matrix.T).toarray()[0]
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [(self.clauses[i], float(s)) for i, s in ranked]
