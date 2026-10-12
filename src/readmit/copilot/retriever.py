"""TF-IDF retrieval over guidance documents, with a retrieval-quality evaluation.

Lexical retrieval suits a small, terminology-heavy corpus: it needs no GPU or external service and
is deterministic, so retrieval quality is testable in CI. The interface allows swapping in dense
embeddings if the corpus grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class Hit:
    doc: dict[str, Any]
    score: float


class TfidfRetriever:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            raise ValueError("no documents to index")
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        corpus = [f"{d['title']} {d['tags']} {d['tags']} {d['text']}" for d in docs]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.docs[i], float(scores[i])) for i in order if scores[i] > 0]


RETRIEVAL_EVAL_SET: list[tuple[str, str]] = [
    ("Congestive heart failure diuretic", "heart-failure-transition"),
    ("Chronic pulmonary disease inhaler", "copd-transition"),
    ("Diabetes with complications insulin", "diabetes-transition"),
    ("Moderate or severe chronic kidney disease", "kidney-disease"),
    ("Emergency department visits in the prior 6 months", "frequent-utilization"),
    ("Documented social needs transportation", "social-needs"),
    ("Dementia caregiver", "cognitive-impairment"),
    ("Polypharmacy distinct medications prescribed", "medication-reconciliation"),
    ("Hospitalizations in the prior 12 months follow-up appointment", "follow-up-visit"),
    ("What does the risk score mean and what are its limitations", "about-the-model"),
]


def evaluate_retrieval(
    retriever: TfidfRetriever, eval_set=RETRIEVAL_EVAL_SET, k: int = 3
) -> dict[str, float]:
    hits, ranks = 0, []
    for query, expected in eval_set:
        ids = [h.doc["id"] for h in retriever.search(query, top_k=k)]
        if expected in ids:
            hits += 1
            ranks.append(1 / (ids.index(expected) + 1))
        else:
            ranks.append(0.0)
    return {
        "hit_rate": hits / len(eval_set),
        "mrr": float(np.mean(ranks)),
        "n": len(eval_set),
        "k": k,
    }
