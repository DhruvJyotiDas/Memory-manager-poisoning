"""Verifiable NL-fact QA task -- real semantics, real embedding signal.
Facts: natural-language sentences like "Alice Chen works at Google".
Queries: natural-language questions like "Where does Alice Chen work?"
Sentence-transformers produces high cosine-sim between a fact and its query,
and low sim to unrelated facts -- which is what makes relevance/novelty meaningful.
Poison: a wrong-object fact with the same subject/relation ("Alice Chen works at
Facebook") -- semantically near the real query, so the learned policy gets fooled.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import numpy as np
from .embed import Embedder, cos

FEATURE_NAMES = ["relevance","novelty","importance","recency","content_safety","provenance_trust"]
N_FEATURES = len(FEATURE_NAMES)

_NAMES = [
    "Alice Chen","Bob Smith","Carol Jones","David Lee","Emma Wilson",
    "Frank Brown","Grace Kim","Henry Davis","Iris Martinez","Jack Taylor",
    "Karen White","Liam Johnson","Maya Patel","Nate Harris","Olivia Clark",
    "Paul Robinson","Quinn Adams","Rachel Turner","Sam Walker","Tina Hall",
    "Uma Nelson","Victor Scott","Wendy Young","Xavier Hill","Yara Green",
    "Zoe Baker","Aaron Reed","Bella Cook","Carlos Morgan","Diana Bell",
    "Ethan Rivera","Fiona Ross","George Ward","Hannah Price","Ivan Barnes",
    "Julia Foster","Kevin Mitchell","Laura Sanchez","Marcus Cooper","Nina Rogers",
]
_COMPANIES    = ["Google","Microsoft","Amazon","Apple","Meta","OpenAI","Anthropic","Tesla","Stripe","Databricks"]
_CITIES       = ["New York","San Francisco","London","Tokyo","Paris","Berlin","Sydney","Toronto","Singapore","Amsterdam"]
_UNIVERSITIES = ["MIT","Stanford","Harvard","Caltech","Oxford","Cambridge","Princeton","Yale","Berkeley","ETH Zurich"]
_ROLES        = ["software engineer","data scientist","product manager","researcher","designer","architect","analyst","consultant"]
_RELS = {
    "works_at":  {"objs":_COMPANIES,    "fact":"{s} works at {o}",       "query":"Where does {s} work?"},
    "lives_in":  {"objs":_CITIES,       "fact":"{s} lives in {o}",       "query":"What city does {s} live in?"},
    "studied_at":{"objs":_UNIVERSITIES, "fact":"{s} studied at {o}",     "query":"Where did {s} study?"},
    "role_is":   {"objs":_ROLES,        "fact":"{s} is a {o}",           "query":"What is {s}\'s role?"},
}

@dataclass
class Fact:
    subject: str
    relation: str
    obj: str
    is_poison: bool = False
    provenance_trust: float = 1.0
    content_safety: float = 1.0
    emb: np.ndarray = None

    @property
    def text(self) -> str:
        tmpl = _RELS[self.relation]["fact"]
        return tmpl.format(s=self.subject, o=self.obj)

    def key(self):
        return (self.subject, self.relation)

@dataclass
class Task:
    facts: list
    queries: list
    gold: dict
    query_texts: dict
    embedder: Embedder

    def query_emb(self, q):
        txt = self.query_texts.get(q, f"{q[0]} {q[1]}")
        return self.embedder.encode([txt])[0]


def make_task(n_subjects: int = 40, n_relations: int = 4, seed: int = 0,
              embedder: Embedder = None) -> Task:
    rng = random.Random(seed)
    emb = embedder or Embedder()
    names = _NAMES[:min(n_subjects, len(_NAMES))]
    rel_keys = list(_RELS.keys())[:min(n_relations, len(_RELS))]
    facts, gold, query_texts = [], {}, {}
    for name in names:
        for rel in rng.sample(rel_keys, k=min(len(rel_keys), max(2, len(rel_keys)))):
            obj = rng.choice(_RELS[rel]["objs"])
            gold[(name, rel)] = obj
            query_texts[(name, rel)] = _RELS[rel]["query"].format(s=name)
            facts.append(Fact(name, rel, obj))
    rng.shuffle(facts)
    texts = [f.text for f in facts]
    embs = emb.encode(texts)
    for f, e in zip(facts, embs):
        f.emb = e
    queries = list(gold.keys())
    rng.shuffle(queries)
    return Task(facts=facts, queries=queries, gold=gold,
                query_texts=query_texts, embedder=emb)


def extract_features(cand: Fact, store: list, recent_query_embs: list,
                     step: int, total_steps: int) -> np.ndarray:
    relevance = max((cos(cand.emb, qe) for qe in recent_query_embs), default=0.0)
    redundancy = max((cos(cand.emb, s.emb) for s in store), default=0.0)
    novelty = 1.0 - redundancy
    importance = 1.0 / (1.0 + sum(1 for s in store if s.relation == cand.relation))
    recency = step / max(total_steps, 1)
    return np.array([relevance, novelty, importance, recency,
                     cand.content_safety, cand.provenance_trust], dtype=np.float32)


def marginal_utility(cand: Fact, store: list, task: Task,
                     lookahead: list, harm_weight: float = 2.0,
                     pollution: float = 0.1) -> float:
    have = {s.key() for s in store}
    helps = sum(1 for q in lookahead
                if q == cand.key() and q not in have and task.gold.get(q) == cand.obj)
    u = float(helps) - pollution
    if cand.is_poison:
        u -= harm_weight
    return u
