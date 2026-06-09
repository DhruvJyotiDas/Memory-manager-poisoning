"""Benign-stream training of the learned manager + utility evaluation + building
representative poison feature contexts.

Coupling that makes the task learnable: a candidate's observable `relevance`
correlates with its true marginal utility on benign data, so the learned policy
learns "relevant/novel/important -> admit". Poison deliberately BREAKS this
coupling (high relevance, negative true utility) -- which is why a policy that
keyed on relevance can be exploited.
"""
from __future__ import annotations
import random
import numpy as np
from .task import Task, Fact, extract_features, marginal_utility


def _query_embs(task: Task, window):
    return [task.query_emb(q) for q in window]


def run_benign_stream(manager, task: Task, train: bool, lookahead: int = 6,
                      win: int = 4, seed: int = 0):
    """Process benign candidates; (optionally) train the learned manager via the
    marginal-utility reward. Returns (store, benign_accuracy)."""
    rng = random.Random(seed)
    cands = [f for f in task.facts]
    rng.shuffle(cands)
    qstream = list(task.queries)
    rng.shuffle(qstream)
    store: list[Fact] = []
    T = len(cands)
    for t, cand in enumerate(cands):
        window = qstream[t % len(qstream): t % len(qstream) + win]
        if not window:
            window = qstream[:win]
        qembs = _query_embs(task, window)
        look = [qstream[(t + i) % len(qstream)] for i in range(lookahead)]
        feats = extract_features(cand, store, qembs, t, T)
        if train and hasattr(manager, "update") and manager.name == "learned":
            p = manager.admit_prob(feats)
            action = int(rng.random() < max(min(p, 0.99), 0.01))  # sample
            reward = marginal_utility(cand, store, task, look)
            manager.update(feats, action, reward)
            if action == 1:
                store.append(cand)
        else:
            if manager.decide(feats) == 1:
                store.append(cand)
    acc = benign_accuracy(store, task)
    return store, acc


def benign_accuracy(store: list[Fact], task: Task, topk: int = 5) -> float:
    """Fraction of queries whose gold fact is present in the store (retrievable)."""
    if not store:
        return 0.0
    have = {}
    for s in store:
        have.setdefault(s.key(), []).append(s)
    ok = 0
    for q in task.queries:
        cands = have.get(q, [])
        if any(c.obj == task.gold[q] and not c.is_poison for c in cands):
            ok += 1
    return ok / len(task.queries)


def poison_base_features(task: Task, poison: list[Fact], store: list[Fact],
                         win: int = 4, seed: int = 0) -> list[np.ndarray]:
    """Observable features of poison candidates in a representative context (a
    trained store + recent queries targeting the poison's keys)."""
    feats = []
    for i, p in enumerate(poison):
        window = [p.key()] + task.queries[:win - 1]  # attacker targets p's topic
        qembs = _query_embs(task, window)
        feats.append(extract_features(p, store, qembs, step=len(store), total_steps=len(task.facts)))
    return feats


def calibrate_heuristic(task: Task, seed: int = 0,
                        grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Pick the heuristic threshold that MAXIMIZES benign accuracy -> a fair,
    strong baseline (not a strawman that admits nothing or everything)."""
    from .managers import HeuristicManager
    best_tau, best_acc = grid[len(grid) // 2], -1.0
    for tau in grid:
        m = HeuristicManager(tau=tau)
        store, _ = run_benign_stream(m, task, train=False, seed=seed)
        acc = benign_accuracy(store, task)
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    return HeuristicManager(tau=best_tau), best_tau, best_acc
