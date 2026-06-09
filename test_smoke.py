"""Smoke test: managers, training, attack, and curve run with sane bounds."""
import numpy as np
from memguard.embed import Embedder
from memguard.task import make_task, N_FEATURES
from memguard.managers import HeuristicManager, LearnedManager
from memguard.evaluate import run_benign_stream, poison_base_features
from memguard.attack import make_poison, asr_admit, degradation_curve


def test_smoke():
    emb = Embedder()
    task = make_task(n_subjects=15, n_relations=4, seed=1, embedder=emb)
    learned = LearnedManager(seed=1)
    store, acc = run_benign_stream(learned, task, train=True, seed=1)
    assert 0.0 <= acc <= 1.0
    heur = HeuristicManager()
    poison = make_poison(task, n=8, seed=1)
    pf = poison_base_features(task, poison, store, seed=1)
    assert all(f.shape == (N_FEATURES,) for f in pf)
    a_l = asr_admit(learned, pf, iters=20)
    a_h = asr_admit(heur, pf, iters=20)
    assert 0.0 <= a_l <= 1.0 and 0.0 <= a_h <= 1.0
    lc, hc = degradation_curve(learned, heur, pf, rounds=4, seed=1)
    assert len(lc) == 4 and len(hc) == 4
    assert all(0.0 <= x <= 1.0 for x in lc + hc)
    print("smoke OK", round(a_l, 3), round(a_h, 3))


if __name__ == "__main__":
    test_smoke()
