"""Two managers sharing the SAME feature inputs and content filter (fair test).

HeuristicManager  -- fixed-weight score + safety gate. Cannot be shifted by an
                     attacker's reward signal (no learning). MemGPT/Mem0 analog.
LearnedManager    -- small MLP admission policy trained by REINFORCE on the task's
                     marginal-utility reward. A-MAC-style learned admission scorer.
                     Has weights that (a) may overweight exploitable features and
                     (b) can be shifted by spoofed reward (the compounding surface).

Memory-R1's full LM-as-manager with sequential ADD/UPDATE/DELETE/NOOP RL is the
Phase-1 upgrade; the learned scorer here is the faithful, cheap kill-test version.
"""
from __future__ import annotations
import numpy as np
from .task import N_FEATURES

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:
    _TORCH = False


class HeuristicManager:
    name = "heuristic"
    # fixed weights over [relevance, novelty, importance, recency], safety-gated
    def __init__(self, w=(0.5, 0.3, 0.2, 0.0), tau: float = 0.45,
                 safety_thresh: float = 0.5):
        self.w = np.array(w, dtype=np.float32)
        self.tau = tau
        self.safety_thresh = safety_thresh

    def admit_prob(self, feats: np.ndarray) -> float:
        if feats[4] < self.safety_thresh:   # content_safety gate
            return 0.0
        score = float(np.dot(self.w, feats[:4]))
        return 1.0 if score >= self.tau else 0.0

    def decide(self, feats: np.ndarray) -> int:
        return int(self.admit_prob(feats) >= 0.5)

    # heuristic does not learn -> these are no-ops (the structural difference)
    def update(self, *a, **k):
        return
    def online_update(self, *a, **k):
        return


if _TORCH:
    class _PolicyNet(nn.Module):
        def __init__(self, d=N_FEATURES, h=16):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                     nn.Linear(h, 1))
        def forward(self, x):
            return torch.sigmoid(self.net(x)).squeeze(-1)


class LearnedManager:
    name = "learned"

    def __init__(self, lr: float = 5e-3, safety_thresh: float = 0.5, seed: int = 0):
        if not _TORCH:
            raise RuntimeError("PyTorch required for LearnedManager")
        torch.manual_seed(seed)
        self.net = _PolicyNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.safety_thresh = safety_thresh
        self.baseline = 0.0

    def admit_prob(self, feats: np.ndarray) -> float:
        # honors the same content filter as the heuristic (so any divergence is
        # NOT just "learned ignores the filter")
        if feats[4] < self.safety_thresh:
            return 0.0
        with torch.no_grad():
            p = self.net(torch.tensor(feats).float()).item()
        return float(p)

    def decide(self, feats: np.ndarray) -> int:
        return int(self.admit_prob(feats) >= 0.5)

    def update(self, feats: np.ndarray, action: int, reward: float):
        """REINFORCE step (contextual bandit): admit vs noop, observed reward."""
        x = torch.tensor(feats).float()
        p = self.net(x)
        # log-prob of the taken action
        logp = torch.log(p + 1e-6) if action == 1 else torch.log(1 - p + 1e-6)
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        adv = reward - self.baseline
        loss = -logp * adv
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def online_update(self, feats_batch: np.ndarray, rewards: np.ndarray,
                      action: int = 1):
        """Attacker-driven online update: a batch of (poison_features, spoofed
        reward) pairs, all with action=admit. This is the compounding mechanism."""
        for f, r in zip(feats_batch, rewards):
            self.update(f, action, float(r))
