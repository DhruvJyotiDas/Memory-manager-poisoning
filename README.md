# MemGuard — Poisoning the Gatekeeper (kill-test)

**Adversarial robustness of learned agent-memory managers.** This repo is the
**Week 1–2 kill-test**, not the paper. It answers one question before you commit:

> Does a **learned** admission policy (A-MAC / Memory-R1 style) admit content-benign
> poison **differently / exploitably** vs a matched **heuristic** (Mem0/MemGPT style),
> and does its poison-admission **rise under online updates** while the heuristic
> stays flat?

If learned ≈ heuristic and the degradation curve is flat → you have "MINJA with
extra steps" → pivot. If the learned manager diverges or its admission compounds →
GO, build the paper.

## Run it

```bash
pip install -r requirements.txt
python run_killtest.py
python tests/test_smoke.py          # or: PYTHONPATH=. python tests/test_smoke.py
```

Output: TM-A ASR-admit (learned vs heuristic), the **Gatekeeper Degradation Curve**
(`degradation_curve.png`) — the headline figure — and a GO/NO-GO verdict.

### Real verdict needs real embeddings

Without `sentence-transformers`, the harness uses deterministic **fallback**
embeddings so it runs anywhere — but relevance/novelty become noise, every manager
saturates, and the numbers are **harness-validation only, not evidence**. Install
`sentence-transformers` and re-run for a real verdict. The script tells you which
mode it's in.

## Why this is a *fair* test (reviewers will probe this)

- **Same inputs.** Heuristic and learned managers see the identical 6-feature
  observation per candidate (`task.py: extract_features`).
- **Same content filter.** Both gate on `content_safety`; the learned manager does
  not get to "cheat" by ignoring it. Any divergence is therefore about the
  *admission policy*, not filtering.
- **Strong baseline.** The heuristic threshold is **auto-calibrated to maximize
  benign accuracy** (`evaluate.py: calibrate_heuristic`) — not a strawman that
  admits nothing or everything.
- **Poison optimized against *each* manager.** TM-A runs a black-box search to
  maximize admission under the heuristic *and* under the learned policy separately,
  so we are not only attacking the learned one.
- **Honest decoupling.** Poison is content-benign (passes the filter) with high
  observable relevance/novelty but **negative true utility**. The learned policy,
  trained to admit relevant/novel content on benign data, is exploited precisely
  because poison mimics those features. That decoupling is the whole thesis.

## What the verdict means

- **(A) TM-A surface divergence** — learned admits poison ≥ 0.10 more than heuristic.
- **(B) compounding via update** — under spoofed-success online updates, the learned
  manager's admission of a *fixed held-out poison probe* has clearly positive slope
  while the heuristic is flat. This is the structurally novel claim (policy
  compounding ≠ retrieval-frequency compounding) and the headline figure.

GO if **A or B**. The compounding curve (B) is the one that distinguishes you from
all the retrieval-poisoning work.

## What is deliberately simplified for the kill-test (Phase-1 TODO)

1. **Learned manager = lightweight RL admission scorer** (A-MAC style: MLP +
   REINFORCE bandit). Memory-R1's full **LM-as-manager** with sequential
   ADD/UPDATE/DELETE/NOOP RL is the Phase-1 upgrade.
2. **Verifiable task, no agent LLM.** Success = gold fact retrievable from the store,
   so the reward is clean and you skip MIMIC-III/EHRAgent credentialing this week.
   The downstream **ASR-r / ASR-a / ASR-t** (poison influences retrieval → thought →
   action) needs the agent LLM (Qwen2.5-7B via vLLM) — that's Phase 1.
3. **TM-A + TM-B only.** TM-C (denial-of-memory eviction, the cleanest-unoccupied
   sub-cell) and the defense matrix (provenance-gated reward vs A-MemGuard/content
   baselines) are Phase 1–2.
4. **Single task.** Add EHRAgent + StrategyQA + ASB for external validity later.

## Files

```
memguard/embed.py      Embedder (sentence-transformers or fallback) + cosine
memguard/task.py       verifiable multi-fact QA; candidate features; true utility
memguard/managers.py   HeuristicManager (fixed, calibrated) + LearnedManager (RL)
memguard/attack.py     poison; per-manager admission optimizer (TM-A); compounding (TM-B)
memguard/evaluate.py   benign training/eval, utility, heuristic calibration
run_killtest.py        the gate: TM-A + degradation curve + verdict + figure
```

## Order of work

1. `python run_killtest.py` with **real embeddings** → the real go/no-go.
2. If GO: swap in the Memory-R1 LM-manager; add the agent LLM for downstream ASR;
   add TM-C eviction; build the provenance-gated-reward defense vs content-level
   baselines; second benchmark. → AISec/SaTML workshop, then a security venue.
3. Ship a workshop flag fast — this area moves monthly and the cell is open *now*.

Ethics: this is attack+defense safety research. Lead with defenses, use
reimplementations/benchmarks, and plan responsible disclosure to manager authors
before any public release of optimized attacks.
