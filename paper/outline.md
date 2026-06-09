# Poisoning the Gatekeeper: Adversarial Robustness of Learned Memory Managers in LLM Agents

## Target: AISec @ CCS 2026 (6 pages + references)
## Deadline: check https://aisec.cc

## §1 Introduction (0.5 pages)
- LLM memory is migrating from heuristic to RL-trained managers (Memory-R1, A-MAC)
- Existing attacks target retrieval store under FIXED manager (MINJA, MemoryGraft)
- We show the LEARNED MANAGER ITSELF is the attack surface
- Three findings: TM-B reward-hacking (0.15->1.0 in 1 round), defense works (flat at 0.15), TM-C eviction (1.0->0.0 in 1 round)

## §2 Background (0.75 pages)
- Learned memory managers: Memory-R1 (ADD/UPDATE/DELETE/NOOP via GRPO), A-MAC
- Existing attacks: MINJA, MemoryGraft, Zombie Agents (all retrieval-level, treat manager as fixed)
- RL poisoning literature: Zhu/Ma 2019, Rakhsha 2020 (not connected to LLM memory)
- The gap: nobody has attacked the learned manager's policy

## §3 Threat Models (0.5 pages)
- TM-B: reward-hacking via spoofed task success → policy drifts to admit attacker content
- TM-C: redundancy-spoofing → policy learns to evict safety memories
- What makes these distinct from retrieval attacks: targets the policy, not the store

## §4 Experiments (2 pages)
- Setup: LongMemEval oracle, Qwen2.5-0.5B LoRA, heuristic baseline, prompted baseline
- Figure 1: Gatekeeper Degradation Curve (TM-B: trained 0.15->1.0, prompted 0.0 flat)
- Figure 2: Defense comparison (undefended 0.15->1.0, defended flat at 0.15, utility preserved)  
- Figure 3: TM-C Survival curve (trained 1.0->0.0, heuristic fixed at 1.0)
- Honest limitations: SFT proxy, single family, single seed

## §5 Defense: Provenance-Gated Reward (0.75 pages)
- Mechanism: training loss masked for low-trust provenance
- Results: slope -0.000 vs +0.071 undefended
- Zero utility cost: 1.0 -> 1.0 benign utility
- Comparison to A-MemGuard (content-level): A-MemGuard cannot prevent policy drift

## §6 Discussion + Limitations (0.5 pages)
- SFT as proxy for RL: limitations, why GRPO results are future work
- Scale: one model family, one seed
- Ethics: responsible disclosure to Memory-R1 authors

## §7 Conclusion (0.25 pages)
