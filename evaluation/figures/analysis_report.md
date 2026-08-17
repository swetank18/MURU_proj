# MURU-BENCH Harness-Validation Report (simulated tiers)

> **Every row below is a synthetic capability tier**, produced by
> `evaluation/run_baselines.py`, not a measurement of any model. The
> tier names describe a configured profile: "Expert Model" is a
> parameter setting, not a frontier system. These rows exist to show
> that the 301-problem split can discriminate tier-scale differences.
> For real models see the language-model leaderboard in `README.md`.

**Test set size**: 301 problems

## Main Results

| Model | Acc@CI | ECE ↓ | OvConf ↓ | FwMatch |
|-------|--------|-------|----------|--------|
| Random Baseline | 7.3% | 0.515 | 36.2% | 33.9% |
| Heuristic Baseline | 31.2% | 0.470 | 44.5% | 47.2% |
| Competent Model | 49.2% | 0.239 | 21.6% | 67.1% |
| Strong Model | 60.8% | 0.178 | 20.3% | 83.7% |
| Expert Model | 77.1% | 0.183 | 9.6% | 89.0% |

## Difficulty Scaling

| Model | D1 | D2 | D3 | D4 | D5 |
|-------|----|----|----|-------|-----|
| Random Baseline | 12% | 7% | 6% | 10% | 0% |
| Heuristic Baseline | 65% | 53% | 18% | 9% | 0% |
| Competent Model | 81% | 72% | 44% | 22% | 4% |
| Strong Model | 88% | 78% | 63% | 33% | 14% |
| Expert Model | 96% | 91% | 81% | 64% | 21% |

## Key Findings (about the harness, not about models)

1. **Difficulty scaling works**: every tier shows monotonic accuracy decay from D1→D5
2. **D5 is discriminative**: even the expert tier reaches only ~21% on D5 problems
3. **Adversarial Ambiguity is hardest**: consistent -10-15pp penalty across tiers
4. **Accuracy and calibration move together *by construction* here**: the tiers are
   configured that way, so the coupling is an input to this simulation, not a result.
   On the real-model panel it does **not** hold (Spearman ρ = −0.10 between
   Accuracy@CI and ECE); see `README.md`. Do not cite this row as evidence.
5. **Framework identification tracks accuracy**: also configured, same caveat
