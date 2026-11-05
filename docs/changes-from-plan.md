# Changes from the plan's Colab code

For Mudit and Abhinav to review, since most of this touches your modules. Please check the rows with your name and say if you disagree. These are ordinary code changes and any of them can be revisited.

The plan's five Colab cells run, but several numbers they produce aren't valid and would have ended up in the report. Their logic now lives in `src/cognitiverec/`, with the problems below fixed.

**Check this yourself:** `make audit` runs the plan's code as written (`scripts/audit_plan_code.py`) and prints every "as written" number on this page.

## Evaluation problems

| Problem as written | Evidence | Now | Review |
|---|---|---|---|
| **Fidelity is a ratio of two near-zero means**, so it's mostly noise. | Model C's ratio ranges from −12.21 to 1.67 across 10 seeds, with the same users and model. Deleting its "most attended" memory raises the score for 59.8% of users. | Paired difference with a confidence interval, plus top-1 change rate and Kendall τ. Compares against all other memories rather than one random pick, so no seed is involved. | Abhinav |
| **Baselines get hard-coded RQ3 and RQ4 scores** (0.00 fidelity, 100% noise retention) and look untested or perfect. | ItemKNN scores 3.53 to 3.90 on the plan's own deletion test. | Every model goes through the same tests. ItemKNN implements attributions. Models with none report NaN, never 0. | Abhinav, Mudit |
| **The noise pass isn't comparable with the clean pass**: only the clean pass updates exposure. | Model D's 0.8% retention doesn't come from noise: its NDCG is 0.0004 with no noise injected. | Both passes are identical apart from the noise, which is the same for all models. Zero noise gives exactly 100%, and a test enforces it. | Mudit |
| **The shift metric compares different things**: cognitive models are scored on later positives after seeing the first half of the stream, baselines on all positives after seeing nothing. | From the code: `Shift_NDCG@10` uses a different target set per model type. | All models get the same replay and the same targets. | Mudit |
| **"Genuine shift" isn't established.** | The rule flags 41.5% of users, and 52.4% of those rated everything in one sitting. Across all users, 59.6% rated in one sitting (55.0% on MovieLens 1M). | Renamed "observed drift" and kept as a secondary check. A simulated shift is the main test, as the brief suggests. | Mudit |
| **The evaluator is hard-coded to the test set**, so tuning on validation isn't possible, and the β sweep runs on test. | From the code: `evaluate_model_comprehensive` always reads `test_df`. | Separate `val` and `test` phases. A test runs tuning with no test split present. | All |
| **Validation items aren't masked at test time**, which wastes recommendation slots and penalises baselines most. | 13.2% of ItemKNN's top-10 slots go to items the user already rated in validation. Masking moves its NDCG from 0.1143 to 0.1332, against 0.0564 to 0.0607 for Model C. | Train and validation items are masked in one central place. | All |
| **"Recent" memory stops at the end of train**, skipping the 15% validation gap before test. | From the code: models never see validation interactions. | For the final test run, all models fit on train + validation. | Mudit |
| **The ablation grid can't test its own RQ2 claim**: the "no memory" row keeps exploration on. | From the code: `use_short_term=False` runs with `exploration_beta=0.18`. | Added long-term-only rows with β = 0, uniform attention, and both embedding choices. | Mudit, Abhinav |
| **"Zero future leakage" is overstated.** | On average, 39.9% of train rows are later in absolute time than a given test row. | Split kept and limit disclosed: a global split leaves 69.0% of its 226 test users with no history. See `docs/methodology.md`. | Mudit |

## Model changes

| Problem as written | Evidence | Now | Review |
|---|---|---|---|
| **The short-term score is divided by `√d`**, which isn't in the plan's formula. Memory barely reaches the ranking. | Short-term score spread is 4.5% of long-term. Top-10 lists overlap 0.91 with memory switched off. Removing the scaling takes that overlap to 0.53 and validation NDCG from 0.0658 to 0.0704. | Reward computed exactly as the plan's section 1 formula. A hand-worked test guards it. | Mudit |
| **Attention is really just recency.** | It favours the newest memory 99.8% of the time, so nearly every recommendation gets the same "explanation". | An explicit temperature `τ`, tuned on validation, with the plan's `√d` as one grid value. Lag stays as buffer position, because 75.8% of ratings share a timestamp. | Mudit |
| **RPE uses the long-term prediction only**, though the formula says the blended one, and barely moves anything. | Most users' α stays between 0.535 and 0.603. Top-10 lists overlap 0.969 between those two values. | Uses the blended `r̂`. `η` is in the search grid. | Mudit |
| **Exposure is counted once per test row, using a stale list, in fixed user order.** | From the code: `step_feedback` adds `top_k` on every row. Low user ids are always served first. | Counted once per recommendation list, with seeded shuffled user order. | Abhinav |
| **Normalisation drives the fairness gain, not cognition.** | With no short-term memory or exploration, L2 embeddings give 61.5% coverage and NDCG 0.0647, and raw embeddings give 14.9% and 0.1142 (validation). | `normalize_embeddings` is a flag, in the search grid and in the ablation table. | Mudit |

Smaller changes, all in `models/cognitive.py` and `models/exploration.py`:

- **Memory threshold:** one `positive_threshold` (4.0) replaces the plan's 3.0 at initialisation and 3.5 on update.
- **Valence weight:** the extra memory weighting is dropped, since it isn't in the formula.
- **α bounds:** 0.1 to 0.9, as in the formula, rather than 0.15 to 0.88.
- **α update:** keeps the code's smoothing rather than the formula's one-shot update, which would spike on a single noisy click.
- **Exposure start:** `N_exp` starts at the interaction count as in the formula, rather than count + 1.

## Added because the slides or brief promise it

- **Cold start** (slide 3): `evaluate_cold_start`.
- **Recommendation stability** (slide 2 and the brief): `evaluate_stability`.
- **Relevance-aware fairness** (the brief says exposure equality alone isn't fairness): `merit_gap` and recall by popularity band.
- **Precision@k and genre diversity** (the brief's RQ1 list).

## Results with these changes

These are provisional, from `make stage3` with the current tuned parameters. Test split, mean over 5 seeds. NaN means the model has no attributions to test.

| RQ | Measure | A: Popularity | B1: ItemKNN | B2: SVD | C: Cognitive | D: Cognitive + UCB |
|---|---|---|---|---|---|---|
| RQ1 | NDCG@10 | 0.088 | 0.157 | 0.110 | 0.148 | 0.145 |
| RQ1 | Recall@10 | 0.083 | 0.155 | 0.094 | 0.142 | 0.141 |
| RQ1 | Novelty (bits) | 1.28 | 2.14 | 1.95 | 2.14 | 2.48 |
| RQ2 | Coverage (%) | 4.6 | 26.1 | 17.2 | 23.8 | 39.6 |
| RQ2 | Gini | 0.987 | 0.904 | 0.945 | 0.919 | 0.897 |
| RQ2 | Merit gap | 0.718 | 0.404 | 0.495 | 0.394 | 0.356 |
| RQ2 | Tail recall | 0.0000 | 0.0097 | 0.0007 | 0.0125 | 0.0128 |
| RQ2 | Activity gap | 0.035 | 0.021 | 0.059 | 0.011 | 0.003 |
| RQ3 | Fidelity gap | NaN | 0.44 | NaN | 1.52 | 1.06 |
| RQ3 | Kendall τ | NaN | 1.00 | NaN | 0.31 | 0.30 |
| RQ3 | Top-1 change, strongest entry (%) | NaN | 21.4 | NaN | 42.0 | 40.0 |
| RQ3 | Top-1 change, other entries (%) | NaN | 5.7 | NaN | 3.0 | 3.1 |
| RQ4 | Replay NDCG | 0.075 | 0.116 | 0.087 | 0.114 | 0.108 |
| RQ4 | Adaptation gain, 5 injections | -0.004 | 0.003 | 0.003 | 0.045 | 0.053 |
| RQ4 | Retention after shift (%) | 100.5 | 99.7 | 91.5 | 92.9 | 90.7 |
| RQ4 | Noise retention, 5 ratings (%) | 100.3 | 99.5 | 93.0 | 95.8 | 91.8 |
| RQ4 | Stability | 0.993 | 0.938 | 0.923 | 0.906 | 0.880 |
| RQ4 | Cold-start NDCG, 5 ratings | 0.082 | 0.074 | 0.066 | 0.072 | 0.070 |
| RQ4 | Cold-start NDCG, full history | 0.084 | 0.149 | 0.104 | 0.130 | 0.129 |

Paired differences over users against ItemKNN, with 95% bootstrap intervals. An interval that includes zero means the difference could come from which users were sampled.

| Model | Compared with | NDCG | Replay NDCG |
|---|---|---|---|
| A: Popularity | B1: ItemKNN | -0.0689 [-0.0838, -0.0550] | -0.0412 [-0.0567, -0.0254] |
| B2: SVD | B1: ItemKNN | -0.0470 [-0.0606, -0.0330] | -0.0299 [-0.0443, -0.0153] |
| C: Cognitive | B1: ItemKNN | -0.0096 [-0.0226, +0.0023] | -0.0025 [-0.0163, +0.0109] |
| D: Cognitive + UCB | B1: ItemKNN | -0.0126 [-0.0246, -0.0006] | -0.0081 [-0.0230, +0.0063] |
| D: Cognitive + UCB | C: Cognitive | -0.0030 [-0.0052, -0.0009] | -0.0056 [-0.0080, -0.0034] |

### Ablations

Test split, mean over 3 seeds. Each row changes tuned Model D as named. RPE can't affect NDCG, fairness or fidelity (see `docs/methodology.md`), so its row only differs under replay and RQ4 tests. The "w/o exploration" and "Model C, raw embeddings" rows are the same model, because the search chose raw embeddings.

| Row | NDCG | Replay NDCG | Coverage (%) | Merit gap | Tail recall | Fidelity gap | Adaptation gain | Noise retention, 5 (%) | Stability |
|---|---|---|---|---|---|---|---|---|---|
| Full Model D | 0.145 | 0.109 | 39.5 | 0.356 | 0.0131 | 1.06 | 0.053 | 91.4 | 0.881 |
| w/o exploration (Model C) | 0.148 | 0.114 | 23.8 | 0.394 | 0.0125 | 1.52 | 0.045 | 95.8 | 0.906 |
| w/o RPE adaptation | 0.145 | 0.109 | 39.5 | 0.356 | 0.0131 | 1.06 | 0.032 | 95.7 | 0.891 |
| w/o short-term memory | 0.140 | 0.099 | 35.2 | 0.378 | 0.0102 | NaN | 0.002 | 98.8 | 0.936 |
| uniform attention | 0.134 | 0.095 | 45.1 | 0.335 | 0.0095 | -0.00 | 0.012 | 93.4 | 0.868 |
| long-term only, L2 embeddings | 0.070 | 0.048 | 61.4 | 0.130 | 0.0305 | NaN | 0.003 | 97.3 | 0.940 |
| long-term only, raw embeddings | 0.142 | 0.101 | 23.8 | 0.400 | 0.0104 | NaN | 0.003 | 99.5 | 0.948 |
| Model C, L2 embeddings | 0.074 | 0.056 | 64.3 | 0.185 | 0.0347 | 0.77 | 0.097 | 36.8 | 0.881 |
| Model C, raw embeddings | 0.148 | 0.114 | 23.8 | 0.394 | 0.0125 | 1.52 | 0.045 | 95.8 | 0.906 |

Paired differences against full Model D, plus short-term memory's own contribution with exploration off:

| Model | Compared with | NDCG | Replay NDCG |
|---|---|---|---|
| w/o exploration (Model C) | Full Model D | +0.0027 [+0.0006, +0.0048] | +0.0054 [+0.0033, +0.0077] |
| w/o RPE adaptation | Full Model D | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0018, +0.0019] |
| w/o short-term memory | Full Model D | -0.0048 [-0.0105, +0.0007] | -0.0093 [-0.0162, -0.0024] |
| uniform attention | Full Model D | -0.0110 [-0.0155, -0.0068] | -0.0133 [-0.0200, -0.0070] |
| long-term only, L2 embeddings | Full Model D | -0.0745 [-0.0855, -0.0635] | -0.0602 [-0.0730, -0.0476] |
| long-term only, raw embeddings | Full Model D | -0.0028 [-0.0087, +0.0029] | -0.0080 [-0.0152, -0.0011] |
| Model C, L2 embeddings | Full Model D | -0.0707 [-0.0809, -0.0610] | -0.0526 [-0.0652, -0.0404] |
| Model C, raw embeddings | Full Model D | +0.0027 [+0.0006, +0.0049] | +0.0054 [+0.0033, +0.0078] |
| Model C, raw embeddings | long-term only, raw embeddings | +0.0055 [+0.0004, +0.0107] | +0.0134 [+0.0068, +0.0205] |
| Model C, L2 embeddings | long-term only, L2 embeddings | +0.0038 [-0.0018, +0.0096] | +0.0076 [+0.0010, +0.0148] |

## Open questions

**Mudit: RPE doesn't yet measure surprise, which blocks H3.** Mean |δ| by rating in the plan's code is 1★ 0.49, 2★ 0.17, 3★ 0.26, 4★ 0.55, 5★ 0.87: it mostly tracks how high the rating is, because predicted rewards are small next to the reward scale. The blended prediction doesn't fix this. Stage 4 shows the consequence: with RPE on, Model D's adaptation gain is 0.053 against 0.032 without it, but its retention under noise falls from 95.7% to 91.4% at 5 ratings (97.9% to 97.4% at 2). So RPE does raise responsiveness, but to noise as well as to shifts, which contradicts the second half of H3. Injected noise is 5-star, the case where |δ| is largest, so the scale problem may be part of this. Calibrating the prediction, or using a sustained-surprise signal, is a modelling decision for you. Until then, RPE results aren't evidence for H3.

**Mudit and Abhinav: which embeddings define Model C?** The plan specified L2 normalisation, but the search chose raw embeddings by a wide margin. Current code follows the search and reports L2 as a control row. On the test split, Model C scores NDCG 0.148 with 23.8% coverage on raw embeddings, and NDCG 0.074 with 64.3% coverage on L2. L2 also gives a better merit gap (0.185 against 0.394) and tail recall (0.0347 against 0.0125), but only 36.8% noise retention at 5 ratings against 95.8%. This is the largest single effect in the system, and it isn't a cognitive mechanism, so the report should say so. It also bears on H1: with raw embeddings, long-term memory alone sits in the same range as the baselines (23.8% coverage and Gini 0.920, against 26.1% and 0.904 for ItemKNN and 17.2% and 0.945 for SVD), and with L2 it spreads exposure far more (61.4% and 0.790), so "amplifies popularity bias" isn't supported under either choice. Adding short-term memory doesn't change coverage either (23.8% with or without it, on raw embeddings). Exploration is what moves it, to 39.5%.

**Abhinav: exploration spreads exposure more than it helps relevant tail items.** Against Model C, Model D raises coverage from 23.8% to 39.6% and improves the merit gap from 0.394 to 0.356, for an NDCG cost of 0.0030. But tail recall barely moves, from 0.0125 to 0.0128. By the fairness definition in `docs/methodology.md`, that supports a modest merit-based claim, but not an equal-opportunity one for tail items. The plan's β = 0.18 sits at a 12.3% NDCG loss on validation, past the 10% budget. A fairness-aware bonus could be worth trying as an extension.

**Abhinav: fidelity needs care in the write-up.** Model D's gap is 1.06 (95% interval 0.97 to 1.14), against -0.00 (-0.02 to 0.01) for the same model with uniform attention, so attention does identify influential memories. But its τ is only 0.30, so attention is moderately correlated with what deletion actually does. Compare it with ItemKNN through τ and top-1 change rather than the gap, and remember that ItemKNN is faithful by construction.

**All: Stage 2 is a modest random search.** Top trials are close together and a few values sit at grid edges, so treat the tuned parameters as reasonable rather than optimal. Widening `configs/stage2_cognitive.yaml` is cheap.
