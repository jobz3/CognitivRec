# Methodology

How CognitiveRec is evaluated, and what its results can and can't claim. Use this as the basis for the report's methodology and limitations sections.

## Data and split

MovieLens 100K: 943 users, 1,682 items, 100,000 ratings. Each user's ratings are split in time order into 70% train, 15% validation and 15% test (69,572 / 14,973 / 15,455 rows). A rating counts as relevant at 4 stars or above.

### What the split doesn't protect

Stage 1 prints these numbers on every run (`temporal_report` in `data/split.py`):

| Measure | Value |
|---|---|
| Users who rated everything in one sitting (no gap over 30 minutes) | 59.6% |
| Ratings sharing an exact timestamp with another rating by the same user | 75.8% |
| Median gap between a user's last train rating and first test rating | 4.9 minutes |
| Mean share of train rows later in absolute time than a given test row | 39.9% |

This has three consequences:

1. **The split is not free of leakage in the global sense, and the report must say so.** A user's own future never reaches their training data, but other users' later ratings do. We keep the per-user split because a global time split would leave 156 of its 226 test users (69%) with no training history.
2. **Real taste changes can't be observed reliably from these timestamps.** Order within one sitting mostly reflects the rating interface. RQ4's main test is therefore simulated, as the brief suggests. Switching to MovieLens 1M wouldn't help: 55% of its users also rated in one sitting.
3. **Recency uses position, not seconds.** Model C measures lag as position in the memory buffer, because most timestamps tie.

## Evaluation protocol

Every model goes through `evaluation/protocol.py`, the only place that masks candidates, takes the top-k and sets user order.

| Phase | Models fit on | Candidates exclude | Predicting | Used for |
|---|---|---|---|---|
| `val` | train | train | validation positives | all tuning and trade-off curves |
| `test` | train + validation | train + validation | test positives | final results only |

- **Candidates:** every catalogue item the user hasn't interacted with. There is no negative sampling.
- **List length:** k = 10.
- **User order:** seeded and shuffled, because Model D's exposure counts depend on who was served first. Results are averaged over 5 seeds (3 for Stage 4).
- **Comparisons:** paired over users, with 95% bootstrap intervals.

## Tuning

Selection only ever sees validation data. A test in `tests/test_stages.py` runs tuning with no test split present.

- **Baselines (Stage 1):** grid search on validation NDCG@10.
- **Model C (Stage 2):** random search over 100 trials plus the defaults, on validation *replay* NDCG. RPE only acts on observed feedback, so a one-shot ranking can't measure it.
- **Model D (Stage 2):** the exploration weight β with the highest coverage that costs at most 10% of Model C's validation NDCG. Choosing by NDCG alone would always pick β = 0.

## What each research question measures

### RQ1: quality

Precision, Recall and NDCG at 10, novelty (mean self-information of recommended items, from training popularity) and diversity (1 minus mean pairwise genre Jaccard similarity within a list).

### RQ2: fairness

The brief requires an explicit definition, and warns that exposure equality alone isn't fairness. We report three separate things, and **a fairness claim needs measure 2 or 3 to improve, not measure 1 alone**.

| # | Measure | Metrics | What it shows |
|---|---|---|---|
| 1 | Exposure inequality | `coverage_pct`, `gini` | How evenly exposure spreads. On its own this is a diversity result. |
| 2 | Merit-based item fairness | `merit_gap` | Distance between each popularity band's share of exposure and its share of relevant test items. 0 means exposure is proportional to observed relevance. |
| 3 | Equal opportunity for items | `recall_head`, `recall_mid`, `recall_tail` | The chance that a relevant item gets recommended, by popularity band. |

Bands hold about a third of the interactions each. We also report `activity_gap`, the NDCG difference between users above and below median activity, as a user-side measure.

### RQ3: explanation fidelity

For each user's top recommendation, each piece of evidence is deleted in turn (memory entries for Models C and D, history items for ItemKNN).

| Metric | Meaning |
|---|---|
| `fidelity_gap` | Effect of deleting the highest-attributed entry minus the mean effect of deleting any other, in units of that user's score spread, with a 95% interval |
| `top1_change_strongest_pct`, `top1_change_other_pct` | How often the deletion changes the top recommendation |
| `kendall_tau` | Rank correlation between attributions and actual deletion effects (the Jain and Wallace test) |
| `score_rises_pct` | How often deleting the top-attributed entry raises the score, which contradicts the explanation |

How to read these:

- **Compare `fidelity_gap` within a model family, not across families.** It depends on how many evidence entries a model has (currently at most 24 memories, against about 90 history items for ItemKNN) and on the model's score spread.
- **ItemKNN's τ of 1.0 holds by construction.** Its score is a sum of its attributions, so it's a comparison point rather than a discovery.
- **Uniform attention is the control.** Its attributions carry no information, so its gap should be about 0, and it is. A gap above that control is the evidence that attention identifies something real.
- **Models with no attributions report NaN.** Popularity and SVD weren't tested on this question, which is different from scoring 0.
- **Deleting a memory leaves long-term memory unchanged,** so this tests short-term attention only.

### RQ4: reliability

| Test | Metrics | Method |
|---|---|---|
| Simulated shift (main test) | `shift_adaptation`, `shift_adaptation_gain`, `shift_retention_pct` | Inject 5-star ratings from a genre the user rarely rates. Adaptation is the top-k share in that genre, and retention is NDCG on the user's original positives. Same injected items for all models. |
| Replay | `replay_ndcg`, `drift_ndcg`, `no_drift_ndcg` | Feed all models the first half of each user's targets, then score the later positives. Drift users are those whose top genre differs between history and targets: a label, not evidence of changed taste. |
| Noise | `noise_retention_<n>_pct` | NDCG with n random 5-star ratings injected, as a percentage of clean NDCG. Both passes are otherwise identical, so zero noise gives exactly 100. |
| Stability | `stability` | Top-k Jaccard overlap between consecutive lists as interactions arrive. High isn't automatically good: Popularity scores close to 1 by never adapting. |
| Cold start | `cold_ndcg_<n>`, `cold_ndcg_full` | Refit with 20% of users cut to their first n interactions; everyone else keeps their full history. |

A one-shot ranking never calls `observe`, so **RPE can't affect RQ1, RQ2 or RQ3 results**. Its ablation row matches the full model there by construction. Its effect only shows in RQ4.

Adaptation numbers need the gain beside them, because popular items can already fall in the new genre: Popularity has a high `shift_adaptation` and a gain of about zero.

## Hypotheses

The plan calls these paradoxes. The brief requires treating them as hypotheses, and any of them failing is a valid result.

| | Hypothesis | Tested by |
|---|---|---|
| **H1** | Long-term memory alone concentrates exposure on popular items, and adding short-term memory and exploration reduces that. | Stage 4 long-term-only rows against Model C and Model D, under both embedding choices |
| **H2** | Attention over recent items explains recommendations faithfully only when short-term memory has real influence on the score. | RQ3 metrics, against the uniform-attention control |
| **H3** | RPE-gated plasticity adapts faster to sustained shifts than a fixed weight, while reacting less to isolated noise. | Shift and noise tests with and without RPE, plus stability |

H1 depends on an embedding choice: with L2-normalised embeddings, long-term memory alone covers far more of the catalogue than any baseline, the reverse of what H1 predicts. H3 currently has an unresolved measurement problem. Both are covered in `docs/changes-from-plan.md`.

## Limitations to state in the report

- **Cognitive labels are analogies.** The mechanisms are simplified computational abstractions and don't reproduce biological memory or attention. Reward prediction isn't a model of human motivation.
- **Observed relevance isn't true preference.** MovieLens ratings reflect what users were shown and chose to rate, so `merit_gap` and band recall inherit that exposure bias.
- **No demographic fairness claims.** `u.user` contains demographic fields, but this project doesn't load or evaluate them.
- **Deletion isn't causal proof.** It shows how the model's score responds, not why a person would like an item.
- **Simulated shifts and noise aren't real users.** They're controlled experiments, and offline metrics don't capture long-term satisfaction.
- **Replay feeds observed interactions, not responses to our recommendations.** The dataset can't show what users would have done with a different list.
