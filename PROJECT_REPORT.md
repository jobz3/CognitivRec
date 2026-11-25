# CognitivRec

**Group E · Human Centred Applications and Technologies · Otto von Guericke University Magdeburg**

Mudit Khandelwal, Jobin Roy and Sai Abhinav Kanagala

## Research question

Can a recommender with separate long-term and recent memory, candidate-specific attention, and feedback-driven adaptation maintain competitive top-10 accuracy while improving selected fairness, explanation and reliability outcomes? We evaluate these outcomes separately against conventional baselines and use matched ablations to test which mechanisms account for changes. The prototype uses computational abstractions of complementary memory and reward prediction error. [Kumaran et al. (2016)](https://web.stanford.edu/class/psych205/papers/KumaranHassabisMcC16CLSUpdate.pdf) · [Schultz et al. (1997)](https://doi.org/10.1126/science.275.5306.1593)

## Data and experimental protocol

- **Dataset.** MovieLens 100K contains 100,000 timestamped 1–5 star ratings from 943 users for 1,682 films. Each user rated at least 20 films across the full dataset. Movie genres support the diversity metric and the controlled shift tests. We define logged relevance as a rating of at least four stars, which also determines admission to recent memory. All historical ratings inform model fitting. [GroupLens dataset README](https://files.grouplens.org/datasets/movielens/ml-100k/README) · [Harper and Konstan (2015)](https://doi.org/10.1145/2827872)
- **Global holdout.** Split near the 70th and 85th percentiles of all rating timestamps, moving each boundary to the start of its timestamp group. Validation fits on train and final testing fits on train plus validation. This prevents another user's later ratings from entering shared popularity, neighbours or embeddings. Each cutoff uses a fixed catalogue of films rated before that cutoff and evaluates users with prior history and at least one eligible positive. This proxy for availability is consistently observable at prediction time. It limits the study to warm users and previously rated films. The original per-user split is retained as a sensitivity analysis with its own tuning and results. [Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation*](https://arxiv.org/abs/2010.11060)
- **Common ranking.** All models rank the full eligible catalogue minus the user's observed items, with ties broken by the lower item ID. The one-shot list is compared with the user's eligible positives in the entire held-out window. Metrics are macro-averaged over users unless the metric explicitly pools counts. Stage 3 averages seeds 42–46 and Stage 4 averages seeds 42–44. Exposure order and stress scenarios share seeds across models. The run records cohort sizes and actual fitted model parameters.
- **Selection and uncertainty.** Tune ItemKNN and SVD by validation NDCG@10. Model C searches 100 distinct sampled configurations plus its default and selects by validation replay NDCG@10, so the objective can see post-feedback effects. Model D starts from the selected C and maximises validation catalogue coverage under a 10% relative one-shot NDCG loss budget, preferring a smaller bonus on a tie. The budget is an explicit project choice. Test estimates use frozen settings. Paired 95% percentile intervals resample per-user differences after averaging each user's seed results (2,000 draws). These intervals are descriptive, conditional on the fitted dataset and experiment orders, and unadjusted for multiple comparisons.

## Models and rationale

- **Popularity** ranks by fitting-period rating counts and establishes the value of non-personalised exposure. **ItemKNN** uses shrunk cosine similarity with tuned neighbour count, shrinkage, implicit or rating-valued input and optional user-frequency weighting. **SVD** factorises the mean-centred rating matrix and folds new ratings into a user's vector while holding item factors fixed. These provide simple and tuned alternatives, including feedback-capable comparators. ItemKNN's additive history contributions also provide a strong fidelity control. [Ferrari Dacrema et al. (2019)](https://arxiv.org/abs/1907.06902)
- **Model C** uses truncated-SVD movie vectors. Let `q_i` be the unit vector for film `i` and `v_i` its raw or unit vector for scoring. Long-term memory starts as a unit-normalised average of `q_i` weighted by `r = (stars - 2.5) / 2.5`. Recent memory keeps the latest `M` positive interactions. For a candidate, softmax combines cosine similarity divided by temperature and a linear penalty for the lag of each memory. Lag is measured in buffer positions because timestamps frequently tie. The score blends long-term similarity with the weighted recent similarities using the short-term weight `alpha`. With no recent entries, the score uses long-term similarity. The raw and unit embedding controls test whether embedding norm changes the ranking conclusions.
- **Feedback.** Surprise is the absolute error between the incoming rating reward and the blended prediction clipped to `[-1, 1]`. A smoothed update moves `alpha` towards `clip(base_alpha + rpe_lr * (surprise - 0.35), 0.1, 0.9)`. The model appends qualifying feedback to recent memory and moves the long-term vector towards the reward-weighted item vector. The no-RPE ablation fixes `alpha` while keeping these memory updates active. This tests the added value of the gate beyond incorporating fresh ratings.
- **Model D** adds `beta * sqrt(log(T + 1) / (N_i + 1))`, with initial item counts from historical ratings and subsequent counts from simulated top-10 impressions. It is a UCB-inspired exposure bonus. Logged feedback follows the dataset or the specified intervention, independently of whether the model ranked the rated item. This design evaluates offline ranking and exposure outcomes. [Li et al. (2010)](https://arxiv.org/abs/1003.0146)

## How the four research questions are tested

1. **RQ1 · Recommendation quality.** Binary NDCG@10 is the main one-shot quality metric. Precision, recall, novelty in bits and mean pairwise genre distance show what changes around it. Compare all five models, then test short-term memory under matched raw and unit embeddings. The directional hypothesis is that recent user evidence improves replay over long-term memory alone. Baseline competitiveness is judged against tuned ItemKNN and SVD.
2. **RQ2 · Fairness.** Item-side fairness is operationalised as alignment between exposure and *logged relevant user–item pairs* across head, middle and tail popularity bands. Bands contain roughly one third of fitting interactions each. The merit gap is half the sum of absolute differences between the three exposure shares and the three relevant-pair shares. A second gap weights exposure at rank `j` by `1 / log2(j + 1)`. Band recall measures retrieval opportunities within observed relevance. Catalogue coverage and Gini describe distribution, and NDCG by history-size group measures user-side service differences. The hypothesis is that adding D's exposure bonus improves distribution and may improve relevance alignment. These effects can diverge. [Abdollahpouri et al. (2019)](https://arxiv.org/abs/1907.13286) · [Singh and Joachims (2018)](https://arxiv.org/abs/1802.07281)
3. **RQ3 · Explanation fidelity.** For each eligible top recommendation in seed 42, delete every evidence entry in turn and recompute scores with the same candidates and exposure. For C/D this means removal from recent memory, with long-term memory fixed. For ItemKNN it means removal from the fitted user profile with neighbours fixed. The gap compares the deletion effect of the model's highest-weight entry with the average effect of other deletions, divided by the original candidate score standard deviation. Kendall's tau-b, top-1 changes and adverse score rises test complementary aspects. At least three entries and nonconstant initial scores are required. Uniform attention is the control for candidate-dependent retrieval. Exported examples report lowest, median and highest gaps with film titles and, for C/D, the actual sum of long-term, recent and exploration components. [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/)
4. **RQ4 · Reliability.** Replay uses the nearest timestamp boundary to the midpoint of each target stream, requiring feedback before later positives. Frozen replay holds that user's candidates and exposure state fixed to measure the effect of learning from feedback. A dominant genre difference across logged samples provides a descriptive subgroup. Controlled shift injects 1, 2, 3, 5 or 10 unseen five-star films from a sparsely represented genre, holds exposure constant within each user's trial and compares genre share with a frozen ranker on the same candidates. Retention is NDCG on the original relevant targets relative to step zero. Noise injects two or five random five-star interactions against a mask-matched clean pass. Stability is the consecutive top-10 Jaccard overlap after removing the consumed item from the earlier list. Sparse-history tests refit after truncating 20% of eligible users to 5, 10 or 20 earliest interactions while keeping each user's full seen-item mask. The primary reliability hypothesis is that recent memory improves responsiveness, with a potential cost to retention and noise robustness. The added value of the RPE gate requires the matched no-RPE comparison.

## Results

### Evaluation cohort and validation choices

The global windows contained 69,997 training ratings through 9 February 1998, 15,003 validation ratings through 27 March 1998, and 15,000 test ratings through 22 April 1998 (UTC). Of the 226 users active in the test window, 91 also had a rating before the test cutoff. The other 135 first appeared in the test window. Among those 91 users, 77 gave at least one test rating of four or five stars. For 76 of them, at least one such rating was for a film already present in the fitted catalogue. These 76 users form the global ranking cohort.

The evaluator retained 2,558 validation rows with 104 ranking users and 2,432 test rows with 1,266 positive targets. Of the raw test rows, 12,486 belonged to users without fitted history and 140 referred to unavailable items. These exclusion counts can overlap. The test catalogue contained 1,628 eligible films. Replay had 57 eligible test users, including 12 flagged for a difference in dominant genre.

Across the full dataset, the median user's rating span was 0.82 hours, 59.6% of users rated in one session under the 30-minute gap rule, and 75.8% of ratings shared a timestamp with another rating from the same user. Under the per-user split, 39.92% of training rows on average were later than each test row. The global protocol yielded 0% on that diagnostic.

The global run selected ItemKNN with 300 neighbours, rating-valued input, user-frequency weighting and zero shrinkage, and SVD with 8 factors. Model C selected 64 factors, 24 recent positives, initial short-term weight 0.35, decay 0.15, temperature 0.10, RPE learning rate 0.60, consolidation rate 0.10 and raw scoring embeddings. The recent component's candidate-score spread was 0.93 times the long-term component's spread on validation, before mixing. The newest entry received the highest attention in 20.2% of candidate comparisons on average. The selected bonus of 0.10 increased validation coverage from 14.24% to 16.30%, with NDCG falling from 0.1519 to 0.1490, a 1.9% relative loss. At 0.18 NDCG fell to 0.1317, outside the 10% budget.

### RQ1 · Quality on the global test window

NDCG@10 was 0.1345 for Popularity, 0.1638 for ItemKNN, 0.1580 for SVD, 0.1825 for C and 0.1816 for D. Recall@10 was 0.0764, 0.0934, 0.0852, 0.1236 and 0.1229, respectively. ItemKNN achieved the highest precision@10 (0.1368). C and D achieved 0.1276 and 0.1263. D had the highest novelty (2.58 bits), with genre diversity of 0.8002 versus ItemKNN's 0.8062.

C's paired NDCG difference against ItemKNN was +0.0187 (95% interval −0.0253 to +0.0658). D's difference against ItemKNN was +0.0178 (−0.0269 to +0.0647). D's difference against C was −0.0009 (−0.0030 to +0.0004). The results support competitiveness on this cohort, with uncertainty about the direction of the population difference.

### RQ2 · Exposure and opportunity

C to D changed coverage from 13.76% to 15.20%, exposure Gini from 0.9283 to 0.9209, and rank-weighted merit gap from 0.3794 to 0.3478. ItemKNN's values were 12.04%, 0.9344 and 0.5218. Middle-band recall rose from 3.40% to 4.08% with exploration. Tail recall stayed at 0.25%, or one of 400 relevant tail pairs on average. The evidence supports improved exposure alignment across popularity bands under this proxy, with limited retrieval benefit for the tail itself.

Both activity groups contained 38 users. D achieved NDCG 0.1885 for lower-history users and 0.1747 for higher-history users (gap 0.0139). ItemKNN achieved 0.2157 and 0.1119 in the same groups (gap 0.1038). SVD's gap was narrower at 0.0077, with group means of 0.1618 and 0.1541. The reported group means show how a change in the gap can accompany different quality outcomes for each group.

### RQ3 · When attention identifies useful evidence

Across 75 eligible users per method, the average standardised deletion gap was 1.494 for C (95% interval 1.117 to 1.898), 1.278 for D (0.944 to 1.636) and 0.150 for ItemKNN (0.117 to 0.190). Removing the highest-weight memory changed the top item in 45.3% of C and D cases, and changed ItemKNN's top item in 4.0%. The tests use model-specific score scales and intervention points. Attention's mean Kendall tau-b was about −0.236 for C and D, while ItemKNN's additive attributions achieved 1.00. Attention often identified influential top evidence, with poor ordering of the remaining evidence. The highest-attention deletion increased the original recommendation's score in 5.33% of C and D cases.

For example, D ranked *Casablanca* using long-term, recent and exploration contributions of 0.2597, 0.2923 and 0.0237. The most attended entry was *The Manchurian Candidate*. Its deletion raised the score from 0.5756 to 0.5974, so the example illustrates why an attentive memory cannot automatically be described as supporting the recommendation.

### RQ4 · Responsiveness, retention and sparse history

Over the 57 replay users, C and D achieved NDCG@10 of 0.1950 and 0.1954 versus ItemKNN's 0.1884. Relative to their own frozen controls, the gains were +0.0712, +0.0770 and +0.0393, respectively. D's paired replay advantage over ItemKNN was +0.0070 (95% interval −0.0512 to +0.0724). The reported genre-difference subgroup has only 12 users.

After five synthetic five-star ratings, C and D gained 8.53 and 9.21 percentage points of new-genre exposure relative to their frozen controls, versus 1.97 points for ItemKNN. The same experiment retained 90.34%, 88.59% and 98.31% of original NDCG, respectively. Under five random ratings, C and D retained 92.10% and 90.26%, compared with ItemKNN's 96.91%. D's mean top-10 overlap was 0.827 versus 0.952 for ItemKNN. In the matched sparse-history group, D moved from 0.1803 full-history NDCG to 0.1772 with five ratings. ItemKNN moved from 0.1539 to 0.1945. These are controlled, small-cohort stress tests. The models that responded most strongly to the simulated shift also lost more quality under noise and after the shift.

### Component ablations and trade-offs

- **Recent memory.** Removing the recent buffer from D lowered NDCG by 0.0286 (95% interval 0.0061 to 0.0536 for full D minus the ablation). The corresponding replay difference was +0.0262 (−0.0061 to +0.0629). Under raw embeddings and zero exploration, adding the recent memory configuration to long-term memory improved NDCG by 0.0249 (+0.0027 to +0.0495). The unit-embedding comparison was smaller, +0.0131 (−0.0101 to +0.0359). Those controls show that both memory and the choice of scoring embeddings matter.
- **Attention.** Uniform attention lowered one-shot NDCG by 0.0296 versus full D and replay NDCG by 0.0460. Both paired intervals excluded zero. Its standardised fidelity gap was 0.012 (−0.012 to +0.041), compared with 1.278 for full D. This supports the value of candidate-specific weighting while the negative tau-b in RQ3 limits the explanation claim.
- **RPE gate.** With the initial state held fixed, the one-shot rankings were identical. The active gate improved replay NDCG by 0.0077 (−0.0023 to +0.0195). The controlled five-interaction shift gain rose from 5.31 to 8.99 percentage points versus the frozen controls. Five-noise retention fell from 92.83% to 89.81%. The gate shows a responsiveness trade-off. Absolute surprise alone gave no evidence of filtering isolated noise.
- **Exploration and embedding scale.** The validation sweep shows why coverage alone is an incomplete objective. At beta 0.30, coverage reached 26.55% and NDCG dropped to 0.0895. At beta 0.50, the rank-weighted merit gap climbed to 0.5245. With zero exploration and only long-term memory, unit scoring embeddings reduced the test merit gap to 0.1609 and reduced NDCG to 0.0651. The approximate exposure alignment can increase while recommendation quality falls sharply.
- **Plasticity.** Raising the validation base short-term weight from 0.10 to 0.85 moved the new genre's share after five injections from 7.82% to 12.66%. Consecutive top-10 overlap moved from 0.835 to 0.808. These endpoints illustrate the response and stability trade-off over settings. The same validation curve includes the chosen weight of 0.35.

### Sensitivity to the original per-user split

The per-user split was tuned independently and included 899 ranking users and 760 replay users. One-shot NDCG@10 was 0.1573 for ItemKNN, 0.1464 for C and 0.1421 for D. C's paired difference from ItemKNN was −0.0109 (95% interval −0.0252 to +0.0018). D's was −0.0152 (−0.0283 to −0.0019). The ranking advantage seen in the global cohort therefore depends on the protocol and the cohort.

Exploration changed coverage from 28.00% for C to 44.67% for D, rank-weighted merit gap from 0.3689 to 0.3272 and tail recall from 1.53% to 1.74%. Final replay NDCG was 0.1155 for ItemKNN, 0.1136 for C and 0.1066 for D. Relative gains over each model's frozen replay were +0.0048, +0.0186 and +0.0163. The sensitivity run repeats ranking, exposure and logged replay. Its different cohort, catalogue and temporal ordering make it a check on robustness of the conclusions across protocols. Use the strictly ordered global results for claims about future recommendations.

## Rolling temporal follow-up

### Why these comparators

**EASE tests competitive ranking accuracy.** ItemKNN estimates local similarities and SVD restricts scores to a low-dimensional reconstruction. EASE fits a global regularised linear mapping from positive interaction histories to item scores. For binary positive matrix `X`, it solves ridge reconstruction with a zero diagonal in the item weight matrix, which prevents copying the target item into its own prediction. We use the existing four-star relevance threshold and select ridge strength by validation NDCG@10. In replay, EASE folds new positives into the user row and keeps the item weights fixed. It adds a stronger collaborative test without new dependencies or sampled negatives. Its dense item-by-item solution grows quadratically in memory and cubically in the direct solve, a reasonable cost for 1,682 movies and a limitation at larger scale. [Steck (2019)](https://arxiv.org/abs/1905.03375) · [Ferrari Dacrema et al. (2019)](https://arxiv.org/abs/1907.06902)

**The matched recency and all-history profiles test the mechanism.** Both use the same SVD dimensionality, candidate-vector normalisation, signed rating transform and fitting data as C at each cutoff. The recency profile weights each interaction by `2^(-lag / half_life)` in units of user interactions, sums the vectors and scores with a single profile. The all-history control weights interactions equally. Both can update their profile from replayed feedback. We select the recency half-life by validation replay NDCG@10, the same selection endpoint as C. A consistent benefit over the all-history control supports recency. C's comparison against the recency profile tests the added value of C's design as a whole. The matched Stage 4 ablations remain the test of attention, memory and reward-gated adaptation individually. Interaction lags share the dataset's arbitrary ordering for same-second ratings. [Koren (2009)](https://doi.org/10.1145/1557019.1557072)

**A genre-content profile is the next useful baseline for a different question.** It could provide user-facing reasons tied to understandable movie features and score items without historical ratings. The current catalogue rule excludes those items and the simulated shift is defined by genre. A genre model could excel mechanically on that stress test, so a fair extension needs a separate cold-item split with independently known availability and a content-independent shift measure. BPR-MF is a reasonable next accuracy comparator if the rubric requires a model trained on pairwise ranking. It introduces negative-sampling and optimisation choices. SASRec is better suited to a larger dataset and a prospective next-item endpoint because the current timestamps often tie and this benchmark uses all relevant items in each window. [Pazzani and Billsus (2007)](https://doi.org/10.1007/978-3-540-72079-9_10) · [Rendle et al. (2009)](https://arxiv.org/abs/1205.2618) · [Kang and McAuley (2018)](https://arxiv.org/abs/1808.09781)

**The cross-scorer exposure control tests how much depends on the scoring model.** C versus D identifies the effect of adding the exposure bonus to C. The factorial follow-up below also applies the same rule to EASE, with a separate validation search and the same relative NDCG loss budget.

**A hybrid is a candidate for a future prediction study.** Start with EASE as the collaborative prior and add the existing episodic score as a residual. Calibrate their scales on validation and evaluate EASE, EASE plus the residual, and EASE plus the residual and exposure bonus. This would test whether recent memory adds predictive signal to the strongest collaborative baseline. Select the mixing weight on earlier data and reserve a new held-out window or dataset for that hypothesis.

### Time cutoffs and run records

Use cumulative training histories and consecutive 15%-of-ratings validation and test blocks. The early test covers the 55th to 70th percentiles, the middle test covers the 70th to 85th percentiles, and the late test covers the 85th percentile onward. Each model is retuned on the immediately preceding validation block. Test windows contain different events and can contain the same users. Report the paired intervals within each window and interpret agreement or disagreement across windows descriptively. The late cohort repeats the original test window with a smaller fixed search budget and an expanded model set, so this remains a retrospective sensitivity analysis. [Ji et al.](https://arxiv.org/abs/2010.11060) · [Meng et al. (2020)](https://arxiv.org/abs/2007.13237)

Record both kinds of timestamps. A UTC run ID and start and finish times identify the execution. The observed first and last UTC rating timestamps for each split identify the data that each model could use. The full run also records cohort counts, fitting parameters, all validation trials, five seeds, per-user metrics, paired comparisons, data and source hashes, dependency versions, and a late-window stress comparison and explanation demo. This allows a notebook result to be traced to a particular experimental run.

### What changed across windows

- **Early, 4 January to 9 February 1998 UTC.** 105 of 229 test users had prior ratings, and 99 entered the ranking comparison. NDCG@10 was 0.1099 for ItemKNN, 0.1290 for EASE, 0.1131 for C and 0.1114 for D. C to D increased coverage from 18.18% to 19.57%, reduced rank-weighted merit gap from 0.4226 to 0.3976 and increased tail recall from 0.18% to 0.35%.
- **Middle, 9 February to 27 March 1998 UTC.** 113 of 247 test users had prior ratings, and 104 entered the ranking comparison. NDCG@10 was 0.1283 for ItemKNN, 0.1491 for EASE, 0.1140 for C and 0.1045 for D. SVD led this window at 0.1524. C to D increased coverage from 21.04% to 26.55%, reduced rank-weighted merit gap from 0.2673 to 0.1967 and increased tail recall from 1.33% to 1.46%.
- **Late, 27 March to 22 April 1998 UTC.** 91 of 226 test users had prior ratings, and 76 entered the ranking comparison. NDCG@10 was 0.1638 for ItemKNN, 0.1998 for EASE, 0.1830 for C and 0.1812 for D. C to D increased coverage from 13.57% to 15.09%, reduced rank-weighted merit gap from 0.4074 to 0.3803 and reduced tail recall from 0.50% to 0.25%.

**EASE is the primary accuracy comparator for this project.** It exceeded C's one-shot and replay NDCG point estimates in every window. Their paired 95% intervals for one-shot NDCG included zero in all windows. EASE's replay NDCG was 0.1070, 0.1172 and 0.1964 in chronological order, the highest of the evaluated models in each window. C and D consistently improved catalogue coverage and merit alignment relative to EASE. D improved those metrics relative to C in every window, while the direction of its effect on tail recall varied.

On the same late cohort, the original 100-trial cognitive search gave C a replay NDCG of 0.1950 and D 0.1954, close to EASE's 0.1964 in the follow-up. The broader prior search matters to the size of the replay comparison. Early and middle windows only used the common follow-up budget of up to 24 sampled configurations plus the default.

**Simple recency weighting captures part of the observed adaptation.** The validation-selected half-life was 50 interactions early and three in the middle and late windows. The recency profile had lower one-shot NDCG than the matched all-history profile in every window and higher replay NDCG in all three, with paired replay intervals spanning zero. C led the recency profile in one-shot NDCG in all three windows, with the late paired difference of +0.0560 (95% interval +0.0137 to +0.0946). Their replay ordering changed by window. This supports the original motivation for comparing stable and recent preferences, while the strength of candidate-specific attention and feedback depends on the endpoint.

**The late-window stress tests distinguish responsiveness from robustness.** After five injected genre ratings, the three-interaction recency profile gained 11.97 percentage points of genre exposure versus its frozen counterpart and retained 41.99% of initial relevance NDCG. EASE gained 2.50 points and retained 96.84%. The refitted C and D gained 2.45 and 2.61 points and retained 93.86% and 91.90%. Under five random high ratings, the recency profile retained 40.82%, EASE 100.67%, C 90.93% and D 89.15%. Those retention ratios describe this stress test, with values over 100% possible. The refitted cognitive models responded substantially less than the original cognitive configurations on the same cohort, so the adaptation claim depends on the selected memory settings.

### Explanation demo

1. Model D originally ranked *Return of the Jedi* first. Its highest-attended memory was *Star Wars*. Removing that memory lowered the original film's score from 1.0942 to 0.9826 and made *L.A. Confidential* the top result.
2. Model D originally ranked *The Shawshank Redemption* first. It attended most to *Circle of Friends*. Removing that memory raised the original score from 1.3091 to 1.3256, and *The Shawshank Redemption* remained first.

These are intentionally contrasting cases for demonstration. The aggregate deletion statistics in RQ3 quantify how often the explanation pattern occurs.

## Exploration across two scorers

The factorial follow-up compares EASE, EASE with exploration, Cognitive, and Cognitive with exploration. It takes the previously selected EASE and Cognitive parameters from the completed rolling run. For each window, both policies start from the same fitted item counts and see the same seeded user orders, candidate masks, and ranking protocol. Only the exploration weight changes within each scorer. The positive-weight cognitive cell uses Model D's formula with its weight selected under this study's common grid.

For each scorer and window, trial weights are multiples of the median candidate-score standard deviation across validation users. This keeps the search meaningful across different raw score scales. The selected weight maximises validation coverage subject to at most 10% relative loss in the scorer's own validation NDCG@10. Ties favour a lower weight. The validation constraint sets a common maximum loss, and the observed test losses can differ. The test comparison is retrospective because these rating windows were used in the previous experiments.

Accuracy effects and differences between policy effects use paired user bootstrap intervals after averaging the same five seeds per user. Coverage, rank-weighted merit gap, and tail recall are aggregate changes averaged over the seed orders. Their reported values describe the observed cohorts. Popularity bands and candidate catalogues are held fixed within each window.

**Exposure gains appeared with both scorers.** In the early window, adding exploration increased coverage by 0.20 percentage points for EASE and 1.29 points for Cognitive. The respective merit gaps fell by 0.0055 and 0.0229. In the middle window, coverage increased by 1.89 and 3.67 points, while merit gaps fell by 0.0310 and 0.0536. Middle-window NDCG changed by −0.0026 for EASE and −0.0071 for Cognitive. The cognitive scorer delivered larger exposure gains in those windows at a larger observed accuracy cost.

**The late window showed the reverse trade-off.** EASE with exploration raised coverage from 13.27% to 20.98% and lowered the merit gap from 0.4374 to 0.3021. Its test NDCG fell from 0.1998 to 0.1638, an 18.0% relative loss despite an 8.6% validation loss. Cognitive with exploration raised coverage from 13.57% to 15.16%, lowered the merit gap from 0.4074 to 0.3790 and changed NDCG from 0.1830 to 0.1810, a 1.1% relative loss. The difference between the two NDCG effects was +0.0341 for Cognitive minus EASE (95% paired interval +0.0097 to +0.0612). This reflects the selected policies on that window, where the realised accuracy costs were different.

**Tail retrieval gains were small and inconsistent.** Early tail recall was unchanged for both scorers. In the middle window it moved from 0.22% to 0.00% for EASE and from 1.33% to 1.42% for Cognitive. In the late window EASE stayed at 0.50% and Cognitive fell from 0.50% to 0.25%. The experiment supports the exposure bonus as a way to broaden coverage across scorers, with scorer- and window-dependent accuracy costs. It leaves open whether any policy improves the retrieval of relevant tail films at a comparable test cost.

## Figures for the paper and presentation

The [figure bundle](results/exploration_study/20261125-162240-626125Z/figures/) contains eight figures. Use SVG for vector placement and PNG at 300 dpi for raster placement. SVG text is outlined for portability. File stems share their numbering across formats, and the source manifest identifies the exact input tables.

1. **Rolling ranking.** One-shot NDCG@10 for both scorers with each policy across the three test windows. Sample sizes count eligible ranking users. The right panel gives 95% paired user bootstrap intervals for Cognitive minus EASE without exploration. [SVG](results/exploration_study/20261125-162240-626125Z/figures/01_rolling_ranking.svg)
2. **Exploration effects.** Within-scorer changes from adding the bonus. The NDCG panel shows paired 95% intervals. Lower rank-weighted merit gap indicates closer alignment to logged relevance. Tail recall is measured among relevant tail items. [SVG](results/exploration_study/20261125-162240-626125Z/figures/02_exploration_effects.svg)
3. **Validation budget.** Trial coverage versus relative NDCG loss within each scorer's validation window. Diamonds identify selected weights. The display extends through 15% loss, and the full grid is in the validation CSV. [SVG](results/exploration_study/20261125-162240-626125Z/figures/03_validation_budget.svg)
4. **Component ablations.** Paired NDCG changes from the original full Model D when disabling or simplifying individual mechanisms, with 95% intervals. The left panel uses the 76-user late ranking cohort and the right panel the 57-user replay cohort. [SVG](results/exploration_study/20261125-162240-626125Z/figures/04_ablations.svg)
5. **Explanation fidelity.** The standardised effect of deleting the highest-weight entry relative to other entries, with 95% intervals. Kendall tau-b reports agreement between attention or attribution weights and the full ordering of observed deletion effects. Each plotted model had 75 eligible users. [SVG](results/exploration_study/20261125-162240-626125Z/figures/05_explanation_fidelity.svg)
6. **Preference shift.** The original five models' new-genre share and retained NDCG as synthetic positive feedback accumulates. Retained NDCG is relative to each model's step-zero result. [SVG](results/exploration_study/20261125-162240-626125Z/figures/06_preference_shift.svg)
7. **Reliability stress.** Late-window comparison of genre exposure gain, retention of prior relevance and robustness to random feedback in the rolling study. The cognitive exploration weight comes from that study. Each point uses five injected ratings for the relevant scenario. [SVG](results/exploration_study/20261125-162240-626125Z/figures/07_reliability_stress.svg)
8. **Plasticity trade-off.** Validation sweep of the short-term memory weight. The diamond is the chosen value. Stability is consecutive-list Jaccard overlap. [SVG](results/exploration_study/20261125-162240-626125Z/figures/08_plasticity_tradeoff.svg)

Figures 1, 2, 5 and 7 tell the main story in a slide deck. Use Figures 3 and 8 to show how hyperparameters were chosen, Figure 4 to explain the mechanism checks, and Figure 6 to show how responsiveness develops over time.

## Responsibility and delivery

- **Mudit Khandelwal** · Global split and cohort diagnostics, Model C and its feedback controls, RQ4 scenarios and ablations. Evidence lives in `src/cognitiverec/data/`, `models/cognitive.py`, `evaluation/reliability.py`, and Stage 4.
- **Jobin Roy** · Popularity, ItemKNN, SVD, baseline validation search and RQ2 fairness. Evidence lives in `models/`, `experiments/stage1_baselines.py`, and `evaluation/fairness.py`.
- **Sai Abhinav Kanagala** · Model D exposure policy, RQ3 deletion fidelity and case examples, and the shift and trade-off figures. Evidence lives in `models/exploration.py`, `evaluation/explainability.py`, and `plots.py`.
- **Shared** · Candidate protocol, RQ1 metrics, benchmark and report. Roles reflect the project plan, with local completion checked across all three workstreams.
- **Follow-up** · Mudit's workstream covers the rolling cutoffs and matched memory controls. Jobin's covers EASE and cross-window exposure and tail recall. Sai Abhinav's covers the two deletion cases and the notebook walkthrough. The notebook presents their work in the order needed to reproduce it.
- **Final handoff** · The recommender prototype, baseline comparison, four research questions, ablations, retrospective sensitivity, exploration control, demo and publication figures are assembled in the repository and documented here.

## What the professor can still challenge

- **Generalisation.** The rolling check tests sensitivity to cutoff dates within MovieLens 100K. Each global cohort still consists of returning users with later eligible positives. An independent dataset or a prospective study would test broader generalisation. The per-user comparison changes both the cohort and the temporal assumptions.
- **Human-centred evidence.** Offline relevance and simulated impressions are proxies. This dataset cannot establish online satisfaction, true user attention, exposure causality, or demographic fairness. The explanation tests establish item-level fidelity to the tested intervention. A user study would be needed to evaluate whether people understand or benefit from it.
- **Real adaptation.** Rating timestamps can reflect batched retrospective ratings. The controlled stress tests measure response to specified injections. The replay and genre analyses cannot establish actual preference transitions or what would happen under live recommendation feedback.
- **Scope and modelling.** The brief's examples include content and neural models, calibration and uncertainty, and demographic analysis where appropriate. The core comparison uses three conventional rankers and lightweight cognitive variants. There are no predicted probabilities to calibrate, and the exposure count is an exploration heuristic. The optional next experimental control is a non-cognitive content profile with matched feedback and ranking conditions.
- **Submission.** The team plan records a 15 January 2027 report and code deadline. Confirm the report format, page count, contribution declaration and demo expectations against the course's current instructions.
- **Questions for the professor.** Does the missing Section 10 specify a governing fairness or exposure criterion? Does the model-deletion demo satisfy the explainability requirement, or is a user-comprehension evaluation expected?

## Local reproduction

From `CognitivRec` in PowerShell with Python 3.11 via `uv`:

```powershell
uv sync --locked
uv run python -m pytest -o addopts=
uv run python -m cognitiverec.experiments
uv run python -m cognitiverec.experiments --split per_user
uv run python -m cognitiverec.experiments --followup

function Get-RunDirectory($Stage) {
    Get-ChildItem "results/$Stage" -Directory |
        Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
}

$rollingRun = Get-RunDirectory temporal_followup
uv run python -m cognitiverec.experiments --exploration-study $rollingRun

$explorationRun = Get-RunDirectory exploration_study
$trustRun = Get-RunDirectory stage3
$ablationRun = Get-RunDirectory stage4
uv run python -m cognitiverec.experiments.paper_figures `
  --exploration $explorationRun --rolling $rollingRun `
  --trust $trustRun --ablation $ablationRun
```

Use `--quick` for a smoke run. The alternate split and smoke-run paths have their own tuned files. For an offline data setup, put MovieLens `u.data` and `u.item` under `data/ml-100k/` before running. The notebook uses the same package and defaults to a smoke run. Set its `QUICK` switch to `False` for report-quality results. The checked-out base was `987493d` from the local `origin/dev` ref. This report describes the working tree changes applied on top of that revision.

The official endpoint returned HTTP 502 from the local environment. The benchmark used local files verified byte-for-byte against a preserved GroupLens ZIP. Its MD5 is `0e33842e24a9c977be4e0107933c0723` and SHA-256 is `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`. [Archived GroupLens ZIP](https://web.archive.org/web/20260508013803id_/https://files.grouplens.org/datasets/movielens/ml-100k.zip)

### Verification and local evidence

Git ignores `results/`. The links in the results and figures sections identify the original local runs. On a fresh checkout, the reproduction commands write equivalent artifacts to new timestamped directories.

The four global stages, rolling follow-up and factorial exploration study ran on Windows with locked Python 3.11.15 dependencies. The current suite passed 157 tests, including eight tests against MovieLens 100K. Ruff lint and formatting checks passed. All eight code cells in the updated notebook executed in a fresh local Jupyter kernel in smoke mode. The hosted Colab runtime remains unverified.

- [Stage 1 temporal diagnostics](results/stage1/20261125-151002-345701Z/temporal_report.yaml) and [cohort counts](results/stage1/20261125-151002-345701Z/cohorts.yaml)
- [Stage 2 model search](results/stage2/20261125-151108-897007Z/search.csv) and [exploration selection](results/stage2/20261125-151108-897007Z/beta_sweep.csv)
- [Stage 3 metrics](results/stage3/20261125-151403-405809Z/metrics.csv), [paired intervals](results/stage3/20261125-151403-405809Z/comparisons.csv) and [fidelity examples](results/stage3/20261125-151403-405809Z/fidelity_examples.csv)
- [Stage 4 ablations](results/stage4/20261125-151958-692581Z/ablation.csv) and [paired intervals](results/stage4/20261125-151958-692581Z/comparisons.csv)
- [Per-user sensitivity metrics](results/per_user/sensitivity/20261125-152310-683274Z/metrics.csv), [paired intervals](results/per_user/sensitivity/20261125-152310-683274Z/comparisons.csv) and [selected model parameters](results/per_user/sensitivity/20261125-152310-683274Z/models.yaml)
- [Rolling study manifest](results/temporal_followup/20261125-160227-772960Z/manifest.yaml), [UTC window boundaries](results/temporal_followup/20261125-160227-772960Z/windows.csv), [model metrics](results/temporal_followup/20261125-160227-772960Z/metrics.csv), [paired intervals](results/temporal_followup/20261125-160227-772960Z/comparisons.csv), [late-window stress tests](results/temporal_followup/20261125-160227-772960Z/anchor_reliability.csv) and [demo cases](results/temporal_followup/20261125-160227-772960Z/demo_examples.csv)
- [Exploration study manifest](results/exploration_study/20261125-162240-626125Z/manifest.yaml), [validation grid](results/exploration_study/20261125-162240-626125Z/validation.csv), [test results](results/exploration_study/20261125-162240-626125Z/metrics.csv), [paired effects](results/exploration_study/20261125-162240-626125Z/effects.csv) and [figure source manifest](results/exploration_study/20261125-162240-626125Z/figures/sources.yaml)
- [Locally executed notebook smoke run](results/notebook_validation.ipynb)
