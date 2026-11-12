"""Run the project plan's Colab code as written, and measure what is wrong with its results.

Every number in docs/changes-from-plan.md under "as written" comes from this script, so anyone can
check them with `make audit`. The classes below are the plan's Cells 1 to 4 with only the variable
names shortened. Nothing in the project imports this file: it exists as evidence, and can be
deleted once the team has reviewed the changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from cognitiverec.config import load_config, resolve_path
from cognitiverec.data import load_movielens_100k

SEED = 42
TOP_K = 10


def heading(text: str) -> None:
    print(f"\n=== {text} ===")


# ---- Cell 1: data and split --------------------------------------------------------------------
dataset = load_movielens_100k(resolve_path(load_config(), "data"))
N_USERS, N_ITEMS = dataset.n_users, dataset.n_items
item_genres = dataset.item_genres
ratings_df = dataset.interactions.rename(columns={"user": "u", "item": "i"})
ratings_df = ratings_df.sort_values(["u", "timestamp"]).reset_index(drop=True)

train_list, val_list, test_list = [], [], []
for _, group in ratings_df.groupby("u", sort=False):
    n = len(group)
    n_train, n_val = max(1, int(0.70 * n)), max(1, int(0.85 * n))
    train_list.append(group.iloc[:n_train])
    val_list.append(group.iloc[n_train:n_val])
    test_list.append(group.iloc[n_val:])
train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)


# ---- Cell 2: baselines --------------------------------------------------------------------------
R_train = csr_matrix(
    (train_df["rating"].values.astype(np.float32), (train_df["u"].values, train_df["i"].values)),
    shape=(N_USERS, N_ITEMS),
)
R_train_bin = csr_matrix(
    (np.ones(len(train_df), dtype=np.float32), (train_df["u"].values, train_df["i"].values)),
    shape=(N_USERS, N_ITEMS),
)
item_pop_counts = np.asarray(R_train_bin.sum(axis=0)).flatten()
item_self_info = -np.log2((item_pop_counts + 1.0) / (N_USERS + 1.0))


class PopularityRecommender:
    def __init__(self, pop_counts):
        self.scores = pop_counts.astype(np.float32)

    def predict_for_user(self, u, seen_items):
        scores = self.scores.copy()
        scores[list(seen_items)] = -1e9
        return scores


class ItemKNNRecommender:
    def __init__(self, R, top_k=100, shrinkage=10.0):
        col_norms = np.sqrt(np.asarray(R.power(2).sum(axis=0)).flatten()) + 1e-8
        R_norm = R.multiply(1.0 / col_norms)
        sim = (R_norm.T @ R_norm).toarray()
        co_counts = ((R > 0).astype(np.float32).T @ (R > 0).astype(np.float32)).toarray()
        sim = sim * (co_counts / (co_counts + shrinkage))
        np.fill_diagonal(sim, 0.0)
        for j in range(sim.shape[1]):
            idx = np.argsort(sim[:, j])[:-top_k]
            sim[idx, j] = 0.0
        self.W = csr_matrix(sim)
        self.R_user = R.toarray()

    def predict_for_user(self, u, seen_items):
        scores = np.asarray(self.R_user[u] @ self.W).flatten()
        scores[list(seen_items)] = -1e9
        return scores


class MatrixFactorizationSVD:
    def __init__(self, R, n_factors=32):
        R_dense = R.toarray()
        mask = R_dense > 0
        self.user_means = R_dense.sum(axis=1) / np.maximum(1, mask.sum(axis=1))
        R_centered = R_dense - (self.user_means[:, None] * mask)
        U, s, Vt = svds(csr_matrix(R_centered), k=n_factors, random_state=SEED)
        order = np.argsort(s)[::-1]
        U, s, Vt = U[:, order], s[order], Vt[order, :]
        self.user_vecs = U * np.sqrt(s)[None, :]
        self.item_vecs = (Vt.T) * np.sqrt(s)[None, :]
        norms = np.linalg.norm(self.item_vecs, axis=1, keepdims=True) + 1e-8
        self.item_vecs_norm = self.item_vecs / norms

    def predict_for_user(self, u, seen_items):
        scores = self.user_vecs[u] @ self.item_vecs.T + self.user_means[u]
        scores[list(seen_items)] = -1e9
        return scores


np.random.seed(SEED)
model_A_pop = PopularityRecommender(item_pop_counts)
model_B_knn = ItemKNNRecommender(R_train, top_k=80, shrinkage=15.0)
model_B_svd = MatrixFactorizationSVD(R_train, n_factors=32)


# ---- Cell 3: Models C and D ----------------------------------------------------------------------
class CognitiveRecommender:
    def __init__(
        self,
        item_embeddings,
        train_df,
        pop_counts,
        memory_size=12,
        base_alpha=0.55,
        decay_lambda=0.15,
        rpe_lr=0.12,
        lt_consolidation_rate=0.05,
        exploration_beta=0.0,
        use_short_term=True,
        use_rpe_adaptation=True,
    ):
        self.V = item_embeddings.copy()
        self.d = self.V.shape[1]
        self.memory_size = memory_size
        self.base_alpha = base_alpha
        self.decay_lambda = decay_lambda
        self.rpe_lr = rpe_lr
        self.lt_rate = lt_consolidation_rate
        self.beta = exploration_beta
        self.use_short_term = use_short_term
        self.use_rpe_adaptation = use_rpe_adaptation
        self.u_LT = np.zeros((N_USERS, self.d), dtype=np.float32)
        self.episodic_buffer = {u: [] for u in range(N_USERS)}
        self.alpha_u = np.full(N_USERS, base_alpha, dtype=np.float32)
        self.exposure_counts = pop_counts.astype(np.float32).copy() + 1.0
        self.total_steps = float(np.sum(self.exposure_counts))
        self._init_from_training_history(train_df)

    def _init_from_training_history(self, df):
        for u, group in df.groupby("u", sort=False):
            items = group["i"].values
            ratings = group["rating"].values
            times = group["timestamp"].values
            weights = np.clip((ratings - 2.5) / 2.5, -1.0, 1.0)
            if len(items) > 0:
                vec = (weights[:, None] * self.V[items]).mean(axis=0)
                self.u_LT[u] = vec / (np.linalg.norm(vec) + 1e-8)
            pos_mask = ratings >= 3.0
            recent_items = items[pos_mask][-self.memory_size :]
            recent_times = times[pos_mask][-self.memory_size :]
            recent_w = np.clip((ratings[pos_mask][-self.memory_size :] - 2.5) / 2.5, 0.2, 1.0)
            self.episodic_buffer[u] = list(zip(recent_items, recent_times, recent_w, strict=True))

    def compute_scores_and_explanations(self, u, seen_items, ignore_buffer_idx=None):
        lt_scores = self.V @ self.u_LT[u]
        buf = self.episodic_buffer[u]
        if ignore_buffer_idx is not None and len(buf) > 0:
            buf = [entry for idx, entry in enumerate(buf) if idx != ignore_buffer_idx]
        if (not self.use_short_term) or len(buf) == 0:
            st_scores = np.zeros(N_ITEMS, dtype=np.float32)
            att_weights = np.array([])
        else:
            mem_items = np.array([x[0] for x in buf], dtype=int)
            mem_val = np.array([x[2] for x in buf], dtype=np.float32)
            rank_lag = np.arange(len(buf) - 1, -1, -1, dtype=np.float32)
            time_penalty = self.decay_lambda * rank_lag[:, None]
            sim_matrix = (self.V[mem_items] @ self.V.T) / np.sqrt(self.d)
            logits = sim_matrix - time_penalty
            logits = logits - np.max(logits, axis=0, keepdims=True)
            exp_logits = np.exp(logits)
            att_weights = exp_logits / (np.sum(exp_logits, axis=0, keepdims=True) + 1e-8)
            st_scores = np.sum((att_weights * mem_val[:, None]) * sim_matrix, axis=0)
        alpha = self.alpha_u[u] if self.use_short_term else 0.0
        reward_est = (1.0 - alpha) * lt_scores + alpha * st_scores
        if self.beta > 0.0:
            ucb_bonus = self.beta * np.sqrt(
                np.log(self.total_steps + 1.0) / (self.exposure_counts + 1.0)
            )
        else:
            ucb_bonus = 0.0
        total_scores = reward_est + ucb_bonus
        if len(seen_items) > 0:
            total_scores[list(seen_items)] = -1e9
        return total_scores, att_weights, buf

    def step_feedback(self, u, item_idx, rating, timestamp, recommended_top_k=None):
        if recommended_top_k is not None:
            self.exposure_counts[recommended_top_k] += 1.0
            self.total_steps += len(recommended_top_k)
        norm_r = float(np.clip((rating - 2.5) / 2.5, -1.0, 1.0))
        pred_r = float(np.clip(self.V[item_idx] @ self.u_LT[u], -1.0, 1.0))
        rpe_surprise = abs(norm_r - pred_r)
        if self.use_rpe_adaptation:
            target_alpha = np.clip(
                self.base_alpha + self.rpe_lr * (rpe_surprise - 0.35), 0.15, 0.88
            )
            self.alpha_u[u] = 0.7 * self.alpha_u[u] + 0.3 * target_alpha
        if rating >= 3.5:
            buf = self.episodic_buffer[u]
            buf.append((item_idx, timestamp, max(0.2, norm_r)))
            if len(buf) > self.memory_size:
                buf.pop(0)
        self.u_LT[u] = (1.0 - self.lt_rate) * self.u_LT[u] + self.lt_rate * (
            norm_r * self.V[item_idx]
        )
        self.u_LT[u] /= np.linalg.norm(self.u_LT[u]) + 1e-8


def build_cognitive(**params) -> CognitiveRecommender:
    return CognitiveRecommender(model_B_svd.item_vecs_norm, train_df, item_pop_counts, **params)


# ---- Cell 4: evaluation --------------------------------------------------------------------------
user_seen = {u: set(group["i"].values) for u, group in train_df.groupby("u")}
val_seen = {u: set(group["i"].values) for u, group in val_df.groupby("u")}


def gini_coefficient(counts):
    arr = np.sort(np.asarray(counts, dtype=np.float64))
    if np.sum(arr) == 0:
        return 0.0
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def ndcg(hits, n_relevant, k=10):
    dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, n_relevant)))
    return dcg / idcg if idcg > 0 else 0.0


def genuine_shift_users():
    shift_users = set()
    for u, test_grp in test_df.groupby("u"):
        pos_test = test_grp[test_grp["rating"] >= 4.0]["i"].values
        pos_train = train_df[(train_df["u"] == u) & (train_df["rating"] >= 4.0)]["i"].values
        if len(pos_test) >= 3 and len(pos_train) >= 5:
            top_train_g = np.argmax(item_genres[pos_train].sum(axis=0))
            top_test_g = np.argmax(item_genres[pos_test].sum(axis=0))
            if top_train_g != top_test_g:
                shift_users.add(u)
    return shift_users


def evaluate_model_comprehensive(model_name, model_factory, is_cognitive=False, k=10):
    model = model_factory()
    user_train_counts = {u: len(seen) for u, seen in user_seen.items()}
    median_activity = np.median(list(user_train_counts.values()))
    shift_users = genuine_shift_users()

    recalls, ndcgs, low_act, high_act, shift_ndcgs = [], [], [], [], []
    exposure_tracker = np.zeros(N_ITEMS, dtype=np.float64)
    top_del_drops, rand_del_drops = [], []

    for u, test_grp in test_df.groupby("u", sort=False):
        pos_targets = set(test_grp[test_grp["rating"] >= 4.0]["i"].values)
        if len(pos_targets) == 0:
            continue
        seen = user_seen.get(u, set()).copy()
        if is_cognitive:
            scores, att_w, buf = model.compute_scores_and_explanations(u, seen)
        else:
            scores = model.predict_for_user(u, seen)
        top_k = np.argsort(scores)[::-1][:k]
        exposure_tracker[top_k] += 1.0

        hits = [1.0 if i in pos_targets else 0.0 for i in top_k]
        user_ndcg = ndcg(hits, len(pos_targets), k)
        recalls.append(sum(hits) / len(pos_targets))
        ndcgs.append(user_ndcg)
        (low_act if user_train_counts.get(u, 0) <= median_activity else high_act).append(user_ndcg)

        if is_cognitive and len(buf) >= 3 and att_w.size > 0:
            top_item = top_k[0]
            most_influential = int(np.argmax(att_w[:, top_item]))
            candidates = [idx for idx in range(len(buf)) if idx != most_influential]
            rand_idx = int(np.random.choice(candidates))
            no_top, _, _ = model.compute_scores_and_explanations(
                u, seen, ignore_buffer_idx=most_influential
            )
            no_rand, _, _ = model.compute_scores_and_explanations(
                u, seen, ignore_buffer_idx=rand_idx
            )
            top_del_drops.append(float(scores[top_item] - no_top[top_item]))
            rand_del_drops.append(float(scores[top_item] - no_rand[top_item]))

        if is_cognitive:
            mid = len(test_grp) // 2
            for row_idx, row in enumerate(test_grp.itertuples()):
                model.step_feedback(u, int(row.i), float(row.rating), float(row.timestamp), top_k)
                seen.add(int(row.i))
                if row_idx == mid and u in shift_users:
                    later = test_grp.iloc[mid + 1 :]
                    later_targets = set(later[later["rating"] >= 4.0]["i"].values)
                    if len(later_targets) > 0:
                        s_post, _, _ = model.compute_scores_and_explanations(u, seen)
                        tk_post = np.argsort(s_post)[::-1][:k]
                        h_post = [1.0 if i in later_targets else 0.0 for i in tk_post]
                        shift_ndcgs.append(ndcg(h_post, len(later_targets), k))
        elif u in shift_users:
            shift_ndcgs.append(user_ndcg)

    noise_retention = 100.0
    if is_cognitive:
        noisy_model = model_factory()
        noisy_ndcgs = []
        for u, test_grp in test_df.groupby("u", sort=False):
            pos_targets = set(test_grp[test_grp["rating"] >= 4.0]["i"].values)
            if len(pos_targets) == 0:
                continue
            seen = user_seen.get(u, set()).copy()
            unseen_pool = list(set(range(N_ITEMS)) - seen)
            for _ in range(2):
                noise_i = int(np.random.choice(unseen_pool))
                noisy_model.step_feedback(u, noise_i, 5.0, 999999999.0)
            s_noisy, _, _ = noisy_model.compute_scores_and_explanations(u, seen)
            tk_n = np.argsort(s_noisy)[::-1][:k]
            h_n = [1.0 if i in pos_targets else 0.0 for i in tk_n]
            noisy_ndcgs.append(ndcg(h_n, len(pos_targets), k))
        noise_retention = 100.0 * (np.mean(noisy_ndcgs) / (np.mean(ndcgs) + 1e-8))

    fidelity = (np.mean(top_del_drops) / (np.mean(rand_del_drops) + 1e-8)) if top_del_drops else 0.0
    return {
        "Model": model_name,
        "Recall@10": round(float(np.mean(recalls)), 4),
        "NDCG@10": round(float(np.mean(ndcgs)), 4),
        "Coverage%": round(100.0 * float(np.sum(exposure_tracker > 0)) / N_ITEMS, 2),
        "Gini": round(gini_coefficient(exposure_tracker), 4),
        "Fidelity_Ratio": round(float(fidelity), 2),
        "Shift_NDCG@10": round(float(np.mean(shift_ndcgs)), 4),
        "Noise_Retain%": round(float(noise_retention), 1),
    }


# ---- Audit --------------------------------------------------------------------------------------
def plan_table() -> None:
    heading("Plan's Stage 3 table, as written (test split, default hyperparameters)")
    rows = [
        evaluate_model_comprehensive("A: Popularity", lambda: model_A_pop),
        evaluate_model_comprehensive("B1: ItemKNN", lambda: model_B_knn),
        evaluate_model_comprehensive("B2: SVD", lambda: model_B_svd),
        evaluate_model_comprehensive(
            "C: Cognitive", lambda: build_cognitive(exploration_beta=0.0), is_cognitive=True
        ),
        evaluate_model_comprehensive(
            "D: Cognitive + UCB", lambda: build_cognitive(exploration_beta=0.18), is_cognitive=True
        ),
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def timestamps() -> None:
    heading("What the timestamps can support")
    by_user = ratings_df.groupby("u")["timestamp"]
    sessions = (by_user.diff() > 1800).groupby(ratings_df["u"]).sum() + 1
    print(f"Users who rated everything in one sitting: {100 * (sessions == 1).mean():.1f}%")
    tied = ratings_df.duplicated(["u", "timestamp"], keep=False).mean()
    print(f"Ratings sharing a timestamp with another by the same user: {100 * tied:.1f}%")
    gap = test_df.groupby("u")["timestamp"].min() - train_df.groupby("u")["timestamp"].max()
    print(f"Median gap, last train rating to first test rating: {gap.median() / 60:.1f} minutes")

    flagged = sorted(genuine_shift_users())
    share = 100 * len(flagged) / N_USERS
    one_sitting_share = 100 * (sessions.loc[flagged] == 1).mean()
    print(f"Users the plan labels a 'genuine' shift: {len(flagged)} ({share:.1f}%)")
    print(f"Of those, users who rated everything in one sitting: {one_sitting_share:.1f}%")


def global_ordering() -> None:
    heading("What the per-user split does not protect")
    train_times = np.sort(train_df["timestamp"].values)
    later = 1 - np.searchsorted(train_times, test_df["timestamp"].values, side="right") / len(
        train_times
    )
    print(f"Mean share of train rows later than a given test row: {100 * later.mean():.1f}%")
    cut_70, cut_85 = np.quantile(ratings_df["timestamp"], [0.70, 0.85])
    train_users = set(ratings_df.loc[ratings_df["timestamp"] <= cut_70, "u"])
    test_users = set(ratings_df.loc[ratings_df["timestamp"] > cut_85, "u"])
    missing = len(test_users - train_users)
    print(
        f"A global time split instead: {len(test_users)} test users, "
        f"{missing} ({100 * missing / len(test_users):.1f}%) with no training history"
    )


def fidelity_stability() -> None:
    heading("Fidelity ratio: same users, same model, ten different random draws")
    model = build_cognitive()
    users = [u for u, group in test_df.groupby("u", sort=False) if (group["rating"] >= 4).any()]

    cognitive_rows = []
    for u in users:
        if len(model.episodic_buffer[u]) < 3:
            continue
        scores, attention, buf = model.compute_scores_and_explanations(u, user_seen[u])
        top_item = int(np.argmax(scores))
        drops = np.array(
            [
                scores[top_item]
                - model.compute_scores_and_explanations(u, user_seen[u], ignore_buffer_idx=entry)[
                    0
                ][top_item]
                for entry in range(len(buf))
            ],
            dtype=np.float64,
        )
        cognitive_rows.append((int(np.argmax(attention[:, top_item])), drops))

    knn_rows = []
    for u in users:
        history = np.where(model_B_knn.R_user[u] > 0)[0]
        scores = model_B_knn.predict_for_user(u, user_seen[u])
        top_item = int(np.argmax(scores))
        weights = np.asarray(model_B_knn.W[history, top_item].todense()).ravel()
        contributions = model_B_knn.R_user[u, history] * weights
        knn_rows.append((int(np.argmax(contributions)), contributions.astype(np.float64)))

    def ratios(rows):
        top_drop = np.mean([drops[best] for best, drops in rows])
        results = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            random_drop = np.mean(
                [drops[rng.choice(np.delete(np.arange(len(drops)), best))] for best, drops in rows]
            )
            results.append(top_drop / (random_drop + 1e-8))
        return [round(float(value), 2) for value in results]

    raises = np.mean([drops[best] < 0 for best, drops in cognitive_rows])
    print(f"Model C ratio by seed: {ratios(cognitive_rows)}")
    print(f"Model C users whose score rises when the top memory is deleted: {100 * raises:.1f}%")
    print(f"ItemKNN ratio by seed, which the plan reports as 0.00: {ratios(knn_rows)}")


def noise_artefact() -> None:
    heading("Model D noise retention, with no noise injected at all")
    model = build_cognitive(exploration_beta=0.18)
    scores = []
    for u, group in test_df.groupby("u", sort=False):
        targets = set(group[group["rating"] >= 4.0]["i"].values)
        if targets:
            ranked, _, _ = model.compute_scores_and_explanations(u, user_seen[u].copy())
            top_k = np.argsort(ranked)[::-1][:10]
            scores.append(ndcg([1.0 if i in targets else 0.0 for i in top_k], len(targets)))
    print(f"NDCG@10 with exposure frozen and zero noise: {np.mean(scores):.4f}")


def memory_connection() -> None:
    heading("Does short-term memory reach the ranking, as written?")
    model = build_cognitive()
    long_spread, short_spread, newest = [], [], []
    for u in range(N_USERS):
        buf = model.episodic_buffer[u]
        if not buf:
            continue
        mem_items = np.array([x[0] for x in buf], dtype=int)
        mem_val = np.array([x[2] for x in buf], dtype=np.float32)
        lag = np.arange(len(buf) - 1, -1, -1, dtype=np.float32)
        sim = (model.V[mem_items] @ model.V.T) / np.sqrt(model.d)
        logits = sim - model.decay_lambda * lag[:, None]
        logits = logits - logits.max(axis=0, keepdims=True)
        attention = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
        short_term = ((attention * mem_val[:, None]) * sim).sum(axis=0)
        long_term = model.V @ model.u_LT[u]
        keep = np.setdiff1d(np.arange(N_ITEMS), list(user_seen[u]))
        long_spread.append(long_term[keep].std())
        short_spread.append(short_term[keep].std())
        newest.append((attention[:, keep].argmax(axis=0) == len(buf) - 1).mean())
    spread_ratio = np.mean(short_spread) / np.mean(long_spread)
    print(f"Short-term score spread as a share of long-term: {spread_ratio:.3f}")
    print(f"Attention favours the newest memory: {100 * np.mean(newest):.1f}% of the time")

    tracked = build_cognitive()
    surprise: dict[float, list[float]] = {}
    for row in test_df.itertuples():
        reward = float(np.clip((row.rating - 2.5) / 2.5, -1, 1))
        predicted = float(np.clip(tracked.V[row.i] @ tracked.u_LT[row.u], -1, 1))
        surprise.setdefault(row.rating, []).append(abs(reward - predicted))
        tracked.step_feedback(row.u, int(row.i), float(row.rating), float(row.timestamp))
    print(
        "Mean |RPE| by star rating: "
        + ", ".join(f"{int(r)}: {np.mean(v):.2f}" for r, v in sorted(surprise.items()))
    )
    low, high = np.percentile(tracked.alpha_u, [5, 95])
    print(f"alpha_u after the whole test stream, 5th to 95th percentile: {low:.3f} to {high:.3f}")

    def top_lists(built):
        return {
            u: set(
                np.argsort(built.compute_scores_and_explanations(u, user_seen[u].copy())[0])[::-1][
                    :TOP_K
                ].tolist()
            )
            for u in range(N_USERS)
            if built.episodic_buffer[u]
        }

    def jaccard(first, second):
        return float(
            np.mean([len(first[u] & second[u]) / len(first[u] | second[u]) for u in first])
        )

    no_memory = build_cognitive(use_short_term=False)
    overlap = jaccard(top_lists(build_cognitive()), top_lists(no_memory))
    print(f"Top-10 overlap with short-term memory switched off: {overlap:.3f}")
    narrow_low, narrow_high = build_cognitive(), build_cognitive()
    narrow_low.alpha_u[:], narrow_high.alpha_u[:] = low, high
    overlap = jaccard(top_lists(narrow_low), top_lists(narrow_high))
    print(f"Top-10 overlap between those two alpha values: {overlap:.3f}")


def masking() -> None:
    heading("Validation items left unmasked at test time")
    cognitive = build_cognitive()
    scorers = {
        "A: Popularity": lambda u: model_A_pop.scores.astype(np.float64).copy(),
        "B1: ItemKNN": lambda u: np.asarray(model_B_knn.R_user[u] @ model_B_knn.W).ravel(),
        "B2: SVD": lambda u: (
            model_B_svd.user_vecs[u] @ model_B_svd.item_vecs.T + model_B_svd.user_means[u]
        ),
        "C: Cognitive": lambda u: cognitive.compute_scores_and_explanations(u, set())[0].astype(
            np.float64
        ),
    }
    for name, scorer in scorers.items():
        unmasked, masked, wasted = [], [], []
        for u, group in test_df.groupby("u", sort=False):
            targets = set(group[group["rating"] >= 4.0]["i"].values)
            if not targets:
                continue
            scores = scorer(u)
            scores[list(user_seen[u])] = -1e9
            top = np.argsort(scores)[::-1][:10]
            wasted.append(len(set(top.tolist()) & val_seen.get(u, set())) / 10)
            unmasked.append(ndcg([1.0 if i in targets else 0.0 for i in top], len(targets)))
            scores[list(val_seen.get(u, set()))] = -1e9
            top = np.argsort(scores)[::-1][:10]
            masked.append(ndcg([1.0 if i in targets else 0.0 for i in top], len(targets)))
        print(
            f"{name:14s} slots on the user's own validation items: {100 * np.mean(wasted):4.1f}%"
            f"  NDCG@10 {np.mean(unmasked):.4f} -> {np.mean(masked):.4f} once masked"
        )


def formula_and_normalisation() -> None:
    """Compare fixes on validation only, so that nothing here can influence a test result."""
    targets = {
        int(u): set(group["i"].values) for u, group in val_df[val_df["rating"] >= 4.0].groupby("u")
    }
    base = build_cognitive()

    def evaluate(score_user):
        scores, exposure, top_lists = [], np.zeros(N_ITEMS), {}
        for u, relevant in targets.items():
            ranked = np.array(score_user(u), dtype=np.float64)
            ranked[list(user_seen[u])] = -1e9
            top = np.argsort(ranked)[::-1][:TOP_K]
            scores.append(ndcg([1.0 if i in relevant else 0.0 for i in top], len(relevant)))
            exposure[top] += 1.0
            top_lists[u] = set(top.tolist())
        coverage = 100.0 * float(np.sum(exposure > 0)) / N_ITEMS
        return float(np.mean(scores)), coverage, top_lists

    def blended(scale_attention, scale_score, use_memory_weight):
        def score_user(u):
            buf = base.episodic_buffer[u]
            long_term = base.V @ base.u_LT[u]
            if not buf:
                return long_term
            mem_items = np.array([x[0] for x in buf], dtype=int)
            mem_val = np.array([x[2] for x in buf], dtype=np.float32)
            lag = np.arange(len(buf) - 1, -1, -1, dtype=np.float32)
            raw = base.V[mem_items] @ base.V.T
            root_d = np.sqrt(base.d)
            logits = (raw / root_d if scale_attention else raw) - 0.15 * lag[:, None]
            logits = logits - logits.max(axis=0, keepdims=True)
            attention = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
            weight = mem_val[:, None] if use_memory_weight else 1.0
            short_term = ((attention * weight) * (raw / root_d if scale_score else raw)).sum(axis=0)
            return 0.45 * long_term + 0.55 * short_term

        return score_user

    heading("Matching the plan's own formula (validation only)")
    _, _, long_term_lists = evaluate(lambda u: base.V @ base.u_LT[u])
    variants = {
        "As written": blended(True, True, True),
        "Reward as in the formula": blended(True, False, False),
        "Reward and attention without sqrt(d)": blended(False, False, False),
    }
    for name, score_user in variants.items():
        value, _, top_lists = evaluate(score_user)
        overlap = np.mean(
            [
                len(top_lists[u] & long_term_lists[u]) / len(top_lists[u] | long_term_lists[u])
                for u in targets
            ]
        )
        print(f"{name:38s} NDCG@10 {value:.4f}  top-10 overlap with long-term only {overlap:.3f}")

    heading("Long-term memory alone, with and without L2 normalisation (validation only)")
    raw_vectors = model_B_svd.item_vecs
    raw_users = np.zeros((N_USERS, raw_vectors.shape[1]))
    for u, group in train_df.groupby("u", sort=False):
        weights = np.clip((group["rating"].values - 2.5) / 2.5, -1, 1)
        vector = (weights[:, None] * raw_vectors[group["i"].values]).mean(axis=0)
        raw_users[u] = vector / (np.linalg.norm(vector) + 1e-8)
    for name, score_user in {
        "L2-normalised embeddings": lambda u: base.V @ base.u_LT[u],
        "Raw embeddings": lambda u: raw_vectors @ raw_users[u],
    }.items():
        value, coverage, _ = evaluate(score_user)
        print(f"{name:26s} NDCG@10 {value:.4f}  catalogue coverage {coverage:.1f}%")


if __name__ == "__main__":
    plan_table()
    timestamps()
    global_ordering()
    fidelity_stability()
    noise_artefact()
    memory_connection()
    masking()
    formula_and_normalisation()
