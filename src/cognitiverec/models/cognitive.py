"""Models C and D: dual memory, associative attention, RPE-gated plasticity. Owner: Mudit.

Model D is this model with `exploration_beta > 0`; its exploration layer is in `exploration.py`
(owner: Abhinav). The modules follow section 1 of the project plan:

1. Memory: a long-term vector u_LT and a buffer of the last M positive interactions.
2. Attention: a_j->i = softmax_j(v_j . v_i / temperature - lambda * lag_j).
3. Reward: r_hat(u, i) = ((1 - alpha_u) * u_LT + alpha_u * u_ST(i)) . v_i, u_ST(i) = sum a_j->i v_j.
4. Decision: rank by r_hat, plus the UCB bonus for Model D.
5. Feedback: surprise |r - r_hat| moves alpha_u; then memory and u_LT are updated.

PROJECT_REPORT.md describes the experimental rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cognitiverec.models.base import Recommender
from cognitiverec.models.exploration import ExposureTracker, ucb_bonus
from cognitiverec.models.latent import rating_to_reward, unit_rows, unit_vector
from cognitiverec.models.svd import SVD


class CognitiveRecommender(Recommender):
    """Brain-inspired recommender. Each `use_*` flag and `attention_mode` supports an ablation.

    `normalize_embeddings` chooses the candidate vectors v_i: unit length, or the raw SVD factors
    that keep popularity in their norm. User-side vectors are always built from unit-length item
    vectors, and attention always uses cosine similarity, so this flag only changes the reward.
    """

    def __init__(
        self,
        *,
        n_factors: int = 32,
        memory_size: int = 12,
        base_alpha: float = 0.55,
        decay_lambda: float = 0.15,
        attention_temperature: float = 1.0,
        attention_mode: str = "associative",
        rpe_lr: float = 0.12,
        rpe_center: float = 0.35,
        rpe_smoothing: float = 0.3,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        lt_rate: float = 0.05,
        positive_threshold: float = 4.0,
        exploration_beta: float = 0.0,
        normalize_embeddings: bool = True,
        use_short_term: bool = True,
        use_rpe: bool = True,
        seed: int = 42,
    ) -> None:
        if attention_mode not in ("associative", "uniform"):
            raise ValueError(f"Unknown attention_mode: {attention_mode}")
        self.n_factors = n_factors
        self.memory_size = memory_size
        self.base_alpha = base_alpha
        self.decay_lambda = decay_lambda
        self.attention_temperature = attention_temperature
        self.attention_mode = attention_mode
        self.rpe_lr = rpe_lr
        self.rpe_center = rpe_center
        self.rpe_smoothing = rpe_smoothing
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lt_rate = lt_rate
        self.positive_threshold = positive_threshold
        self.exploration_beta = exploration_beta
        self.normalize_embeddings = normalize_embeddings
        self.use_short_term = use_short_term
        self.use_rpe = use_rpe
        self.seed = seed

    def fit(self, history: pd.DataFrame, n_users: int, n_items: int) -> CognitiveRecommender:
        raw = SVD(n_factors=self.n_factors, seed=self.seed).fit(history, n_users, n_items).item_vecs
        self._unit = unit_rows(raw)
        self._set_embeddings(self._unit if self.normalize_embeddings else raw, n_users)

        for user, group in history.groupby("user", sort=False):
            items = group["item"].to_numpy()
            ratings = group["rating"].to_numpy(dtype=np.float64)
            weighted = rating_to_reward(ratings)[:, None] * self._unit[items]
            self.u_lt[user] = unit_vector(weighted.mean(axis=0))
            positives = items[ratings >= self.positive_threshold]
            self.memory[user] = [int(item) for item in positives[-self.memory_size :]]

        self.exposure = ExposureTracker(np.bincount(history["item"].to_numpy(), minlength=n_items))
        return self

    def _set_embeddings(self, candidate_vectors: np.ndarray, n_users: int) -> None:
        self._candidates = candidate_vectors
        self.u_lt = np.zeros((n_users, candidate_vectors.shape[1]), dtype=np.float64)
        self.memory: list[list[int]] = [[] for _ in range(n_users)]
        self.alpha = np.full(n_users, self.base_alpha, dtype=np.float64)

    def _attention(self, similarities: np.ndarray) -> np.ndarray:
        """Return attention weights over memory for each candidate, shape (M, candidates).

        Lag is position in the buffer rather than elapsed seconds, because most MovieLens 100K
        ratings share a timestamp with another rating by the same user.
        """
        size = similarities.shape[0]
        if self.attention_mode == "uniform":
            return np.full_like(similarities, 1.0 / size)
        lags = np.arange(size - 1, -1, -1, dtype=np.float64)[:, None]
        logits = similarities / self.attention_temperature - self.decay_lambda * lags
        logits -= logits.max(axis=0, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=0, keepdims=True)

    def _short_term(
        self, memory: list[int], candidates: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return attention weights and the short-term score u_ST(i) . v_i for each candidate."""
        memory_vectors = self._unit[memory]
        unit_targets = self._unit if candidates is None else self._unit[candidates]
        similarities = memory_vectors @ unit_targets.T
        weights = self._attention(similarities)
        if self.normalize_embeddings:
            reward_similarities = similarities
        else:
            targets = self._candidates if candidates is None else self._candidates[candidates]
            reward_similarities = memory_vectors @ targets.T
        return weights, (weights * reward_similarities).sum(axis=0)

    def _expected_reward(
        self, user: int, memory: list[int], candidates: np.ndarray | None = None
    ) -> np.ndarray:
        """Return r_hat for each candidate. With no usable memory this is long-term only."""
        targets = self._candidates if candidates is None else self._candidates[candidates]
        long_term = targets @ self.u_lt[user]
        if not self.use_short_term or not memory:
            return long_term
        _, short_term = self._short_term(memory, candidates)
        alpha = self.alpha[user]
        return (1.0 - alpha) * long_term + alpha * short_term

    def _bonus(self) -> np.ndarray | float:
        if self.exploration_beta <= 0.0:
            return 0.0
        return ucb_bonus(self.exposure.counts, self.exposure.total, self.exploration_beta)

    def score(self, user: int) -> np.ndarray:
        return self._expected_reward(user, self.memory[user]) + self._bonus()

    def observe(self, user: int, item: int, rating: float, timestamp: float) -> None:
        """Apply the feedback module: surprise moves alpha, then memory and u_LT update.

        alpha moves part of the way towards its target rather than jumping there, so one surprising
        interaction cannot fully redirect a user's plasticity.
        """
        reward = float(rating_to_reward(rating))
        if self.use_rpe:
            expected = self._expected_reward(user, self.memory[user], np.array([item]))[0]
            surprise = abs(reward - float(np.clip(expected, -1.0, 1.0)))
            target = np.clip(
                self.base_alpha + self.rpe_lr * (surprise - self.rpe_center),
                self.alpha_min,
                self.alpha_max,
            )
            self.alpha[user] = (1.0 - self.rpe_smoothing) * self.alpha[
                user
            ] + self.rpe_smoothing * target

        if rating >= self.positive_threshold:
            buffer = self.memory[user]
            buffer.append(int(item))
            del buffer[: -self.memory_size]

        consolidated = (1.0 - self.lt_rate) * self.u_lt[user] + self.lt_rate * reward * self._unit[
            item
        ]
        self.u_lt[user] = unit_vector(consolidated)

    def record_impression(self, items: np.ndarray) -> None:
        self.exposure.record(items)

    def evidence_items(self, user: int) -> np.ndarray:
        return np.asarray(self.memory[user], dtype=np.int64)

    def score_components(self, user: int, item: int) -> dict[str, float]:
        """Decompose the displayed score into its actual additive components."""
        long_term = float(self._candidates[item] @ self.u_lt[user])
        reward = float(self._expected_reward(user, self.memory[user], np.array([item]))[0])
        if self.use_short_term and self.memory[user]:
            long_term *= 1.0 - self.alpha[user]
        bonus = self._bonus()
        return {
            "long_term": long_term,
            "short_term": reward - long_term,
            "exploration": float(bonus[item]) if isinstance(bonus, np.ndarray) else float(bonus),
        }

    def attributions(self, user: int, item: int) -> np.ndarray:
        """Return attention over memory for `item`; empty when short-term memory is off."""
        memory = self.memory[user]
        if not self.use_short_term or not memory:
            return np.empty(0)
        return self._short_term(memory, np.array([item]))[0][:, 0]

    def score_without(self, user: int, entry: int) -> np.ndarray:
        """Score as if one memory entry were absent; remaining lags are recomputed.

        Only the buffer changes: u_LT is left as it is, so this isolates short-term attention.
        """
        memory = self.memory[user]
        return self._expected_reward(user, memory[:entry] + memory[entry + 1 :]) + self._bonus()

    def memory_diagnostics(self, user: int, exclude: np.ndarray) -> dict[str, float] | None:
        """Measure whether memory reaches the ranking. None when the user has no memory."""
        memory = self.memory[user]
        if not memory:
            return None
        keep = np.ones(self._unit.shape[0], dtype=bool)
        keep[exclude] = False
        weights, short_term = self._short_term(memory)
        long_term = self._candidates @ self.u_lt[user]
        return {
            "long_term_spread": float(long_term[keep].std()),
            "short_term_spread": float(short_term[keep].std()),
            "recency_share": float(np.mean(weights[:, keep].argmax(axis=0) == len(memory) - 1)),
        }
