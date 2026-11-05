"""MovieLens 100K: download, load and index. Owner: Mudit."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
FILES = ("u.data", "u.item")
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


@dataclass(frozen=True)
class Dataset:
    """Interactions and item metadata, using contiguous 0-based user and item ids."""

    interactions: pd.DataFrame  # columns: user, item, rating, timestamp
    item_genres: np.ndarray  # shape (n_items, n_genres), binary
    item_titles: tuple[str, ...]
    genre_names: tuple[str, ...]
    n_users: int
    n_items: int


def ensure_movielens_100k(data_dir: str | Path) -> Path:
    """Download the files this project uses, if missing, and return their directory."""
    target = Path(data_dir) / "ml-100k"
    if all((target / name).exists() for name in FILES):
        return target
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in FILES:
            (target / name).write_bytes(archive.read(f"ml-100k/{name}"))
    return target


def load_movielens_100k(data_dir: str | Path) -> Dataset:
    """Load MovieLens 100K. The item catalogue comes from `u.item`, not from the ratings."""
    directory = ensure_movielens_100k(data_dir)
    ratings = pd.read_csv(
        directory / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    items = pd.read_csv(
        directory / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=["item_id", "title", "release_date", "video_release", "imdb_url", *GENRES],
    ).sort_values("item_id", kind="stable")

    user_index = {user: index for index, user in enumerate(sorted(ratings["user_id"].unique()))}
    item_index = {item: index for index, item in enumerate(items["item_id"])}

    # File order is kept: u.data is in random order, which is what decides the order of ratings
    # that share a timestamp (see `chronological_split`).
    interactions = pd.DataFrame(
        {
            "user": ratings["user_id"].map(user_index).to_numpy(dtype=np.int64),
            "item": ratings["item_id"].map(item_index).to_numpy(dtype=np.int64),
            "rating": ratings["rating"].to_numpy(dtype=np.float64),
            "timestamp": ratings["timestamp"].to_numpy(dtype=np.int64),
        }
    )
    return Dataset(
        interactions=interactions,
        item_genres=items[list(GENRES)].to_numpy(dtype=np.int8),
        item_titles=tuple(items["title"]),
        genre_names=GENRES,
        n_users=len(user_index),
        n_items=len(item_index),
    )
