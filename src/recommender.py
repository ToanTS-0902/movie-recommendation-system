"""Reusable item-based KNN recommender for MovieLens-style data."""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Recommendation:
    title: str
    distance: float


class MovieRecommender:
    """Item-based collaborative filtering using cosine-distance KNN."""

    def __init__(self, neighbors: int = 20, min_ratings: int = 10) -> None:
        self.neighbors = neighbors
        self.min_ratings = min_ratings
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.movies: pd.DataFrame | None = None
        self.matrix: csr_matrix | None = None
        self.movie_ids: list[int] = []

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "MovieRecommender":
        required_movie_columns = {"movieId", "title"}
        required_rating_columns = {"userId", "movieId", "rating"}
        if not required_movie_columns.issubset(movies.columns):
            raise ValueError(f"movies data must contain {sorted(required_movie_columns)}")
        if not required_rating_columns.issubset(ratings.columns):
            raise ValueError(f"ratings data must contain {sorted(required_rating_columns)}")

        counts = ratings.groupby("movieId").size()
        eligible_ids = counts[counts >= self.min_ratings].index
        filtered = ratings[ratings["movieId"].isin(eligible_ids)]
        pivot = filtered.pivot_table(
            index="movieId", columns="userId", values="rating", fill_value=0
        )
        if pivot.empty:
            raise ValueError("No movies remain after applying the minimum-rating filter.")

        self.movie_ids = [int(movie_id) for movie_id in pivot.index]
        self.matrix = csr_matrix(pivot.values)
        self.movies = movies.drop_duplicates("movieId").set_index("movieId")
        self.model.fit(self.matrix)
        return self

    def _resolve_movie_id(self, title: str) -> int:
        if self.movies is None:
            raise RuntimeError("Call fit before requesting recommendations.")
        available = self.movies.loc[self.movies.index.intersection(self.movie_ids)]
        normalized = {str(name).casefold(): int(movie_id) for movie_id, name in available["title"].items()}
        match = difflib.get_close_matches(title.casefold(), normalized.keys(), n=1, cutoff=0.45)
        if not match:
            raise ValueError(f"No close movie title found for: {title}")
        return normalized[match[0]]

    def recommend(self, title: str, count: int = 10) -> list[Recommendation]:
        if self.matrix is None or self.movies is None:
            raise RuntimeError("Call fit before requesting recommendations.")
        movie_id = self._resolve_movie_id(title)
        row_index = self.movie_ids.index(movie_id)
        neighbor_count = min(count + 1, len(self.movie_ids))
        distances, indices = self.model.kneighbors(
            self.matrix[row_index], n_neighbors=neighbor_count
        )

        recommendations: list[Recommendation] = []
        for distance, index in zip(distances[0], indices[0]):
            candidate_id = self.movie_ids[int(index)]
            if candidate_id == movie_id:
                continue
            recommendations.append(
                Recommendation(
                    title=str(self.movies.loc[candidate_id, "title"]),
                    distance=float(distance),
                )
            )
        return recommendations[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend similar movies with item-based KNN")
    parser.add_argument("title", help="movie title, for example 'Toy Story (1995)'")
    parser.add_argument("--movies", type=Path, default=Path("data/movie.csv"))
    parser.add_argument("--ratings", type=Path, default=Path("data/rating.csv"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-ratings", type=int, default=10)
    args = parser.parse_args()

    recommender = MovieRecommender(min_ratings=args.min_ratings).fit(
        pd.read_csv(args.movies), pd.read_csv(args.ratings)
    )
    for rank, item in enumerate(recommender.recommend(args.title, args.count), start=1):
        similarity = 1 - item.distance
        print(f"{rank:>2}. {item.title} (similarity={similarity:.3f})")


if __name__ == "__main__":
    main()
