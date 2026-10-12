import numpy as np
import pytest

from evaluate import ranking_metrics
from recommender import METHODS, MovieRecommender, temporal_split


@pytest.fixture
def model():
    movies = [{"id": i, "title": str(i), "genres": ["Action" if i < 4 else "Comedy"]} for i in range(1, 5)]
    train = np.array([[1, 1, 5, 1], [2, 1, 5, 2], [2, 3, 5, 3], [3, 3, 5, 4], [3, 4, 5, 5]])
    return MovieRecommender(movies, train)


@pytest.mark.parametrize("method", METHODS)
def test_no_seen_or_unavailable_movies(model, method):
    rows, _ = model.recommend(1, method)
    assert {row["id"] for row in rows} == {3, 4}
    assert len(rows) == len({row["id"] for row in rows})


@pytest.mark.parametrize("method", ["content", "collaborative"])
def test_personalized_ranking(model, method):
    rows, fallback = model.recommend(1, method)
    assert rows[0]["id"] == 3
    assert not fallback


@pytest.mark.parametrize("method", ["content", "collaborative"])
def test_cold_start_uses_popularity(model, method):
    rows, fallback = model.recommend(999, method)
    expected, _ = model.recommend(999, "popular")
    assert rows == expected
    assert fallback


def test_temporal_split_keeps_equal_timestamps_together():
    ratings = np.array([[1, i + 1, 5, t] for i, t in enumerate([1, 2, 2, 3, 4])])
    train, test, _ = temporal_split(ratings, 0.4)
    assert train[:, 3].max() < test[:, 3].min()
    assert len(train) + len(test) == len(ratings)


def test_metrics_use_rank_and_all_relevant_items():
    precision, recall, ndcg = ranking_metrics([1, 2], {2, 3}, 2)
    assert precision == recall == 0.5
    assert ndcg == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_invalid_method(model):
    with pytest.raises(ValueError):
        model.recommend(1, "unknown")
