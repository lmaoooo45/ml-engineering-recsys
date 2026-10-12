from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parent / "data" / "ml-100k"
METHODS = ("content", "collaborative", "popular")


def load_data(directory=DATA):
    directory = Path(directory)
    if not (directory / "u.data").is_file():
        raise FileNotFoundError("Сначала запустите download_data.py")
    genres = [line.split("|")[0] for line in (directory / "u.genre").read_text().splitlines() if line]
    movies = []
    for line in (directory / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        tags = [genre for genre, flag in zip(genres, fields[5:]) if flag == "1"]
        movies.append({"id": int(fields[0]), "title": fields[1], "genres": tags})
    movies.sort(key=lambda movie: movie["id"])
    ratings = np.loadtxt(directory / "u.data", dtype=np.int64)
    return movies, ratings


def temporal_split(ratings, train_fraction=0.8):
    if not 0 < train_fraction < 1:
        raise ValueError("Доля обучения должна быть между 0 и 1")
    timestamps = np.sort(ratings[:, 3])
    cutoff = int(timestamps[int(len(timestamps) * train_fraction) - 1])
    train = ratings[ratings[:, 3] <= cutoff]
    test = ratings[ratings[:, 3] > cutoff]
    if not len(train) or not len(test):
        raise ValueError("Для разбиения нужны разные временные отметки")
    return train, test, cutoff


class MovieRecommender:
    def __init__(self, movies, train, neighbors=30):
        if neighbors < 1:
            raise ValueError("Число соседей должно быть положительным")
        self.movies = movies
        self.neighbors = neighbors
        self.item_ids = np.array([movie["id"] for movie in movies])
        self.item_index = {int(item): index for index, item in enumerate(self.item_ids)}
        self.user_ids = np.unique(train[:, 0])
        self.user_index = {int(user): index for index, user in enumerate(self.user_ids)}
        self.ratings = np.zeros((len(self.user_ids), len(movies)), dtype=np.float32)
        for user, item, rating, _ in train:
            self.ratings[self.user_index[int(user)], self.item_index[int(item)]] = rating
        self.seen = self.ratings > 0
        self.likes = (self.ratings >= 4).astype(np.float32)
        self.popularity = self.likes.sum(axis=0)
        self.available = self.seen.any(axis=0)
        descriptions = [" ".join(movie["genres"]) or "unknown" for movie in movies]
        self.vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
        self.vectorizer.fit([text for text, available in zip(descriptions, self.available) if available])
        self.features = self.vectorizer.transform(descriptions)
        self.user_similarity = cosine_similarity(self.likes)
        np.fill_diagonal(self.user_similarity, 0)

    def score(self, user, method):
        if method not in METHODS:
            raise ValueError("Неизвестный способ рекомендаций")
        index = self.user_index.get(user)
        if method == "popular" or index is None:
            return self.popularity.copy(), method != "popular"
        if method == "content":
            liked = np.flatnonzero(self.likes[index])
            if not len(liked):
                return self.popularity.copy(), True
            profile = np.asarray(self.features[liked].mean(axis=0))
            return cosine_similarity(self.features, profile).ravel(), False
        similarities = self.user_similarity[index]
        order = np.lexsort((self.user_ids, -similarities))[:self.neighbors]
        order = order[similarities[order] > 0]
        if not len(order):
            return self.popularity.copy(), True
        weights = similarities[order]
        scores = weights @ self.likes[order] / weights.sum()
        return scores, False

    def recommend(self, user, method, k=10):
        if user < 0 or k < 1:
            raise ValueError("ID пользователя неотрицательный, k положительный")
        scores, fallback = self.score(user, method)
        allowed = self.available.copy()
        index = self.user_index.get(user)
        if index is not None:
            allowed &= ~self.seen[index]
        candidates = np.flatnonzero(allowed)
        # При равенстве оценок сначала идёт фильм с меньшим ID.
        order = np.lexsort((self.item_ids[candidates], -scores[candidates]))[:k]
        rows = []
        for position in candidates[order]:
            rows.append({**self.movies[position], "score": float(scores[position])})
        return rows, fallback
