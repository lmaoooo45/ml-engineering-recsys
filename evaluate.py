import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn

from recommender import METHODS, MovieRecommender, load_data, temporal_split


def ranking_metrics(recommended, relevant, k):
    hits = np.array([item in relevant for item in recommended], dtype=float)
    precision = float(hits.sum() / k)
    recall = float(hits.sum() / len(relevant)) if relevant else 0.0
    dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum())
    return precision, recall, dcg / ideal if ideal else 0.0


def main():
    parser = argparse.ArgumentParser(description="Сравнение трёх способов рекомендаций")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top должен быть положительным")
    movies, ratings = load_data()
    train, test, cutoff = temporal_split(ratings)
    started = perf_counter()
    model = MovieRecommender(movies, train)
    build_seconds = perf_counter() - started
    eligible_items = set(model.item_ids[model.available].tolist())
    relevant = {}
    skipped_positive = 0
    for user, item, rating, _ in test:
        if rating >= 4:
            if int(item) in eligible_items:
                relevant.setdefault(int(user), set()).add(int(item))
            else:
                skipped_positive += 1
    users = sorted(relevant)
    summaries = {}
    per_user = []
    for method in METHODS:
        values, durations, union, fallback_count = [], [], set(), 0
        for user in users:
            started = perf_counter()
            rows, fallback = model.recommend(user, method, args.top)
            durations.append((perf_counter() - started) * 1000)
            ids = [movie["id"] for movie in rows]
            assert not set(ids) & set(train[train[:, 0] == user, 1].tolist())
            metrics = ranking_metrics(ids, relevant[user], args.top)
            values.append(metrics)
            union.update(ids)
            fallback_count += int(fallback)
            per_user.append({"user": user, "method": method, "recommended": ids,
                             "relevant": sorted(relevant[user]), "precision": metrics[0],
                             "recall": metrics[1], "ndcg": metrics[2]})
        mean = np.mean(values, axis=0)
        warm_values = [value for user, value in zip(users, values) if user in model.user_index]
        warm_mean = np.mean(warm_values, axis=0)
        summaries[method] = {"precision_at_k": float(mean[0]), "recall_at_k": float(mean[1]),
                             "ndcg_at_k": float(mean[2]), "catalog_coverage": len(union) / len(eligible_items),
                             "recommended_unique": len(union), "mean_latency_ms": float(np.mean(durations)),
                             "p95_latency_ms": float(np.percentile(durations, 95)), "fallback_users": fallback_count,
                             "warm_users": {"n": len(warm_values), "precision_at_k": float(warm_mean[0]),
                                            "recall_at_k": float(warm_mean[1]), "ndcg_at_k": float(warm_mean[2])}}
    summary = {"dataset": "MovieLens 100K", "python": platform.python_version(),
               "numpy": np.__version__, "sklearn": sklearn.__version__, "k": args.top,
               "ratings": len(ratings), "movies": len(movies), "users_total": len(np.unique(ratings[:, 0])),
               "train_ratings": len(train), "test_ratings": len(test), "cutoff_timestamp": cutoff,
               "train_max_timestamp": int(train[:, 3].max()), "test_min_timestamp": int(test[:, 3].min()),
               "evaluated_users": len(users), "cold_users_evaluated": sum(u not in model.user_index for u in users),
               "eligible_movies": len(eligible_items), "excluded_test_positive_new_items": skipped_positive,
               "build_seconds": build_seconds, "methods": summaries}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_name("per_user.json").write_text(json.dumps(per_user, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
