import argparse
import json
from pathlib import Path

from recommender import METHODS, MovieRecommender, load_data, temporal_split


def main():
    parser = argparse.ArgumentParser(description="Рекомендации фильмов")
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--method", choices=METHODS, default="collaborative")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", default="results/recommendations.json")
    args = parser.parse_args()
    movies, ratings = load_data()
    train, _, cutoff = temporal_split(ratings)
    model = MovieRecommender(movies, train)
    recommendations, fallback = model.recommend(args.user, args.method, args.top)
    result = {"user": args.user, "method": args.method, "fallback_to_popular": fallback,
              "train_cutoff": cutoff, "recommendations": recommendations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Пользователь:", args.user, "Способ:", args.method, "Резерв по популярности:", fallback)
    for rank, movie in enumerate(recommendations, start=1):
        print(f"{rank}. {movie['title']} | {', '.join(movie['genres'])} | {movie['score']:.4f}")
    print("Сохранено:", output)


if __name__ == "__main__":
    main()
