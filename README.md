# Рекомендации фильмов

Другие части: [работа с моделями](https://github.com/lmaoooo45/ml-engineering-1), [API и тесты](https://github.com/lmaoooo45/ml-engineering-api).

Третья часть задания. Два классических способа: рекомендации по жанрам (content-based) и по похожим пользователям (user-based collaborative filtering). Простой вариант для сравнения: популярность по числу оценок 4 и 5.

Данные: MovieLens 100K, 100000 оценок, 943 пользователя, 1682 фильма. Оценки разделяются по времени: примерно 80% ранних записей для построения рекомендаций, остальные для проверки. Все оценки с одинаковой временной отметкой попадают в одну часть. Уже оценённые фильмы не предлагаются. Новому пользователю выдаётся список по популярности.

Установка на Windows с Python 3.12 из папки проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
```

Три способа для одного пользователя:

```powershell
.\.venv\Scripts\python.exe recommend.py --user 1 --method content --output results/content.json
.\.venv\Scripts\python.exe recommend.py --user 1 --method collaborative --output results/collaborative.json
.\.venv\Scripts\python.exe recommend.py --user 1 --method popular --output results/popular.json
```

Проверка нового пользователя, сравнение и тесты:

```powershell
.\.venv\Scripts\python.exe recommend.py --user 0 --method collaborative --output results/cold_start.json
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m pytest -v
```

В `results/evaluation.json` сохраняются Precision@10, Recall@10, NDCG@10, покрытие каталога и время ответа. Есть отдельные результаты для пользователей с историей. `per_user.json` содержит списки рекомендованных и релевантных ID для проверки расчётов. Релевантным считается фильм с тестовой оценкой 4 или 5, имеющий хотя бы одну обучающую оценку. Фильмы без обучающих оценок исключаются из кандидатов для всех трёх способов.

`score` служит для сортировки, это не вероятность и не предсказанная оценка от 1 до 5. Значения разных способов напрямую не сравниваются. Сравнение бизнес-показателей и ограничения приведены в `report.pdf`. Производственный сервис и настоящий A/B-тест не запускались.

Данные скачиваются с [GroupLens](https://grouplens.org/datasets/movielens/100k/). Набор нельзя включать в публичный репозиторий без отдельного разрешения владельца, поэтому папка `data` и файл `per_user.json` с эталонными взаимодействиями исключены из Git и архива проекта. Они создаются локально при запуске. 
