import hashlib
import io
import json
import ssl
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import certifi

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA = Path(__file__).resolve().parent / "data" / "ml-100k"


def main():
    names = ["u.data", "u.item", "u.genre", "README"]
    if all((DATA / name).is_file() for name in names):
        print("Данные уже скачаны:", DATA)
        return
    with urlopen(URL, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        archive = response.read()
    DATA.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(archive)) as source:
        for name in names:
            (DATA / name).write_bytes(source.read("ml-100k/" + name))
    info = {"source": URL, "sha256": hashlib.sha256(archive).hexdigest()}
    (DATA / "download.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Данные сохранены:", DATA)


if __name__ == "__main__":
    main()
