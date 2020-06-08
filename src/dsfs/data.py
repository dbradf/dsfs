import csv
from io import BytesIO
import os
import random
import requests
import tarfile
from typing import Callable, List, Any

from dsfs.linalg.vector import Vector, distance

DATASET_DIR = "datasets"
SPAM_DIR = "spam_data"
BASE_SPAM_URL = "https://spamassassin.apache.org/old/publiccorpus"
SPAM_FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
]

DATASETS = {
    "iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
}


def get_dataset(name: str, parser: Callable) -> List[Any]:
    location = os.path.join(DATASET_DIR, f"{name}.csv")
    if not os.path.exists(location):
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)

        data = requests.get(DATASETS[name])
        with open(location, "w") as f:
            f.write(data.text)

    with open(location) as f:
        reader = csv.reader(f)
        return [parser(row) for row in reader if row]


def get_spam_data():
    target_dir = os.path.join(DATASET_DIR, SPAM_DIR)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in SPAM_FILES:
        content = requests.get(f"{BASE_SPAM_URL}/{filename}").content

        fin = BytesIO(content)

        with tarfile.open(fileobj=fin, mode="r:bz2") as tf:
            tf.extractall(target_dir)


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]
