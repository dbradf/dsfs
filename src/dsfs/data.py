import csv
import os
import random 
import requests
from typing import Callable, List, Any

from dsfs.linalg.vector import Vector, distance

DATASET_DIR = "datasets"

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


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]
