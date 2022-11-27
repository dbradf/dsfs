import csv
from io import BytesIO
import os
import random
import re
import tarfile
from typing import Callable, List, Any

from bs4 import BeautifulSoup
import requests

from dsfs.linalg.vector import Vector, distance

NLP_URL = "https://www.oreilly.com/ideas/what-is-data-science"
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

num_friends = [
    100.0,
    49,
    41,
    40,
    25,
    21,
    21,
    19,
    19,
    18,
    18,
    16,
    15,
    15,
    15,
    15,
    14,
    14,
    13,
    13,
    13,
    13,
    12,
    12,
    11,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
daily_minutes = [
    1,
    68.77,
    51.25,
    52.08,
    38.36,
    44.54,
    57.13,
    51.4,
    41.42,
    31.22,
    34.76,
    54.01,
    38.79,
    47.59,
    49.1,
    27.66,
    41.03,
    36.73,
    48.65,
    28.12,
    46.62,
    35.57,
    32.98,
    35,
    26.07,
    23.77,
    39.73,
    40.57,
    31.65,
    31.21,
    36.32,
    20.45,
    21.93,
    26.02,
    27.34,
    23.49,
    46.94,
    30.5,
    33.8,
    24.23,
    21.4,
    27.94,
    32.24,
    40.57,
    25.07,
    19.42,
    22.39,
    18.42,
    46.96,
    23.72,
    26.41,
    26.97,
    36.76,
    40.32,
    35.02,
    29.47,
    30.2,
    31,
    38.11,
    38.18,
    36.31,
    21.03,
    30.86,
    36.07,
    28.66,
    29.08,
    37.28,
    15.28,
    24.17,
    22.31,
    30.17,
    25.53,
    19.85,
    35.37,
    44.6,
    17.23,
    13.47,
    26.33,
    35.02,
    32.09,
    24.81,
    19.33,
    28.77,
    24.26,
    31.98,
    25.73,
    24.86,
    16.28,
    34.51,
    15.23,
    39.72,
    40.8,
    26.06,
    35.76,
    34.76,
    16.13,
    44.04,
    18.03,
    19.65,
    32.62,
    35.59,
    39.43,
    14.18,
    35.24,
    40.13,
    41.82,
    35.45,
    36.07,
    43.67,
    24.61,
    20.9,
    21.9,
    18.79,
    27.61,
    27.21,
    26.61,
    29.77,
    20.59,
    27.53,
    13.82,
    33.2,
    25,
    33.1,
    36.65,
    18.63,
    14.87,
    22.2,
    36.81,
    25.53,
    24.62,
    26.25,
    18.21,
    28.08,
    19.42,
    29.79,
    32.8,
    35.99,
    28.32,
    27.79,
    35.88,
    29.06,
    36.28,
    14.1,
    36.63,
    37.49,
    26.9,
    18.58,
    38.48,
    24.48,
    18.95,
    33.55,
    14.24,
    29.04,
    32.51,
    25.63,
    22.22,
    19,
    32.73,
    15.16,
    13.9,
    27.2,
    32.01,
    29.27,
    33,
    13.74,
    20.42,
    27.32,
    18.23,
    35.35,
    28.48,
    9.08,
    24.62,
    20.12,
    35.26,
    19.92,
    31.02,
    16.49,
    12.16,
    30.7,
    31.22,
    34.65,
    13.13,
    27.51,
    33.2,
    31.57,
    14.1,
    33.42,
    17.44,
    10.12,
    24.42,
    9.82,
    23.39,
    30.93,
    15.03,
    21.67,
    31.09,
    33.29,
    22.61,
    26.89,
    23.48,
    8.38,
    27.81,
    32.35,
    23.84,
]
daily_hours = [dm / 60 for dm in daily_minutes]

outlier = num_friends.index(100)  # index of outlier

num_friends_good = [x for i, x in enumerate(num_friends) if i != outlier]

daily_minutes_good = [x for i, x in enumerate(daily_minutes) if i != outlier]

daily_hours_good = [dm / 60 for dm in daily_minutes_good]

cluster_inputs: List[List[float]] = [
    [-14, -5],
    [13, 13],
    [20, 23],
    [-19, -11],
    [-9, -16],
    [21, 27],
    [-49, 15],
    [26, 13],
    [-46, 5],
    [-34, -1],
    [11, 15],
    [-49, 0],
    [-22, -16],
    [19, 28],
    [-12, -8],
    [-13, -19],
    [-41, 8],
    [-11, -6],
    [-25, -9],
    [-18, -3],
]


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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, target_dir)


def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]


def random_distances(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]


def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary


def argmax(xs: List) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]


def fix_unicode(text: str) -> str:
    return text.replace("\u2019", "'")


def get_nlp_data():
    html = requests.get(NLP_URL).text
    soup = BeautifulSoup(html, "html5lib")

    contents = soup.find("div", "main-post-radar-content")
    regex = r"[\w']+|[\.]"

    document = []

    for paragraph in contents("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document


nlp_cluster_data = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"],
]


def get_companies():
    url = "https://www.ycombinator.com/topcompanies/"
    soup = BeautifulSoup(requests.get(url).text, "html5lib")

    return list({b.text for b in soup("b") if "h4" in b.get("class", ())})
