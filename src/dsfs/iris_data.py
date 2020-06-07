from collections import defaultdict
import random
from typing import List, Tuple, Dict

from dsfs.data import get_dataset
from dsfs.k_nearest_neighbors import LabeledPoint, knn_classify
from dsfs.training_ml import split_data


def parse_iris_row(row: List[str]) -> LabeledPoint:
    measurements = [float(value) for value in row[:-1]]
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)


def main():
    random.seed(12)

    iris_data = get_dataset("iris", parse_iris_row)
    iris_train, iris_test = split_data(iris_data, 0.7)

    confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for iris in iris_test:
        predicted = knn_classify(5, iris_train, iris.point)
        actual = iris.label

        if predicted == actual:
            num_correct += 1

        confusion_matrix[(predicted, actual)] += 1

    pct_correct = num_correct / len(iris_test)
    print(pct_correct, confusion_matrix)
