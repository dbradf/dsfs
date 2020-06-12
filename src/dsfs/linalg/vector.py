import math
from typing import List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "no vectors"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes"

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


def squared_distance(v: Vector, w: Vector) -> float:
    dist = distance(v, w)
    return dist ** 2
