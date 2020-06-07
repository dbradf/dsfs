import random
from typing import Callable, TypeVar, List, Iterator, Tuple

from dsfs.linalg.vector import Vector, add, scalar_multiply, vector_mean

T = TypeVar("T")


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float) -> float:
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)

    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicated = slope * x + intercept
    error = predicated - y
    # squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad


def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


def gradient_decent_linear(
    dataset: List[T],
    learning_rate: float = 0.001,
    batch_size: int = 20,
    n_epochs: int = 1000,
    shuffle: bool = True,
    debug: bool = False,
) -> Tuple[float, float]:
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for epoch in range(n_epochs):
        for batch in minibatches(dataset, batch_size, shuffle):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)

        if debug:
            print(epoch, theta)

    return theta


def stochastic_gradient_decent_linear(
    dataset: List[T],
    learning_rate: float = 0.001,
    batch_size: int = 20,
    n_epochs: int = 1000,
    shuffle: bool = True,
    debug: bool = False,
) -> Tuple[float, float]:
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for epoch in range(n_epochs):
        for x, y in dataset:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)

        if debug:
            print(epoch, theta)

    return theta
