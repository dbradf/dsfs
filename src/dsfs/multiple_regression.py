import random
import tqdm
from typing import List, Callable, TypeVar, Tuple

from dsfs.linalg.vector import Vector, dot, vector_mean, add
from dsfs.grad_descent import gradient_step
from dsfs.linear_regression import total_sum_of_squares
from dsfs.stats.probability import normal_cdf

X = TypeVar("X")
Stat = TypeVar("Stat")


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def least_squares_fit(
    xs: List[Vector],
    ys: List[float],
    learning_rate: float = 0.001,
    num_steps: int = 1000,
    batch_size: int = 1,
) -> Vector:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start : start + batch_size]
            batch_ys = ys[start : start + batch_size]

            gradient = vector_mean(
                [sqerror_gradient(x, y, guess) for x, y in zip(batch_xs, batch_ys)]
            )
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]


def bootstrap_statistic(
    data: List[X], stats_fn: Callable[[List[X]], Stat], num_samples: int
) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(101)]


def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(x_sample, y_sample, 0.001, 5000, 25)
    return beta


def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x: Vector, y: float, beta: Vector, alpha: float) -> float:
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    return [0.0] + [2 * alpha * beta_j for beta_j in beta[1:]]


def sqerror_ridge_gradient(x: Vector, y: float, beta: Vector, alpha: float) -> Vector:
    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))
