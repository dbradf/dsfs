import json
import math
import operator
import random
from typing import List, Callable, Iterable

from dsfs.linalg.vector import dot
from dsfs.neural_networks import sigmoid
from dsfs.stats.probability import inverse_normal_cdf

Tensor = list


def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes


def is_1d(tensor: Tensor) -> bool:
    return not isinstance(tensor[0], list)


def tensor_sum(tensor: Tensor) -> float:
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)


def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]


def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)


def tensor_combine(f: Callable[[float, float], float], t1: Tensor, t2: Tensor) -> Tensor:
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]


class Layer:
    def forward(self, inp):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        return ()

    def grads(self) -> Iterable[Tensor]:
        return ()


class Sigmoid(Layer):
    def forward(self, inp: Tensor) -> Tensor:
        self.sigmoids = tensor_apply(sigmoid, inp)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad, self.sigmoids, gradient)


def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]


def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])]


def random_tensor(*dims: int, init: str = "normal") -> Tensor:
    if init == "normal":
        return random_normal(*dims)
    elif init == "uniform":
        return random_uniform(*dims)
    elif init == "xavier":
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str = "xavier") -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = random_tensor(output_dim, input_dim, init=init)
        self.b = random_tensor(output_dim, init=init)

    def forward(self, inp: Tensor) -> Tensor:
        self.inp = inp
        return [dot(inp, self.w[o]) + self.b[o] for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        self.b_grad = gradient
        self.w_grad = [
            [self.inp[i] * gradient[o] for i in range(self.input_dim)]
            for o in range(self.output_dim)
        ]

        return [
            sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
            for i in range(self.input_dim)
        ]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]


class Sequential(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        return (grad for layer in self.layers for grad in layer.grads())


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class SSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2, predicted, actual
        )
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(lambda predicted, actual: 2 * (predicted - actual), predicted, actual)


class Optimzer:
    def step(self, layer: Layer) -> None:
        raise NotImplementedError


class GradientDescent(Optimzer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = tensor_combine(lambda param, grad: param - grad * self.lr, param, grad)


class Momentum(Optimzer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []

    def step(self, layer: Layer) -> None:
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            update[:] = tensor_combine(lambda u, g: self.mo * u + (1 - self.mo) * g, update, grad)
            param[:] = tensor_combine(lambda p, u: p - self.lr * u, param, update)


def tanh(x: float) -> float:
    if x < -100:
        return -1
    elif x > 100:
        return 1

    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)


class Tanh(Layer):
    def forward(self, inp: Tensor) -> Tensor:
        self.tanh = tensor_apply(tanh, inp)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda tanh, grad: (1 - tanh ** 2) * grad, self.tanh, gradient)


class Relu(Layer):
    def forward(self, inp: Tensor) -> Tensor:
        self.inp = inp
        return tensor_apply(lambda x: max(x, 0), inp)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0, self.inp, gradient)


def softmax(tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor]


class SoftmaxCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        probabilities = softmax(predicted)

        likelihoods = tensor_combine(
            lambda p, act: math.log(p + 1e-30) * act, probabilities, actual
        )
        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)
        return tensor_combine(lambda p, actual: p - actual, probabilities, actual)


class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True

    def forward(self, inp: Tensor) -> Tensor:
        if self.train:
            self.mask = tensor_apply(lambda _: 0 if random.random() < self.p else 1, inp)
            return tensor_combine(operator.mul, inp, self.mask)
        else:
            return tensor_apply(lambda x: x * (1 - self.p), inp)

    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            return tensor_combine(operator.mul, gradient, self.mask)
        else:
            raise RuntimeError("don't call backward when not in train mode")


def save_weights(model: Layer, filename: str) -> None:
    weights = list(model.params())
    with open(filename, "w") as f:
        json.dump(weights, f)


def load_weights(model: Layer, filename: str) -> None:
    with open(filename) as f:
        weights = json.load(f)

    assert all(shape(param) == shape(weight) for param, weight in zip(model.params(), weights))

    for param, weight in zip(model.params(), weights):
        param[:] = weight
