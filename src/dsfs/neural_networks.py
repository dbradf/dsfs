import math
from typing import List

from dsfs.linalg.vector import Vector, dot


def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0


def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))


def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)

        input_vector = output

    return outputs


def sqerror_gradients(
    network: List[List[Vector]], input_vector: Vector, target_vector: Vector
) -> List[List[Vector]]:
    hidden_outputs, outputs = feed_forward(network, input_vector)

    output_deltas = [
        output * (1 - output) * (output - target) for output, target in zip(outputs, target_vector)
    ]

    output_grads = [
        [output_deltas[i] * hidden_output for hidden_output in hidden_outputs + [1]]
        for i, output_neuron in enumerate(network[-1])
    ]
    hidden_deltas = [
        hidden_output * (1 - hidden_output) * dot(output_deltas, [n[i] for n in network[-1]])
        for i, hidden_output in enumerate(hidden_outputs)
    ]
    hidden_grads = [
        [hidden_deltas[i] * input for input in input_vector + [1]]
        for i, hidden_neuron in enumerate(network[0])
    ]
    return [hidden_grads, output_grads]
