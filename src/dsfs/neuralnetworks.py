"""Module containing functions from Chapter 18."""
import itertools
import math

from src.dsfs.vector import dot


# Perceptrons
def step_function(x):
    return 1 if x >= 0 else 0


def perceptron_output(weights, bias, x):
    """Returns 1 if the perceptron 'fires', 0 if not."""
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def example_base_gate_test(gate_name, weights, bias, test_cases):
    for x, expected in test_cases:
        assert perceptron_output(weights, bias, x) == expected
        print(f"Gate {gate_name} test {x}, {expected} passed.")


def example_and_gate():
    cases = [
        ([0, 0], 0),
        ([1, 0], 0),
        ([0, 1], 0),
        ([1, 1], 1)
    ]
    example_base_gate_test('AND', [2, 2], -3, cases)


def example_or_gate():
    cases = [
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 1], 1)
    ]
    example_base_gate_test('OR', [2, 2], -1, cases)


# Feed-Forward neural networks
def sigmoid(t):
    return 1 / (1 + math.exp(-t))


def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """Takes ina neural network (which is represented as a list of lists of
    lists of weights) and returns the output from forward-propagating the
    input."""
    outputs = []
    # process layers
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs


def example_xor_gate():
    xor_network = [[[20, 20, -30],
                    [20, 20, -10]],
                   [[-60, 60, -30]]]
    for x, y in itertools.product([0, 1], repeat=2):
        output = feed_forward(xor_network, [x, y])[-1]
        print(f"{x}, {y} -> {output}")


# Backpropagation
def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, targets)]

    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in hidden_outputs])
                     for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, inp in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * inp


if __name__ == '__main__':
    example_and_gate()
    example_or_gate()
    example_xor_gate()
