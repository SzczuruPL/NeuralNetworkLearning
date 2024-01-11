import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
outputs_expected = np.array([[0], [1], [1], [0]])

neurons_input = 3
neurons_hidden = 2
neurons_output = 1
epochs = 10000
learning_rate = 0.7
m = 0.8

hidden_weights_input = np.random.uniform(low=-1, high=1, size=(neurons_input, neurons_hidden))
hidden_weights_output = np.random.uniform(low=-1, high=1, size=(neurons_hidden, neurons_output))

hidden_weights_input_old = np.zeros_like(hidden_weights_input)
hidden_weights_output_old = np.zeros_like(hidden_weights_output)

for epoch in range(epochs):
    # 5
    hidden_input = np.dot(inputs, hidden_weights_input)
    hidden_output = sigmoid(hidden_input)

    # 6
    output_input = np.dot(hidden_output, hidden_weights_output)
    outputs_predicted = sigmoid(output_input)

    # 7
    output_error = (outputs_expected - outputs_predicted) * sigmoid_derivative(outputs_predicted)
    hidden_weights_output_update = np.dot(hidden_output.T, output_error) * learning_rate

    # 8
    hidden_error = output_error.dot(hidden_weights_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    hidden_weights_input_update = np.dot(inputs.T, hidden_delta) * learning_rate

    # 9
    hidden_weights_output += hidden_weights_output_update + m * hidden_weights_output_old
    hidden_weights_input += hidden_weights_input_update + m * hidden_weights_input_old

    hidden_weights_output_old = hidden_weights_output_update
    hidden_weights_input_old = hidden_weights_input_update

    if epoch == 0:
        print("\nPrzed uczeniem:")
        for i in range(len(inputs)):
            print(f"{outputs_expected[i]} - {outputs_predicted[i]}")

print("\nPo uczeniu:")
for i in range(len(inputs)):
    print(f"{outputs_expected[i]} - {outputs_predicted[i]}")