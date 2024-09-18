import numpy as np

# Define a simple neural network class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size)
    
    def forward(self, x):
        self.hidden = np.tanh(np.dot(x, self.hidden_weights))
        self.output = np.dot(self.hidden, self.output_weights)
        return self.output
    
    def train(self, x, y, learning_rate=0.01):
        output = self.forward(x)
        output_error = y - output
        hidden_error = np.dot(output_error, self.output_weights.T)
        
        self.output_weights += learning_rate * np.outer(self.hidden, output_error)
        self.hidden_weights += learning_rate * np.outer(x, hidden_error * (1 - np.tanh(self.hidden)**2))

# Create and train the network
input_size = 5
hidden_size = 3
output_size = 1

network = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# Training data: lists of 5 numbers and their sums
X_train = np.array([
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10],
    [1, 3, 5, 7, 9],
    [10, 20, 30, 40, 50]
])
y_train = np.array([[sum(x)] for x in X_train])

# Train the network
epochs = 10000
for _ in range(epochs):
    for x, y in zip(X_train, y_train):
        network.train(x, y)

# Test the network
test_list = np.array([5, 10, 15, 20, 25])
predicted_sum = network.forward(test_list)
actual_sum = np.sum(test_list)

print(f"Input list: {test_list}")
print(f"Predicted sum: {predicted_sum[0]:.2f}")
print(f"Actual sum: {actual_sum}")