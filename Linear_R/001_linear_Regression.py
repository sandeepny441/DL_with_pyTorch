import torch
from torch import nn  # neural network building blocks
import matplotlib.pyplot as plt  # To visualize our datapoints

# Let's see what version of PyTorch we're working with
print(torch.__version__)

# Create *known* parameters
weight = 0.8
bias = 0.2

# Create data
start = 0
end = 1
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
'''
X -  tensor([[0.0000],
        [0.0100],
        [0.0200],
        [0.0300],
        [0.0400],
        [0.0500],
        [0.0600],
        [0.0700],
        [0.0800],
        [0.0900]]) 
 y -  tensor([[0.2000],
        [0.2080],
        [0.2160],
        [0.2240],
        [0.2320],
        [0.2400],
        [0.2480],
        [0.2560],
        [0.2640],
        [0.2720]])
'''

# a simple script to plot x,y datapoints 
def plot_datapoints(x_values,y_values):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values,s=3)

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Plot of Data Points")
    
    # Display the plot
    plt.show()
plot_datapoints(X,y)