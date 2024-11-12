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


# **Calculate the split point for training and testing sets:**
train_split = int(0.85 * len(X))  # 85% of data will be used for training, 
                                  # 15% for testing

# **Split the data into training and testing sets:**
# - X_train: Features for the training set
# - y_train: Labels for the training set
# - X_test: Features for the testing set
# - y_test: Labels for the testing set
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# **Verify the lengths of the split sets for consistency:**
print(len(X_train), len(y_train), len(X_test), len(y_test))
# 85 85 15 15


def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Visualizes the training data, test data, and optionally model predictions for model performance.

  Args:
    train_data: training features.
    train_labels: training labels.
    test_data: testing features.
    test_labels: testing labels.
    predictions: model predictions, if available.
  """

  plt.figure(figsize=(6, 3))  # Create a clear and sizable plot

  # Plot the training data in green
  plt.scatter(train_data, train_labels, c="g", s=4, label="Training Data", alpha=0.5)

  # Plot the test data in yellow
  plt.scatter(test_data, test_labels, c="y", s=4, label="Testing Data", alpha=0.5)

  if predictions is not None:
    # Plot the model's predictions in red
    plt.scatter(test_data, predictions, c="r", s=4, label="Model Predictions", alpha=0.5)

  plt.legend(fontsize=14)

  plt.xlabel("Features")  # Label the x-axis for context
  plt.ylabel("Labels")  # Label the y-axis for clarity

  plt.title("Train Test and Predictions")  # Add a descriptive title

  plt.show()  # Display the plot for analysis

plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test) 

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the learnable parameters: weight (m) and bias (c)
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        """
        Forward pass function that computes the predicted y values.
        
        Args:
        - x: Input features (independent variable)
        
        Returns: Predicted output
        """
        # Linear regression formula: y = mx + c
        return self.weight * x + self.bias

# Set a manual seed to ensure reproducibility since nn.Parameters are randomly initialized
torch.manual_seed(7)

# Create an instance of our Linear Regression model
LR_model = LinearRegressionModel()

# Display the nn.Parameters within the nn.Module subclass we've just created
# These parameters are randomly initialized and will be adjusted during training to improve the model
print("Learnable Parameters:")
for parameter in LR_model.parameters():
    print(parameter)


# Retrieve and display the named parameters from the model's state dictionary
named_parameters = LR_model.state_dict()
print("Named Parameters:")
for name, parameter in named_parameters.items():
    print(f"{name}: {parameter.shape}")

# Generating Predictions using the Model
with torch.inference_mode():
    # Utilizing the model to predict y values for the test data, X_test
    y_preds = LR_model(X_test)

# Visualizing Predictions for the Straight Line Model
plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)

# Stochastic Gradient Descent (SGD) Optimizer
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam Optimizer
optimizer_adam = optim.Adam([var1, var2], lr=0.0001)

# Define the Loss Function
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=LR_model.parameters(), lr=0.001)

    # Set a manual seed to ensure reproducibility since nn.Parameters are randomly initialized
    torch.manual_seed(10)

    # Set the number of epochs (how many times the model will pass over the training data)
    epochs = 100

    # Create empty lists to track loss values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    # Training loop
    for epoch in range(epochs):
        # Training Phase
        LR_model.train()  # Put model in training mode

        # Forward pass on training data
        y_pred_train = LR_model(X_train)

        # Calculate the Mean Absolute Error (MAE) loss on training data
        loss_train = loss_fn(y_pred_train, y_train)

        # Zero gradient of the optimizer
        optimizer.zero_grad()

        # Backward pass: Compute gradients and update weights
        loss_train.backward()
        optimizer.step()

        # Testing Phase
        LR_model.eval()  # Put model in evaluation mode

        with torch.inference_mode():
            # Forward pass on test data
            y_pred_test = LR_model(X_test)

            # Calculate MAE loss on test data
            # Predictions are torch.float, so comparisons need the same datatype
            loss_test = loss_fn(y_pred_test, y_test.type(torch.float))  

            # Print out progress
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss_train.item())
                test_loss_values.append(loss_test.item())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss_train.item()} | MAE Test Loss: {loss_test.item()} ")



# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

# 1. Set the model in evaluation mode
LR_model.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
    # 3. Ensure calculations are done with the model and data on the same device
    # In our case, assuming both model and data are on the CPU
    # Perform inference and make predictions
    y_preds = LR_model(X_test)

print(y_preds)

# Save the state_dict of the trained model
MODEL_SAVE_PATH = 'LR_model_save.pth'
torch.save(obj=LR_model.state_dict(), f=MODEL_SAVE_PATH)

# Instantiate a new instance of our model (initialized with random weights)
loaded_model_LR = LinearRegressionModel()

# Load the state_dict of the saved model to update the new instance with trained weights
loaded_model_LR.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_LR.eval()

# Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_LR(X_test)  # Perform a forward pass on the test data with the loaded model

# Check if predictions from the loaded model are equal to the original predictions
print(y_preds == loaded_model_preds)


