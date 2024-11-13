import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
torch.manual_seed(7)

def create_data(weight=0.8, bias=0.2, start=0, end=1, step=0.01):
    """Generates linear data with known weight and bias."""
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    return X, y

def plot_data(x_values, y_values):
    """Plots x and y data points."""
    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values, s=3)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Plot of Data Points")
    plt.show()

# Prepare data
X, y = create_data()
plot_data(X, y)

# Split the data
def split_data(X, y, train_ratio=0.85):
    """Splits data into training and testing sets based on train_ratio."""
    train_split = int(train_ratio * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_data(X, y)

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """Plots training, testing data, and model predictions if provided."""
    plt.figure(figsize=(6, 3))
    plt.scatter(train_data, train_labels, c="g", s=4, label="Training Data", alpha=0.5)
    plt.scatter(test_data, test_labels, c="y", s=4, label="Testing Data", alpha=0.5)
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Model Predictions", alpha=0.5)
    plt.legend()
    plt.xlabel("Features")
    plt.ylabel("Labels")
    plt.title("Train, Test and Predictions")
    plt.show()

plot_predictions(X_train, y_train, X_test, y_test)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.weight * x + self.bias

LR_model = LinearRegressionModel()

# Define optimizer and loss function
def create_optimizer_and_loss(model, lr=0.001):
    """Creates optimizer and loss function for training the model."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    return optimizer, loss_fn

optimizer, loss_fn = create_optimizer_and_loss(LR_model)

# Training function
def train_model(model, optimizer, loss_fn, X_train, y_train, X_test, y_test, epochs=100):
    """Trains the model and returns lists of train and test losses."""
    train_losses, test_losses, epoch_count = [], [], []
    for epoch in range(epochs):
        model.train()
        y_pred_train = model(X_train)
        loss_train = loss_fn(y_pred_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.inference_mode():
            y_pred_test = model(X_test)
            loss_test = loss_fn(y_pred_test, y_test)
        
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_losses.append(loss_train.item())
            test_losses.append(loss_test.item())
            print(f"Epoch {epoch}: Train Loss: {loss_train.item()}, Test Loss: {loss_test.item()}")
    
    return train_losses, test_losses, epoch_count

train_losses, test_losses, epoch_count = train_model(LR_model, optimizer, loss_fn, X_train, y_train, X_test, y_test)

# Plot loss curves
def plot_loss_curves(epoch_count, train_losses, test_losses):
    """Plots training and testing loss curves."""
    plt.plot(epoch_count, train_losses, label="Train Loss")
    plt.plot(epoch_count, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.show()

plot_loss_curves(epoch_count, train_losses, test_losses)

# Make predictions
def make_predictions(model, X_test):
    """Makes predictions using the trained model."""
    model.eval()
    with torch.inference_mode():
        predictions = model(X_test)
    return predictions

y_preds = make_predictions(LR_model, X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Save model
def save_model(model, path='LR_model_save.pth'):
    """Saves the model's state_dict to the specified path."""
    torch.save(model.state_dict(), path)

save_model(LR_model)

# Load model and verify predictions
def load_and_verify_model(path='LR_model_save.pth'):
    """Loads a model from a saved state_dict and verifies predictions match."""
    model = LinearRegressionModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.inference_mode():
        loaded_model_preds = model(X_test)
    return loaded_model_preds

loaded_model_preds = load_and_verify_model()

# Check if loaded model predictions match original
print(torch.allclose(y_preds, loaded_model_preds))

