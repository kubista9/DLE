import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (y = 2x + 3 with noise)
np.random.seed(42)
X = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y = 2 * X + 3 + np.random.randn(100) * 2  # Add some noise

# Initialize parameters
m, b = 0, 0  # Initial values for slope and intercept
learning_rate = 0.01
epochs = 1000
n = len(X)  # Number of training examples

# Batch Gradient Descent
for _ in range(epochs):
    y_pred = m * X + b  # Compute predictions
    error = y_pred - y  # Compute error

    # Compute gradients
    dm = (2 / n) * np.sum(X * error)
    db = (2 / n) * np.sum(error)

    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db

# Plot resultsj
plt.scatter(X, y, label="Data")
plt.plot(X, m * X + b, color='red', label="Batch GD Model")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Batch Gradient Descent")
plt.show()

print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")
