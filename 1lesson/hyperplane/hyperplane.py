import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset with two classes
np.random.seed(0)
class1 = np.random.randn(50, 2) + np.array([2, 2])
class2 = np.random.randn(50, 2) + np.array([-2, -2])

# Combine them into X, with labels y
X = np.vstack((class1, class2))
y = np.hstack((np.ones(50), -1 * np.ones(50)))

# Define a simple hyperplane: theta values
theta = np.array([1, -1])   # means line equation: x - y = 0 or y = x
bias = 0

# Plot data points
plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Class 2')

# Plot hyperplane line: (theta[0] * x + theta[1] * y + bias = 0)
x_vals = np.linspace(-5, 5, 100)
y_vals = (-bias - theta[0] * x_vals) / theta[1]
plt.plot(x_vals, y_vals, 'k--', label='Hyperplane (y = x)')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.title('Simple Hyperplane Example')
plt.show()
