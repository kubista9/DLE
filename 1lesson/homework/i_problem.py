import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
w = np.array([0.5, 1.0, 1.5])  # Weights

# Weight matrix
W = np.diag(w)

# Parameters
theta = np.array([0.1, 0.2])

# Residuals
residuals = X.dot(theta) - y

# Cost in sum form
J_sum = 0.5 * sum(w[i] * residuals[i]**2 for i in range(3))

# Cost in matrix form
J_matrix = residuals.T.dot(W).dot(residuals)

print("J(θ) (sum form):", J_sum)
print("J(θ) (matrix form):", J_matrix)

# Visualization
plt.figure(figsize=(10, 4))
plt.bar(range(3), residuals**2, alpha=0.5, label="Squared residuals")
plt.bar(range(3), w * residuals**2, alpha=0.5, label="Weighted squared residuals")
plt.xticks([0, 1, 2], ["Example 1", "Example 2", "Example 3"])
plt.ylabel("Contribution to J(θ)")
plt.legend()
plt.title("Effect of Weights on Cost Function")
plt.show()