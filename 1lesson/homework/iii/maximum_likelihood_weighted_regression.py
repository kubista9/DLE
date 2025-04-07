import numpy as np

# Example data
X = np.array([[1, 1], [1, 2], [1, 3]])  # Design matrix
y = np.array([2, 3, 4])                  # Targets
sigma = np.array([0.5, 1.0, 2.0])        # Standard deviations σ^(i)

# Weights: w^(i) = 1 / (σ^(i))^2
w = 1 / (sigma**2)

# Weight matrix W
W = np.diag(w)

# Solve weighted regression
XT_W_X = X.T @ W @ X
XT_W_y = X.T @ W @ y
theta_ml = np.linalg.inv(XT_W_X) @ XT_W_y

print("Standard deviations (σ):", sigma)
print("Weights (w = 1/σ^2):", w)
print("Maximum likelihood theta:", theta_ml)