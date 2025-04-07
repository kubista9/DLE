import numpy as np

# Example data
X = np.array([[1, 1],  # x^(1): [1, 1]
              [1, 2],  # x^(2): [1, 2]
              [1, 3]]) # x^(3): [1, 3]
y = np.array([2, 3, 4])  # Target values

# Weights for each example (arbitrary choice for demonstration)
w = np.array([1.0, 2.0, 0.5])  # w^(1)=1, w^(2)=2, w^(3)=0.5

# Step 1: Construct the diagonal weight matrix W
W = np.diag(w)

# Step 2: Compute X^T W X
XT_W_X = X.T @ W @ X  # Equivalent to np.dot(X.T, np.dot(W, X))

# Step 3: Compute X^T W y
XT_W_y = X.T @ W @ y

# Step 4: Solve for theta using the weighted normal equation
theta = np.linalg.inv(XT_W_X) @ XT_W_y

print("Weights (w):", w)
print("Design matrix (X):\n", X)
print("Weight matrix (W):\n", W)
print("X^T W X:\n", XT_W_X)
print("X^T W y:\n", XT_W_y)
print("Optimal theta (closed-form solution):", theta)