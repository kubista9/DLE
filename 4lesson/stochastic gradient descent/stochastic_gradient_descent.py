# Reset parameters
m, b = 0, 0  
learning_rate = 0.01
epochs = 1000

# Stochastic Gradient Descent
for _ in range(epochs):
    for i in range(n):  # Loop through each data point
        xi = X[i]
        yi = y[i]

        # Compute prediction and error
        y_pred = m * xi + b
        error = y_pred - yi

        # Compute gradients for a single data point
        dm = 2 * xi * error
        db = 2 * error

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

# Plot results
plt.scatter(X, y, label="Data")
plt.plot(X, m * X + b, color='green', label="SGD Model")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Stochastic Gradient Descent")
plt.show()

print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")
