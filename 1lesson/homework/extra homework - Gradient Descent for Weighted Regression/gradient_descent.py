def weighted_gradient_descent(X, y, w, lr=0.01, epochs=1000):
    theta = np.zeros(X.shape[1])  # Initialize
    W = np.diag(w)
    for _ in range(epochs):
        error = X @ theta - y
        gradient = X.T @ W @ error  # Weighted gradient
        theta -= lr * gradient
    return theta

theta_gd = weighted_gradient_descent(X, y, w)
print("Gradient descent theta:", theta_gd)