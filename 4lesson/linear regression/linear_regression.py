import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: Hours studied (X) vs Exam score (Y)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([52, 55, 60, 63, 65, 68, 72, 75, 78, 82])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, Y)

# Make predictions based on X
Y_pred = model.predict(X)

# Print the regression equation
print(f"Regression equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# Plot the data points and the regression line
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label=f'Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()