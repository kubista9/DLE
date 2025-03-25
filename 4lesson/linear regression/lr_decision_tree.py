from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load data (in this case, Iris flower dataset)
iris = load_iris()
X = iris.data  # Features (sepal length, width, petal length, width)
y = iris.target  # Classes (setosa, versicolor, virginica)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize the model (decision tree classifier)
model = DecisionTreeClassifier()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")
