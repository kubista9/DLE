# Simple Hyperplane Visualization

## verview

This script demonstrates the concept of a hyperplane as a decision boundary for classifying data points into two different classes. It uses Python and matplotlib to generate a simple dataset and visualize how a hyperplane separates two groups.

## How It Works

1. **Generate synthetic data**:
    - Create two classes of data points using numpy.

2. **Define the hyperplane**:

    - Use a simple linear equation  as the decision boundary.

3. **Plot the dataset**:

Visualize the two classes and the hyperplane.

## Expected Output

A scatter plot showing:

🔵 Blue points representing Class 1.

🔴 Red points representing Class 2.

⚫ Dashed black line (the hyperplane) separating the two classes.

## Explanation
    - The hyperplane is defined by the equation:  (or ).
    - Points on one side of the hyperplane belong to Class 1.
    - Points on the other side belong to Class 2.
    - The dot product method is used to determine which side a point falls on.

```sh
pip install numpy matplotlib
