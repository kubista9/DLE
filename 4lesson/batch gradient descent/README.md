# Batch Gradient Descent for Linear Regression

## Overview
This script implements **Batch Gradient Descent (BGD)** to fit a simple linear regression model to a synthetic dataset.  
Batch Gradient Descent updates the model parameters using **all training samples** in each iteration.

## How It Works
1. **Generate synthetic data**: 
   - \( y = 2x + 3 \) with some added noise.
2. **Initialize parameters**:
   - Slope `m` and intercept `b` are initialized to zero.
3. **Iteratively update the parameters**:
   - Compute predictions for all data points.
   - Calculate the gradient of the loss function.
   - Update the parameters using the learning rate.
4. **Plot the results**.

## Installation
Make sure you have Python installed along with NumPy and Matplotlib.  
To install dependencies, run:

```sh
pip install numpy matplotlib