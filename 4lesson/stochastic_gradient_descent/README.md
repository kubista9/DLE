# Stochastic Gradient Descent for Linear Regression

## Overview
This script implements **Stochastic Gradient Descent (SGD)** to fit a simple linear regression model to a synthetic dataset.  
Unlike **Batch Gradient Descent**, which updates parameters using the entire dataset, **SGD updates the parameters after processing each individual training example**.  

## How It Works
1. **Generate synthetic data**:
   - We create a dataset following the equation \( y = 2x + 3 \) with random noise.
2. **Initialize parameters**:
   - Set the slope `m` and intercept `b` to initial values (e.g., zero).
3. **Iterate through the dataset**:
   - Pick a single data point.
   - Compute the gradient and update parameters.
   - Repeat for multiple epochs.
4. **Plot the results**.

## Installation
Make sure you have Python installed along with the required libraries.  
To install dependencies, run:

```sh
pip install numpy matplotlib
