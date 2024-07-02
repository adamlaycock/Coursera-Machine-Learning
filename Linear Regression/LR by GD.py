import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt#
import numpy as np

# Reading in the training set
ts_df=pd.read_csv('training set.csv')

def gradient_descent(df, input_var, target_var, w, b, learning_rate, tolerance, max_iterations, r_p):
    """
    Performs gradient descent to find the optimal values of w and b.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the input and target variables.
        input_var (str): Name of the input variable column.
        target_var (str): Name of the target variable column.
        w (float): Initial weight.
        b (float): Initial bias.
        learning_rate (float): Learning rate for gradient descent.
        tolerance (float): Tolerance for stopping criteria.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        (float, float): The optimized values of w and b.
    """
    
    def y_prediction(x_input, w, b):
        # Predicts a target value based on current w and b
        return w * x_input + b
    
    def compute_gradients(df, w, b, r_p):
        # Calculates gradient of cost function with respect to w and b
        y_pred = y_prediction(df[input_var], w, b)
        errors = y_pred - df[target_var]
        w_gradient = (errors * df[input_var]).mean() + r_p / len(df) * w
        b_gradient = errors.mean()
        return w_gradient, b_gradient
    
    iteration = 0
    while iteration <= max_iterations:
        # Calculates cost gradients for current w and b
        w_gradient, b_gradient = compute_gradients(df, w, b)

        # Updates w and b using calculated gradients
        w_new = w - learning_rate * w_gradient
        b_new = b - learning_rate * b_gradient

        # Checks for convergence at minima
        if abs(w_new - w) < tolerance and abs(b_new - b) < tolerance:
            break
        
        # Updates w and b for next iteration
        w, b = w_new, b_new
        iteration += 1

    return(w, b)

# Running model to find optimal w and b
w,b = gradient_descent(ts_df, 'x', 'y', 0, 0, 0.01, 1e-6, 1000)


# Creating straight line x and y coordinates using w and b
x_values = np.linspace(min(ts_df['x']), max(ts_df['x']), 100)
y_values = w * x_values + b

# Plotting the training set and line together
plt.figure()
plt.plot(x_values, 
         y_values, 
         color='red', 
         label=f'y = {w}x + {b}'
)
sns.scatterplot(x='x',
                y='y', 
                data=ts_df
)
plt.show()