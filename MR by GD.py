import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading in training set
ts_df=pd.read_csv('mr_gd training set.csv')

def gradient_descent(df,
                    feature_column_list, 
                    target_column, 
                    w_array, 
                    b, learning_rate, 
                    tolerance, 
                    max_iterations
):
    
    x_array = df[feature_column_list].values
    y_array = df[target_column].values

    def y_predictions(x_array, w_array, b):
        return np.dot(x_array, w_array) + b
    
    def compute_gradients(w_array, b):
        predictions = y_predictions(x_array, w_array, b)
        errors = predictions - y_array
        w_gradient = np.dot(errors, x_array)
        w_gradient.mean(axis=0)      # Issue was likely on lines 27 & 28.
        b_gradient = errors.mean()
        return w_gradient, b_gradient
    
    iteration = 0
    while iteration < max_iterations:
        w_gradient, b_gradient = compute_gradients(w_array, b)

        w_array_new = w_array - learning_rate * w_gradient
        b_new = b - learning_rate * b_gradient

        if np.linalg.norm(w_array_new - w_array) < tolerance and abs(b_new - b) < tolerance:
            break

        w_array, b = w_array_new, b_new
        iteration += 1

    return(w_array, b)


list=['Feature_1', 'Feature_2', 'Feature_3']
w_1 = np.array([0, 0, 0])
gradient_descent(ts_df, list, 'Target', w_1, 0, 0.001, 1e-10, 100000)