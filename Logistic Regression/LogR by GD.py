import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.special import expit

# Reading in the training set
ts_df = pd.read_csv('logistic_regression_training_set.csv')

def gradient_descent(
        df, feature_column_list, target_column, 
        w_array, b, learning_rate, 
        tolerance, max_iterations
):

    def y_prediction(x_array, w_array, b):
        prob = expit(np.dot(x_array, w_array) + b)
        return np.where(prob >= 0.5, 1, 0)
    
    def mean_normalisation(feature_column_list):
        # Scales each feature in the feature list and replaces columns
        for i in feature_column_list:
            mu=df[i].mean()
            ran=df[i].max() - df[i].min()
            df[i] = (df[i] - mu) / ran

    mean_normalisation(feature_column_list)
    
    x_array = df[feature_column_list].values
    y_array = df[target_column].values

    def compute_gradients(w_array, b):
        # Calculates the cost function gradient with respect to the weight array and b
        predictions = y_prediction(x_array, w_array, b)
        errors = predictions - y_array
        w_gradient = np.dot(errors, x_array) / len(x_array)
        # w_gradient += r_p / len(x_array) * w_array
        b_gradient = errors.mean()
        return w_gradient, b_gradient
    
    iteration = 0
    while iteration < max_iterations:
        # Calculates the cost gradients for the current weight array and b values.
        w_gradient, b_gradient = compute_gradients(w_array, b)

        # Updates the weight array and b value using calculated gradients.
        w_array_new = w_array - learning_rate * w_gradient
        b_new = b - learning_rate * b_gradient

        # Checks for convergence
        if np.linalg.norm(w_array_new - w_array) < tolerance and abs(b_new - b) < tolerance:
            break
        
        # Updates the weight array and b for the next iteration.
        w_array, b = w_array_new, b_new
        x_test = np.array([69,78984,2,31,1])
        iteration += 1

    return(w_array, b)


lst = ['Age', 'Income', 'Education Level', 'Years of Experience', 'Has Mortgage']
w = np.array([0, 0, 0, 0, 0])

weights, bias = gradient_descent(ts_df, lst, 'Outcome', w, 0, 0.01, 1e-10, 10000)

def new_example(x_array, weights, bias):
        prob = expit(np.dot(x_array, weights) + bias)
        return np.where(prob >= 0.5, 1, 0)