import pandas as pd
import numpy as np
import math
from scipy.special import expit

# Reads in the training set and saving a copy for later normalisation
ts_df = pd.read_csv('logistic_regression_training_set.csv')
ts_df_to_norm = pd.read_csv('logistic_regression_training_set.csv')

def gradient_descent(
        df, feature_column_list, target_column, 
        weights, bias, learning_rate, 
        tolerance, max_iterations, r_p
):
    """
    Performs gradient descent to find the optimal values of the weight array and bias for a logistic regression classification algorithm.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the input and target variables.
        feature_column_list (list): List of feature column names in the dataframe.
        target_column (str): Name of the target variable column.
        weights (numpy.ndarray): Initial weights in the form of a one dimensional NumpPy array.
        bias (float): Initial bias.
        learning_rate (float): Learning rate for gradient descent.
        tolerance (float): Tolerance for stopping criteria.
        max_iterations (int): Maximum number of iterations.
        r_p (float): Regularisation parameter.
    
    Returns:
        (numpy.ndarray, float): The optimised values of the weight array and bias.
    """

    def y_prediction(x_array, weights, bias):
        # Computes the classification of the target using each feature, weight, and the bias
        prob = expit(np.dot(x_array, weights) + bias)
        return np.where(prob >= 0.5, 1, 0)
    
    def mean_normalisation(feature_column_list):
        """
        Scales each feature in the feature list using mean normalisation and modifies the original DataFrame in place.
    
        Parameters:
            df (pd.DataFrame): DataFrame containing the input variables to be normalised. This DataFrame will be modified in place.
            feature_column_list (list): List of feature column names in the dataframe.
        """
        for i in feature_column_list:
            mu=df[i].mean()
            ran=df[i].max() - df[i].min()
            # Modifies the original DataFrame in place
            df[i] = (df[i] - mu) / ran

    mean_normalisation(feature_column_list)
    
    # Creates the vectors for the features and their corresponding target values
    x_array = df[feature_column_list].values
    y_array = df[target_column].values

    def compute_gradients(weights, bias, r_p):
        # Calculates the cost function gradient with respect to the weight array and bias
        predictions = y_prediction(x_array, weights, bias)
        errors = predictions - y_array
        w_gradient = np.dot(errors, x_array) / len(x_array)
        w_gradient += r_p / len(x_array) * weights
        b_gradient = errors.mean()
        return w_gradient, b_gradient
    
    iteration = 0
    while iteration < max_iterations:
        # Calculates the cost gradients for the current weight array and bias values
        w_gradient, b_gradient = compute_gradients(weights, bias, r_p)

        # Updates the weight array and bias value using calculated gradients
        weights_new = weights - learning_rate * w_gradient
        b_new = bias - learning_rate * b_gradient

        # Checks for convergence
        if np.linalg.norm(weights_new - weights) < tolerance and abs(b_new - bias) < tolerance:
            break
        
        # Updates the weight array and bias for the next iteration
        weights, bias = weights_new, b_new
        iteration += 1

    return(weights, bias)


# Capturing feature list and initialising weights for gradient descent
lst = ['Age', 'Income', 'Education Level', 'Years of Experience', 'Has Mortgage']
initial_weights = np.array([0, 0, 0, 0, 0])

# Example usage of the gradient descent
weights, bias = gradient_descent(ts_df, lst, 'Outcome', initial_weights, 0, 0.01, 1e-10, 100000, 0.0005)


def new_example_prediction(x_list, weights, bias, df):
    # Calculates the means and ranges for each column in the dataframe
    means = df.iloc[:, :-1].mean()
    ranges = df.iloc[:, :-1].max() - df.iloc[:, :-1].min()

    # Performs mean normalisation on the new data using the training set
    x_array = (x_list - means) / ranges

    # Calcuates probability and gives an output according to the decision boundary
    prob = expit(np.dot(x_array, weights) + bias)
    return np.where(prob >= 0.5, 1, 0)


# Example usage of the new data prediction using the gradient descent output
x_list = [26,90000,6,2,1]
new_example_prediction(x_list, weights, bias, ts_df_to_norm)