import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading in training set
ts_df=pd.read_csv('mr_gd training set.csv')

def gradient_descent(
        df, feature_column_list, target_column, 
        w_array, b, learning_rate, 
        tolerance, max_iterations, r_p
):
    """
    Performs gradient descent to find the optimal values of the w array and b.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the input and target variables.
        feature_column_list (list): List of feature column names in the dataframe.
        target_column (str): Name of the target variable column.
        w_array (numpy.ndarray): Initial weights in the form of a one dimensional NumpPy array.
        b (float): Initial bias.
        learning_rate (float): Learning rate for gradient descent.
        tolerance (float): Tolerance for stopping criteria.
        max_iterations (int): Maximum number of iterations.
        r_p (float): Regularisation parameter.
    
    Returns:
        (numpy.ndarray, float): The optimised values of the w array and b.
    """

    def mean_normalisation(feature_column_list):
        # Scales each feature in the feature list and replaces columns
        for i in feature_column_list:
            mu=df[i].mean()
            ran=df[i].max() - df[i].min()
            df[i] = (df[i] - mu) / ran

    mean_normalisation(feature_column_list)
    
    # Conversion of dataframe columns to NumPy arrays
    x_array = df[feature_column_list].values
    y_array = df[target_column].values

    def y_predictions(x_array, w_array, b):
        # Computes the value of the target predictions for each feature
        return np.dot(x_array, w_array) + b
    
    def compute_gradients(w_array, b, r_p):
        # Calculates the cost function gradient with respect to the weight array and b
        predictions = y_predictions(x_array, w_array, b)
        errors = predictions - y_array
        w_gradient = np.dot(errors, x_array) / len(x_array)
        w_gradient += r_p / len(x_array) * w_array
        b_gradient = errors.mean()
        return w_gradient, b_gradient
    
    iteration = 0
    while iteration < max_iterations:
        # Calculates the cost gradients for the current weight array and b values.
        w_gradient, b_gradient = compute_gradients(w_array, b, r_p)

        # Updates the weight array and b value using calculated gradients.
        w_array_new = w_array - learning_rate * w_gradient
        b_new = b - learning_rate * b_gradient

        # Checks for convergence
        if np.linalg.norm(w_array_new - w_array) < tolerance and abs(b_new - b) < tolerance:
            break
        
        # Updates the weight array and b for the next iteration.
        w_array, b = w_array_new, b_new
        iteration += 1

    return(w_array, b)


# Example input arguments
list=['Feature_1', 'Feature_2', 'Feature_3']
w_1 = np.array([0, 0, 0])

# Running the model to find optimum w values for each feature and the value of b
gradient_descent(ts_df, list, 'Target', w_1, 0, 0.01, 1e-10, 10000, 0.005)