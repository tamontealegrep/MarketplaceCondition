
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

#---------------------------------------------------------------------------------------------------

def import_data():
    """
    Import data for training and testing.

    Args:

    Returns:
    - X_train (DataFrame): Training data features.
    - y_train (DataFrame): Training data labels.
    - X_test (DataFrame): Testing data features.
    - y_test (DataFrame): Testing data labels.
    """

    X_train = pd.read_csv(f"../data/processed/train_data.csv")
    y_train = pd.read_csv(f"../data/processed/train_target.csv")

    X_test = pd.read_csv(f"../data/processed/test_data.csv")
    y_test = pd.read_csv(f"../data/processed/test_target.csv")

    return X_train, y_train, X_test, y_test

def percentile_range_data(column, min_percentile=0, max_percentile=100):
    """
    Returns the data within the specified percentile range for a given column.

    Parameters:
        column (pandas.Series): The DataFrame column from which data will be extracted.
        min_percentile (int, optional): The minimum percentile value (inclusive). Default is 0.
        max_percentile (int, optional): The maximum percentile value (inclusive). Default is 100.

    Returns:
        pandas.Series: A Series containing the data within the specified percentile range.
    """
    # Check if min_percentile is greater than max_percentile
    if min_percentile > max_percentile:
        raise ValueError("min_percentile cannot be greater than max_percentile.")
    
    # Calculate the values of the percentiles
    min_value = column.quantile(min_percentile / 100)
    max_value = column.quantile(max_percentile / 100)
    
    # Filter the data within the percentile range
    filtered_data = column[(column >= min_value) & (column <= max_value)]
    
    return filtered_data

def find_best_threshold(fpr, tpr, thresholds, y_test, y_prob):
    """
    Find the best threshold on ROC curve.

    Args:
        fpr (array-like): Array containing the false positive rates.
        tpr (array-like): Array containing the true positive rates.
        thresholds (array-like): Array containing the thresholds.
        y_test (array-like): Array containing true labels.
        y_prob (array-like): Array containing predicted probabilities.

    Returns:
        float: Best threshold and corresponding maximizing accuracy.
    """
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = np.where(y_prob >= best_threshold, 1, 0)
    y_test = np.squeeze(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    return best_threshold, accuracy

def optimize_threshold_for_accuracy(y_test, y_prob):
    """
    Find the optimal threshold to maximize accuracy.

    Parameters:
        y_test (array-like): True labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        tuple: Optimal threshold for maximizing accuracy and corresponding accuracy.
    """
    thresholds = np.linspace(0, 1, 100)  
    accuracies = []
    for th in thresholds:
        y_pred = np.where(y_prob >= th, 1, 0)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    optimal_accuracy_index = np.argmax(accuracies)
    optimal_threshold = thresholds[optimal_accuracy_index]
    optimal_accuracy = accuracies[optimal_accuracy_index]
    return optimal_threshold, optimal_accuracy

def npv_score(y_true, y_pred):
    """
    Compute the Negative Predictive Value (NPV).

    NPV measures the proportion of instances classified correctly as negative 
    among all instances classified as negative.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - npv: float
        Negative Predictive Value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fp)
    return npv

def specificity_score(y_true, y_pred):
    """
    Compute the specificity.

    Specificity measures the proportion of negative instances that were correctly identified as negative.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns:
    - specificity: float
        Specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def predict_classification(model, X_test, y_test, optimal_threshold=0.5, print_results=False):
    """
    Predict labels using a classification model and optionally print evaluation metrics.

    Args:
        model (object): The trained classification model.
        X_test (array-like): Test features.
        y_test (array-like): True labels for the test set.
        optimal_threshold (float, optional): Threshold for binary classification. Default is 0.5.
        print_results (bool, optional): Whether to print evaluation metrics. Default is False.

    Returns:
        y_pred (array-like): Predicted labels.

    Prints:
        If print_results is True, evaluation metrics including Accuracy, F1 score, Precision, NVP, Sensitivity, and Specificity.
    """
    # Get the predicted probabilities
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Apply the optimal threshold for binary classification
    y_pred = np.where(y_prob_test >= optimal_threshold, 1, 0)

    if print_results:
        # Calculate evaluation metrics
        accuracy_value = accuracy_score(y_test, y_pred)
        f1_score_value = f1_score(y_test, y_pred)
        precision_value = precision_score(y_test, y_pred)
        npv_value = npv_score(y_test, y_pred)
        sensitivity_value = recall_score(y_test, y_pred)
        specificity_value = specificity_score(y_test, y_pred)

        # Format the output for readability
        results = {
            "Accuracy": accuracy_value,
            "F1 score": f1_score_value,
            "Precision": precision_value,
            "NVP": npv_value,
            "Sensitivity": sensitivity_value,
            "Specificity": specificity_value
        }
        
        # Print the results
        print("Evaluation metrics:")
        for metric, value in results.items():
            print(f"{metric:<12}:\t{value:.3f}")

    return y_pred
#---------------------------------------------------------------------------------------------------