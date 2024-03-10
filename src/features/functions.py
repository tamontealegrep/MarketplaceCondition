
import pandas as pd
import numpy as np

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

#---------------------------------------------------------------------------------------------------