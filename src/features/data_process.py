
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import yeojohnson

#---------------------------------------------------------------------------------------------------

class WeekScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to scale weeks in the year to range 0 - 1.

    Args:
        - None

    Attributes:
        - range_min (float, default=0): Minimum value of the desired range.
        - range_max (float, default=51):  Maximum value of the desired range.   
    """
    def __init__(self):
        self.range_min = 1
        self.range_max = 52

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_scaled = (X - self.range_min) / (self.range_max - self.range_min)
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class WeekdayScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to scale days in the week to range 0 - 1.

    Args:
        - None

    Attributes:
        - range_min (float, default=0): Minimum value of the desired range.
        - range_max (float, default=51):  Maximum value of the desired range.   
    """
    def __init__(self):
        self.range_min = 0
        self.range_max = 6

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_scaled = (X - self.range_min) / (self.range_max - self.range_min)
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
class OneHotScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle one-hot encoded variables.

    This transformer does not perform any scaling and simply returns the input data unchanged.

    Args:
        - None

    Attributes:
        - None
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
class DataScaler():
    """
    Class to scale new data using a predefined transformation pipeline.

    Args:
        None

    Attributes:
        columns_to_standardize (list): List of columns to be standardized.
        columns_to_minmax_scale (list): List of columns to be min-max scaled.
        columns_to_week_scale (list): List of columns to be week scaled.
        columns_to_weekday_scale (list): List of columns to be weekday scaled.
        columns_to_one_hot_scale (list): List of columns to be one-hot encoded.
        ct (ColumnTransformer): ColumnTransformer object for data scaling.
        new_order (list): New order of columns after scaling.
        old_order (list): Original order of columns in the dataset.
    """

    def __init__(self):
        self.columns_to_standardize = ['price', 'initial_quantity', 'sold_quantity', 'available_quantity']
        self.columns_to_minmax_scale = ['num_pictures']
        self.columns_to_week_scale = ['start_week', 'stop_week']
        self.columns_to_weekday_scale = ['start_day', 'stop_day']
        self.columns_to_one_hot_scale = [ 'warranty_info', 'mode_buy_it_now',
                                          'cash_payment', 'card_payment',
                                          'bank_payment', 'mercadopago_payment', 'agree_with_buyer_payment',
                                          'shipping_me2', 'shipping_custom',
                                          'shipping_not_specified', 'free_shipping', 'local_pick_up',
                                          'buenos_aires_seller', 'listing_free', 'listing_bronze',
                                          'listing_silver', 'listing_gold', 'is_active',
                                          'automatic_relist', 'dragged_bids_or_visits']
        self.ct = ColumnTransformer(
            [('std', StandardScaler(), self.columns_to_standardize),
             ('minmax', MinMaxScaler(), self.columns_to_minmax_scale),
             ('week', WeekScaler(), self.columns_to_week_scale),
             ('weekday', WeekdayScaler(), self.columns_to_weekday_scale),
             ('one_hot', OneHotScaler(), self.columns_to_one_hot_scale)
             ],
            remainder='passthrough')
        self.new_order = self.columns_to_standardize + self.columns_to_minmax_scale + \
                         self.columns_to_week_scale + self.columns_to_weekday_scale + \
                         self.columns_to_one_hot_scale
        self.old_order = None

    def fit_transform(self, X_train):
        """
        Fit and transform the input data using the predefined transformation pipeline.

        Parameters:
            - X_train (DataFrame): Input data to be scaled.

        Returns:
            - X_train_scaled (DataFrame): Scaled data.
        """
        self.old_order = X_train.columns
        X_train_scaled = self.ct.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.new_order)
        X_train_scaled = X_train_scaled.reindex(columns=self.old_order)
        return X_train_scaled
    
    def transform(self, X):
        """
        Transform new data using the predefined transformation pipeline.

        Parameters:
            - X (DataFrame): New data to be scaled.

        Returns:
            - X_scaled (DataFrame): Scaled data.
        """
        X_scaled = self.ct.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.new_order)
        X_scaled = X_scaled.reindex(columns=self.old_order)
        return X_scaled
    
def process_dataset(X_train, y_train, X_test, y_test):
    """
    Preprocesses the dataset by performing the following steps:
    1. Removes specified columns with low information content.
    2. Removes rows with outlier values beyond predefined thresholds.
    3. Applies Yeojohnson transformation to specified numeric features.
    4. Splits the dataset into features (X) and target (y) variables.
    5. Scales the features using a DataScaler instance.

    Args:
    - X_train (pd.DataFrame): Training set features.
    - y_train (pd.DataFrame): Training set target variable.
    - X_test (pd.DataFrame): Test set features.
    - y_test (pd.DataFrame): Test set target variable.

    Returns:
    - X_train_scaled (np.ndarray): Scaled training set features.
    - y_train_scaled (pd.DataFrame): Training set target variable.
    - X_test_scaled (np.ndarray): Scaled test set features.
    - y_test_scaled (pd.DataFrame): Test set target variable.
    """
    COLUMNS_TO_REMOVE = ["price_in_usd", "mode_classified", "mode_auction", "shipping_me1", "days_active"]
    PRICE_THRESHOLD = 50000
    INITIAL_QUANTITY_THRESHOLD = 150
    SOLD_QUANTITY_THRESHOLD = 150
    AVAILABLE_QUANTITY_THRESHOLD = 150
    NUM_PICTURES_THRESHOLD = 10

    YJ_PRICE = -0.1326933439177492
    YJ_INITIAL_QUANTITY = -1.8163258668093907
    YJ_SOLD_QUANTITY = -3.65996684652743
    YJ_AVAILABLE_QUANTITY = -1.8854981114246059

    df_train = pd.concat([X_train, y_train], axis=1).copy(deep=True)
    df_test = pd.concat([X_test, y_test], axis=1).copy(deep=True)

    df_list = [df_train, df_test]

    for i in df_list:
        for column in COLUMNS_TO_REMOVE:
            i.drop(columns=[column], inplace=True)

        i.drop(i[i['price'] > PRICE_THRESHOLD].index, inplace=True)
        i.drop(i[i['initial_quantity'] > INITIAL_QUANTITY_THRESHOLD].index, inplace=True)
        i.drop(i[i['sold_quantity'] > SOLD_QUANTITY_THRESHOLD].index, inplace=True)
        i.drop(i[i['available_quantity'] > AVAILABLE_QUANTITY_THRESHOLD].index, inplace=True)
        i.drop(i[i['num_pictures'] > NUM_PICTURES_THRESHOLD].index, inplace=True)

    for i in range(len(df_list)):
        df_list[i]['price'] = yeojohnson(df_list[i]['price'],YJ_PRICE)
        df_list[i]['initial_quantity'] = yeojohnson(df_list[i]['initial_quantity'],YJ_INITIAL_QUANTITY)
        df_list[i]['sold_quantity'] = yeojohnson(df_list[i]['sold_quantity'],YJ_SOLD_QUANTITY)
        df_list[i]['available_quantity'] = yeojohnson(df_list[i]['available_quantity'],YJ_AVAILABLE_QUANTITY)

    
    X_train_scaled = df_train.drop(columns=['is_new']).copy(deep=True)
    y_train_scaled = df_train[['is_new']].copy(deep=True)
    X_test_scaled = df_test.drop(columns=['is_new']).copy(deep=True)
    y_test_scaled = df_test[['is_new']].copy(deep=True)

    scaler = DataScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_scaled)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
#---------------------------------------------------------------------------------------------------