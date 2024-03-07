
import pandas as pd

#---------------------------------------------------------------------------------------------------

def categorize_data_types(df):
    """
    Categorize data types present in each column of the DataFrame into 'basic_types' and 'others'.

    Args:
    - df: pandas DataFrame containing the data.

    Returns:
    - pandas DataFrame: A DataFrame where each row represents a column of the original DataFrame.
                        It contains two columns: 'basic_types' and 'others', which record the data types found in each column.
    """
    # Create an empty DataFrame to store the results
    types_df = pd.DataFrame(columns=['basic_types', 'others'])

    # Iterate over each column of the original DataFrame
    for column in df.columns:
        # Get unique data types present in the column
        types = set(df[column].apply(lambda x: type(x).__name__))
        
        # Categorize types into 'basic_types' and 'others'
        basic_types = {t for t in types if t in ['bool','int', 'float', 'str', 'list', 'tuple', 'dict', 'NoneType']}
        other_types = types - basic_types
        
        # Add data types to the corresponding row in the result DataFrame
        types_df.loc[column] = [', '.join(basic_types), ', '.join(other_types)]

    return types_df

def column_unique_values(column):
    """
    Return a list of unique values present in the specified column, regardless of their data type.

    Args:
    - column: pandas Series or list containing elements of any data type.

    Returns:
    - list: A list containing unique values present in the column.
    """
    unique_values = set()
    for item in column:
        # Convert the object to a string before adding it to the set
        unique_values.add(str(item))
    return list(unique_values)


def count_column_unique_values(column):
    """
    Count the occurrences of each unique value in the specified column, regardless of their data type.

    Args:
    - column: pandas Series or list containing elements of any data type.

    Returns:
    - dict: A dictionary where keys represent unique values and values represent their occurrences in the column.
    """
    unique_counts = {}
    for item in column:
        # Convert the object to a string before adding it to the dictionary
        key = str(item)
        if key in unique_counts:
            unique_counts[key] += 1
        else:
            unique_counts[key] = 1
    return unique_counts

def column_unique_keys(column, in_list=False):
    """
    Return a set of unique keys present in the dictionaries within the specified column, including None.
    If in_list is True, the column is assumed to contain a list of dictionaries.

    Args:
    - column: pandas Series or list containing dictionaries or None.
    - in_list (optional): Boolean flag indicating if the column contains a list of dictionaries. Default is False.

    Returns:
    - set: A set containing unique keys present in the dictionaries within the column, including None.
    """
    unique_keys = set()
    if not in_list:
        for item in column:
            if item is not None:
                unique_keys.update(item.keys())
    else:
        for lst in column:
            if lst is not None:
                for item in lst:
                    if item is not None:
                        unique_keys.update(item.keys())
    return unique_keys

def missing_values_table(df, message = False):
    """
    Generate a table with the number and percentage of missing values for each column in the DataFrame.

    Args:
    - df (pandas DataFrame): Dataframe to check.
    - message (bool): If print the message

    Returns:
    - pandas DataFrame: A table with columns for the number of missing values, the percentage of missing values,
      and the data type of each column.
    """
    # Calculate the total number of missing values per column
    missing_values = df.isnull().sum()
    
    # Calculate the percentage of missing values per column
    missing_percent = 100 * missing_values / len(df)
    missing_percent = missing_percent.round(2)
    
    # Create a DataFrame with the number and percentage of missing values per column
    missing_table = pd.DataFrame({'Missing Values': missing_values, '% of Total Values': missing_percent})
    
    # Add a column with the data type of each column
    missing_table['Data Type'] = df.dtypes
    
    # Filter only columns with missing values
    missing_table = missing_table[missing_table['Missing Values'] != 0]
    
    # Sort the DataFrame by the percentage of missing values in descending order
    missing_table = missing_table.sort_values('% of Total Values', ascending=False)
    
    # Display information about the total number of columns and rows
    if message:
        print(f"Your selected dataframe has {df.shape[1]} columns and {df.shape[0]} rows.\n"
          f"There are {missing_table.shape[0]} columns that have missing values.")
    
    return missing_table

def preprocess_str_columns(df, columns):
    """
    Preprocess specified columns in a DataFrame.

    Args:
    - df (DataFrame): The DataFrame containing the columns to preprocess.
    - columns (list): A list of column names to preprocess.

    Returns:
    - DataFrame: The DataFrame with the specified columns preprocessed.
    """

    for col in columns:
        # Convert empty strings or strings with only spaces to None
        df[col] = df[col].apply(lambda x: None if x is None or (isinstance(x, str) and x.strip() == "") else x)

        # Convert string numerical values with decimals to float
        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, str) and x.replace(".", "", 1).isdigit() else x)

        # Convert string numerical values without decimals to int
        df[col] = df[col].apply(lambda x: int(float(x)) if isinstance(x, str) and x.replace(".", "", 1).isdigit() and float(x) == int(float(x)) else x)


#---------------------------------------------------------------------------------------------------