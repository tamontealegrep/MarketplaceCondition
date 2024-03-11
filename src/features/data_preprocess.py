
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
    missing_values = df.isnull().sum()
    missing_percent = 100 * missing_values / len(df)
    missing_percent = missing_percent.round(2)
    
    missing_table = pd.DataFrame({'Missing Values': missing_values, '% of Total Values': missing_percent})
    
    missing_table['Data Type'] = df.dtypes
    missing_table = missing_table[missing_table['Missing Values'] != 0]
    missing_table = missing_table.sort_values('% of Total Values', ascending=False)
    
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


def preprocess_dataset(df_data,df_target=None,del_na=False,bool_values=False, message=False):
    """
    Preprocesses a dataset by performing various transformations such as unpacking nested columns,
    creating new features, deleting unnecessary columns, and organizing the dataframe columns.

    Args:
        df_data (DataFrame): The raw dataset to preprocess in the format of MLA_100k_checked_v3.jsonlines. 
        df_target (DataFrame, optional): The target dataset containing the 'condition' column. Defaults to None.
        del_na (bool, optional): Whether to delete rows with missing values. Defaults to False.
        bool_values (bool, optional): Wheter to mantain the boolean columns, if False transforms bool columns into 0 and 1. Default False
        message (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        DataFrame or tuple of DataFrames: The preprocessed dataset(s).

    Raises:
        ValueError: If 'df_target' does not contain the 'condition' column.
    """

    COLUMN_TARGET = "condition"

    COLUMNS_ORDER = [
    # Info del producto
    "price","price_in_usd","initial_quantity","sold_quantity","available_quantity","warranty_info",
    # Info del modo de compra
    "mode_buy_it_now","mode_classified","mode_auction",
    # Info de medios de pago
    "cash_payment","card_payment","bank_payment","mercadopago_payment","agree_with_buyer_payment",
    # Info del envio
    "shipping_me1","shipping_me2","shipping_custom","shipping_not_specified","free_shipping","local_pick_up",
    # Info del vendedor
    "buenos_aires_seller","listing_free","listing_bronze","listing_silver","listing_gold",
    # Info de la publicacion
    "start_week","start_day","stop_week","stop_day","days_active","is_active",
    #"is_paused","is_closed",
    "num_pictures","automatic_relist","dragged_bids_or_visits"
    ]  

    df = df_data
    if df_target is not None:
        if COLUMN_TARGET not in df_target.columns:
            raise ValueError(f"df_target must have the column {COLUMN_TARGET}") 

    # STEP 01: Unpack and delete seller_address 
    df['seller_country'] = df.apply(lambda x : x['seller_address']['country']['name'], axis = 1) 
    df['seller_state'] = df.apply(lambda x : x['seller_address']['state']['name'], axis = 1)
    df['seller_city'] = df.apply(lambda x : x['seller_address']['city']['name'], axis = 1) 

    df = df.drop(columns=["seller_address"])

    # STEP 02: Unpack and delete shipping
    df['shipping_dimensions'] = df.apply(lambda x : x['shipping'].get('dimensions', None), axis = 1)
    df['free_shipping'] = df.apply(lambda x : x['shipping'].get('free_shipping', None), axis = 1)
    df['local_pick_up'] = df.apply(lambda x : x['shipping'].get('local_pick_up', None), axis = 1)
    df['shipping_methods'] = df.apply(lambda x : x['shipping'].get('methods', None), axis = 1) # Will be deleted later
    df['shipping_free_methods'] = df.apply(lambda x : x['shipping'].get('free_methods', None), axis = 1) # Will be deleted later
    df['shipping_mode'] = df.apply(lambda x : x['shipping'].get('mode', None), axis = 1)
    df['shipping_tags'] = df.apply(lambda x : x['shipping'].get('tags', None), axis = 1) # Will be deleted later

    df = df.drop(columns=["shipping"])

    # STEP 03: Unpack ists with a length of up to 1.
    columns_to_unpack = ["coverage_areas", "sub_status", "deal_ids", "descriptions", "shipping_tags","shipping_methods"]

    for column in columns_to_unpack:
        df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

    # STEP 04: Unpack "descriptions"
    df['descriptions'] = df['descriptions'].apply(lambda x: eval(x) if x is not None else None)

    # STEP 05: Unpack tags in boolean columns and delete tags
    unique_tags = set(item for sublist in df['tags'] for item in sublist)

    for value in unique_tags:
        df[value] = df['tags'].apply(lambda x: value in x if x else False)

    df = df.drop(columns=['tags'])

    # STEP 06: Get the description of the available payment methods.
    df['non_mercado_pago_payment_methods'] = df['non_mercado_pago_payment_methods'].apply(lambda x: [d.get('description') for d in x] if x else [])

    # STEP 07: Unpack non_mercado_pago_payment_methods in boolean columns and delete non_mercado_pago_payment_methods
    unique_payments = set(item for sublist in df['non_mercado_pago_payment_methods'] for item in sublist)

    for value in unique_payments:
        df[value] = df['non_mercado_pago_payment_methods'].apply(lambda x: value in x if x else False)

    df = df.drop(columns=['non_mercado_pago_payment_methods'])

    # STEP 08: Get the number of pictures of the product and delete pictures.
    df['num_pictures']  = df['pictures'].apply(lambda x: len(x) if isinstance(x, list) else None)

    df = df.drop(columns=["pictures"])

    # STEP 09: Delete variations
    df = df.drop(columns=["variations"])

    # STEP 10: Delete attributes
    df = df.drop(columns=["attributes"])

    # STEP 11: Cast str columns
    types_df = categorize_data_types(df)
    columns_w_str = types_df[types_df['basic_types'].str.contains("str")].index.tolist()

    preprocess_str_columns(df, columns_w_str)

    # STEP 12: Get columns with missing data
    missing_table = missing_values_table(df)
    empty_columns = missing_table.index.tolist()

    # STEP 13: Delete empty columns
    empty_columns_expeptions = ["warranty", "parent_item_id", "descriptions", "thumbnail", "secure_thumbnail", "seller_country", "seller_state", "seller_city"]

    for column in empty_columns:
        if column not in empty_columns_expeptions:
            df = df.drop(columns=[column])

    # STEP 14: Delete descriptions
    df = df.drop(columns=['descriptions'])

    # STEP 15: Create warranty_info and delete descriptions
    df['warranty_info'] = df['warranty'].apply(lambda x: True if x is not None else False)
    df = df.drop(columns=["warranty"])

    # STEP 16: Create buenos_aires_seller
    df['buenos_aires_seller'] = df['seller_state'].isin(['Buenos Aires', 'Capital Federal'])

    # STEP 17: Delete seller_country, seller_state, seller_city
    address_columns = ["seller_country", "seller_state", "seller_city"]
    for column in address_columns:
        df = df.drop(columns=[column])

    # STEP 18: Unpack and delete shipping_mode
    shipping_modes = list(df['shipping_mode'].unique())

    for mode in shipping_modes:
        df[f'shipping_{mode}'] = df['shipping_mode'] == mode

    df = df.drop(columns=['shipping_mode'])

    # STEP 19: Unpack and delete shipping_mode
    df = df.drop(columns=['international_delivery_mode'])

    # STEP 20: Create dragged_bids_or_visits
    df["dragged_bids_or_visits"] = df["dragged_bids_and_visits"] | df["dragged_visits"]

    # STEP 21: Delete dragged_bids_and_visits, dragged_visits, free_relist, good_quality_thumbnail, poor_quality_thumbnail
    tag_columns = ['dragged_bids_and_visits','dragged_visits','free_relist','good_quality_thumbnail','poor_quality_thumbnail']

    for column in tag_columns:
        df = df.drop(columns=[column])

    # STEP 22: Group the payment methods.
    payment_mapping = {
        'agree_with_buyer_payment': ['Acordar con el comprador'],
        'card_payment': ['American Express', 'MasterCard', 'Mastercard Maestro', 'Tarjeta de crédito', 'Visa', 'Visa Electron', 'Diners'],
        'cash_payment': ['Efectivo', 'Giro postal', 'Contra reembolso'],
        'bank_payment': ['Transferencia bancaria', 'Cheque certificado'],
        'mercadopago_payment': ['MercadoPago']
    }
    for new_column, original_columns in payment_mapping.items():
        df[new_column] = df[original_columns].any(axis=1)

    # STEP 23: Join accepts_mercadopago and mercadopago_payment in mercadopago_payment
    df["mercadopago_payment"] = df["accepts_mercadopago"] | df["mercadopago_payment"]
    
    # STEP 24: Delete payment_methods_columns
    payment_methods_columns = ['Acordar con el comprador','American Express','Cheque certificado','Contra reembolso','Diners','Efectivo','Giro postal','MasterCard','Mastercard Maestro','MercadoPago','Tarjeta de crédito','Transferencia bancaria','Visa','Visa Electron','accepts_mercadopago']
    for column in payment_methods_columns:
        df = df.drop(columns=[column])

    # STEP 25: Make price_in_usd and delete currency_id
    df['price_in_usd'] = df['currency_id'].replace({'USD': True, 'ARS': False})

    df = df.drop(columns=['currency_id'])

    # STEP 26: Delete base_price
    df = df.drop(columns=['base_price'])

    # STEP 27: Change columns to datetime object.
    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
    df['stop_time'] = pd.to_datetime(df['stop_time'], unit='ms')
    df['date_created'] = pd.to_datetime(df['date_created'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    
    # STEP 28: Make start_week, start_day, stop_week, stop_day, days_active
    df['start_week'] = df['start_time'].dt.isocalendar().week
    df['start_day'] = df['start_time'].dt.dayofweek
    df['stop_week'] = df['stop_time'].dt.isocalendar().week
    df['stop_day'] = df['stop_time'].dt.dayofweek
    df['days_active'] = (df['stop_time'] - df['start_time']).dt.days

    # STEP 29: Delete time columns
    time_columns = ['start_time', 'stop_time', 'date_created', 'last_updated']
    for column in time_columns:
        df = df.drop(columns=[column])
            
    # STEP 30: Unpack and delete status
    status_values = list(df['status'].unique())
    status_exceptions = ["paused", "closed","not_yet_active"]
    for i in status_exceptions:
        try:
            status_values.remove("not_yet_active")
        except:
            pass
    for status in status_values:
        df[f'is_{status}'] = df['status'] == status

    df = df.drop(columns=['status'])

    # STEP 31: Unpack and delete buying_mode
    buying_modes = list(df['buying_mode'].unique())
    for mode in buying_modes:
        df[f'mode_{mode}'] = df['buying_mode'] == mode

    df = df.drop(columns=['buying_mode'])

    # STEP 32: Unpack and delete listing_type_id
    gold_categories = ['gold_special', 'gold', 'gold_premium', 'gold_pro']
    df['listing_free'] = df['listing_type_id'] == 'free'
    df['listing_bronze'] = df['listing_type_id'] == 'bronze'
    df['listing_silver'] = df['listing_type_id'] == 'silver'
    df['listing_gold'] = df['listing_type_id'].isin(gold_categories)

    df = df.drop(columns=['listing_type_id'])

    # STEP 33: Delete link columns
    links_columns = ["thumbnail", "secure_thumbnail", "permalink"]
    for column in links_columns:
        df = df.drop(columns=[column])

    # STEP 34: Delete id columns
    id_columns = ["title","seller_id","id","parent_item_id","category_id","site_id" ]
    for column in id_columns:
        df = df.drop(columns=[column])

    # STEP 35: Organize dataframe columns
    df = df.reindex(columns=COLUMNS_ORDER)

    # Concatenate dataframes
    if df_target is not None:
        df = pd.concat([df, df_target[COLUMN_TARGET]], axis=1)
        df['is_new'] = df[COLUMN_TARGET].map({'new': True, 'used': False})
        df = df.drop(columns=[COLUMN_TARGET])

    # Transforms boolean column
    if not bool_values:
        bool_columns = df.select_dtypes(include=bool).columns
        df[bool_columns] = df[bool_columns].astype(int)

    # Delete na rows:
    if del_na:
        df = df.dropna()
        
    if df_target is not None:
        df_data = df.drop(columns=['is_new'])
        df_target = df[['is_new']]
        return df_data, df_target
    else:
        df_data = df
        return df_data

#---------------------------------------------------------------------------------------------------