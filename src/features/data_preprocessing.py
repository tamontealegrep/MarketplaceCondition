
import pandas as pd
from .data_preprocess_funct import missing_values_table, categorize_data_types, preprocess_str_columns

#---------------------------------------------------------------------------------------------------

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
    COLUMNS_EMPTY = []
    COLUMNS_USED = []
    COLUMNS_UNUSED = []
    COLUMNS_TRANSFORMED = []
    COLUMNS_CUSTOM = []
    COLUMNS_CUSTOM_UNUSED = []

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

    # STEP 00: Delete target column
    if COLUMN_TARGET in df.columns:
        df.drop(COLUMN_TARGET, axis=1, inplace=True)

    # STEP 01: Unpack and delete seller_address 
    df['seller_country'] = df.apply(lambda x : x['seller_address']['country']['name'], axis = 1) # Will be deleted later
    df['seller_state'] = df.apply(lambda x : x['seller_address']['state']['name'], axis = 1)
    df['seller_city'] = df.apply(lambda x : x['seller_address']['city']['name'], axis = 1) # Will be deleted later
    COLUMNS_CUSTOM.extend(['seller_country','seller_state','seller_city'])

    df = df.drop(columns=["seller_address"])
    COLUMNS_TRANSFORMED.append("seller_address")

    # STEP 02: Unpack and delete shipping
    df['shipping_dimensions'] = df.apply(lambda x : x['shipping'].get('dimensions', None), axis = 1)
    df['free_shipping'] = df.apply(lambda x : x['shipping'].get('free_shipping', None), axis = 1)
    df['local_pick_up'] = df.apply(lambda x : x['shipping'].get('local_pick_up', None), axis = 1)
    df['shipping_methods'] = df.apply(lambda x : x['shipping'].get('methods', None), axis = 1) # Will be deleted later
    df['shipping_free_methods'] = df.apply(lambda x : x['shipping'].get('free_methods', None), axis = 1) # Will be deleted later
    df['shipping_mode'] = df.apply(lambda x : x['shipping'].get('mode', None), axis = 1)
    df['shipping_tags'] = df.apply(lambda x : x['shipping'].get('tags', None), axis = 1) # Will be deleted later
    COLUMNS_CUSTOM.extend(['shipping_dimensions','free_shipping','local_pick_up','shipping_methods','shipping_free_methods','shipping_mode','shipping_tags'])

    df = df.drop(columns=["shipping"])
    COLUMNS_TRANSFORMED.append("shipping")

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
        COLUMNS_CUSTOM.append(value)

    df = df.drop(columns=['tags'])
    COLUMNS_TRANSFORMED.append('tags')

    # STEP 06: Get the description of the available payment methods.
    df['non_mercado_pago_payment_methods'] = df['non_mercado_pago_payment_methods'].apply(lambda x: [d.get('description') for d in x] if x else [])

    # STEP 07: Unpack non_mercado_pago_payment_methods in boolean columns and delete non_mercado_pago_payment_methods
    unique_payments = set(item for sublist in df['non_mercado_pago_payment_methods'] for item in sublist)

    for value in unique_payments:
        df[value] = df['non_mercado_pago_payment_methods'].apply(lambda x: value in x if x else False)
        COLUMNS_CUSTOM.append(value)

    df = df.drop(columns=['non_mercado_pago_payment_methods'])
    COLUMNS_TRANSFORMED.append('non_mercado_pago_payment_methods')

    # STEP 08: Get the number of pictures of the product and delete pictures.
    df['num_pictures']  = df['pictures'].apply(lambda x: len(x) if isinstance(x, list) else None)
    COLUMNS_CUSTOM.append('num_pictures')

    df = df.drop(columns=["pictures"])
    COLUMNS_TRANSFORMED.append('pictures')

    # STEP 09: Delete variations
    df = df.drop(columns=["variations"])
    COLUMNS_EMPTY.append("variations")

    # STEP 10: Delete attributes
    df = df.drop(columns=["attributes"])
    COLUMNS_EMPTY.append("attributes")

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
            COLUMNS_EMPTY.append(column)

    # STEP 14: Delete descriptions
    df = df.drop(columns=['descriptions'])
    COLUMNS_UNUSED.append('descriptions')

    # STEP 15: Create warranty_info and delete descriptions
    df['warranty_info'] = df['warranty'].apply(lambda x: True if x is not None else False)
    COLUMNS_CUSTOM.append('warranty_info')
    df = df.drop(columns=["warranty"])
    COLUMNS_TRANSFORMED.append('warranty')

    # STEP 16: Create buenos_aires_seller
    df['buenos_aires_seller'] = df['seller_state'].isin(['Buenos Aires', 'Capital Federal'])
    COLUMNS_CUSTOM.append('buenos_aires_seller')

    # STEP 17: Delete seller_country, seller_state, seller_city
    address_columns = ["seller_country", "seller_state", "seller_city"]
    for column in address_columns:
        df = df.drop(columns=[column])
        if column == "seller_state":
            COLUMNS_TRANSFORMED.append(column)
        else:
            COLUMNS_CUSTOM_UNUSED.append(column)

    # STEP 18: Unpack and delete shipping_mode
    shipping_modes = list(df['shipping_mode'].unique())

    for mode in shipping_modes:
        df[f'shipping_{mode}'] = df['shipping_mode'] == mode
        COLUMNS_CUSTOM.append(f'shipping_{mode}')

    df = df.drop(columns=['shipping_mode'])
    COLUMNS_TRANSFORMED.append('shipping_mode')

    # STEP 19: Unpack and delete shipping_mode
    df = df.drop(columns=['international_delivery_mode'])
    COLUMNS_UNUSED.append('international_delivery_mode')

    # STEP 20: Create dragged_bids_or_visits
    df["dragged_bids_or_visits"] = df["dragged_bids_and_visits"] | df["dragged_visits"]

    # STEP 21: Delete dragged_bids_and_visits, dragged_visits, free_relist, good_quality_thumbnail, poor_quality_thumbnail
    tag_columns = ['dragged_bids_and_visits','dragged_visits','free_relist','good_quality_thumbnail','poor_quality_thumbnail']

    for column in tag_columns:
        df = df.drop(columns=[column])
        if column in ["dragged_bids_and_visits", "dragged_visits"]:
            COLUMNS_TRANSFORMED.append(column)
        else:
            COLUMNS_CUSTOM_UNUSED.append(column)

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
        COLUMNS_TRANSFORMED.append(column)

    # STEP 25: Make price_in_usd and delete currency_id
    df['price_in_usd'] = df['currency_id'].replace({'USD': True, 'ARS': False})
    COLUMNS_CUSTOM.append('price_in_usd')

    df = df.drop(columns=['currency_id'])
    COLUMNS_TRANSFORMED.append('currency_id')

    # STEP 26: Delete base_price
    df = df.drop(columns=['base_price'])
    COLUMNS_UNUSED.append('base_price')

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
    COLUMNS_CUSTOM.extend(['start_week','start_day','stop_week','stop_day','days_active'])

    # STEP 29: Delete time columns
    time_columns = ['start_time', 'stop_time', 'date_created', 'last_updated']
    for column in time_columns:
        df = df.drop(columns=[column])
        if column in ['start_time', 'stop_time']:
            COLUMNS_TRANSFORMED.append(column)
        else:
            COLUMNS_UNUSED.append(column)
            
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
        COLUMNS_CUSTOM.append(f'is_{status}')

    df = df.drop(columns=['status'])
    COLUMNS_TRANSFORMED.append('status')

    # STEP 31: Unpack and delete buying_mode
    buying_modes = list(df['buying_mode'].unique())
    for mode in buying_modes:
        df[f'mode_{mode}'] = df['buying_mode'] == mode
        COLUMNS_CUSTOM.append(f'mode_{mode}')

    df = df.drop(columns=['buying_mode'])
    COLUMNS_TRANSFORMED.append('buying_mode')

    # STEP 32: Unpack and delete listing_type_id
    gold_categories = ['gold_special', 'gold', 'gold_premium', 'gold_pro']
    df['listing_free'] = df['listing_type_id'] == 'free'
    df['listing_bronze'] = df['listing_type_id'] == 'bronze'
    df['listing_silver'] = df['listing_type_id'] == 'silver'
    df['listing_gold'] = df['listing_type_id'].isin(gold_categories)
    COLUMNS_CUSTOM.extend(['listing_free','listing_bronze','listing_silver','listing_gold'])

    df = df.drop(columns=['listing_type_id'])
    COLUMNS_TRANSFORMED.append('listing_type_id')

    # STEP 33: Delete link columns
    links_columns = ["thumbnail", "secure_thumbnail", "permalink"]
    for column in links_columns:
        df = df.drop(columns=[column])
        COLUMNS_UNUSED.append(column)

    # STEP 34: Delete id columns
    id_columns = ["title","seller_id","id","parent_item_id","category_id","site_id" ]
    for column in id_columns:
        df = df.drop(columns=[column])
        COLUMNS_UNUSED.append(column)

    # STEP 35: Organize dataframe columns
    df = df.reindex(columns=COLUMNS_ORDER)

    # Update COLUMNS_USED
    used_columns = ["initial_quantity", "sold_quantity", "available_quantity",'automatic_relist']
    for column in used_columns:
        COLUMNS_USED.append(column)

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