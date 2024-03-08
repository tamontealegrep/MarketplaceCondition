---

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets.
    │   ├── staging        <- Intermediate data that has been transformed.
    │   └── raw            <- The original MLA_100k_checked_v3.jsonlines.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks with steps for training and evaluating models.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.

--------

MeLi Code Exercise
==============================
# Challenge

In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the marketplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function `build_dataset` to read that dataset in `new_or_used.py`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. Additionally, you will have to choose an appropriate secondary metric and also elaborate an argument on why that metric was chosen.

- The file, including all the code needed to define and evaluate a model.
- A document with an explanation on the criteria applied to choose the features, the proposed secondary metric and the performance achieved on that metrics. 
- (Optional) EDA analysis with other format like .ipynb

--------
# **PREDICT WETHER MARKETPLACE PRODUCTS ARE NEW OR USED**

## 1. Load Data
```python
import random
import pandas as pd
from src.features.new_or_used import build_dataset
X_train, y_train, X_test, y_test = build_dataset()
```

```python
# Convert it into dataframes for easier manipulation.
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
```
--------
```python
# Evaluate if there are unique columns in the training and evaluation dataframes.
train_columns = set(X_train.columns)
test_columns = set(X_test.columns)

print("Number of training columns:\t", len(train_columns))
print("Number of test columns:\t\t", len(test_columns))

train_columns_unique = train_columns - test_columns

print(f"Unique training columns.: {train_columns_unique}")
```

*Output:*
```
Number of training columns:	 45
Number of test columns:		 44
Unique training columns.: {'condition'}
```
--------

```python
# Rename variable for easier workflow.
df = X_train
```
# 2. Data Preprocessing 

```python
def categorize_data_types(df):
    """
    Categorize data types present in each column of the DataFrame into 'basic_types' and 'others'.

    Args:
    - df: pandas DataFrame containing the data.

    Returns:
    - pandas DataFrame: A DataFrame where each row represents a column of the original DataFrame.
                        It contains two columns: 'basic_types' and 'others', which record the data types found in each column.
    """
    types_df = pd.DataFrame(columns=['basic_types', 'others'])

    for column in df.columns:
        types = set(df[column].apply(lambda x: type(x).__name__))
        
        basic_types = {t for t in types if t in ['bool','int', 'float', 'str', 'list', 'tuple', 'dict', 'NoneType']}
        other_types = types - basic_types
        
        types_df.loc[column] = [', '.join(basic_types), ', '.join(other_types)]

    return types_df
```

```python
# We obtain the types of objects present in each column of the dataframe.
types_df = categorize_data_types(df)
```

First, we will focus on working with columns composed of iterable types, as these can be unpacked into multiple lists, thus facilitating data manipulation.

## 2.1 Unpack dictonary columns
```python
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
```
--------
```python
# Retrieve the columns with dictionary values.
columns_w_dict = types_df[types_df['basic_types'].str.contains("dict")].index.tolist()
print(f"dict columns:\t",columns_w_dict)
```
*Output:*
```
dict columns:	 ['seller_address', 'shipping']
```
--------
### 2.1.1 seller_address
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'seller_address'
print(f"Unique keys in seller_address: {column_unique_keys(df['seller_address'])}")
```
*Output:*
```
Unique keys in seller_address: {'country', 'city', 'state'}
```
--------
```python
# Unpack and delete seller_address 
df['seller_country'] = df.apply(lambda x : x['seller_address']['country']['name'], axis = 1)
df['seller_state'] = df.apply(lambda x : x['seller_address']['state']['name'], axis = 1)
df['seller_city'] = df.apply(lambda x : x['seller_address']['city']['name'], axis = 1)

df = df.drop(columns=["seller_address"])
```
### 2.1.2 shipping
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'shipping'
print(f"Unique keys in shipping: {column_unique_keys(df['shipping'])}")
```
*Output:*
```
Unique keys in shipping: {'mode', 'tags', 'free_methods', 'local_pick_up', 'free_shipping', 'methods', 'dimensions'}
```
--------

```python
# Unpack and delete shipping
df['shipping_dimensions'] = df.apply(lambda x : x['shipping'].get('dimensions', None), axis = 1)
df['free_shipping'] = df.apply(lambda x : x['shipping'].get('free_shipping', None), axis = 1)
df['local_pick_up'] = df.apply(lambda x : x['shipping'].get('local_pick_up', None), axis = 1)
df['shipping_methods'] = df.apply(lambda x : x['shipping'].get('methods', None), axis = 1) 
df['shipping_free_methods'] = df.apply(lambda x : x['shipping'].get('free_methods', None), axis = 1) 
df['shipping_mode'] = df.apply(lambda x : x['shipping'].get('mode', None), axis = 1)
df['shipping_tags'] = df.apply(lambda x : x['shipping'].get('tags', None), axis = 1) 

df = df.drop(columns=["shipping"])
```
```python
# Update the types of objects present in each column of the dataframe.
types_df = categorize_data_types(df)
```
## 2.2 Unpack list columns
--------
```python
# Retrieve the columns with list values.
columns_w_list = types_df[types_df['basic_types'].str.contains("list")].index.tolist()
print(f"list columns:\t",columns_w_list)
```
*Output:*
```
list columns:	 ['sub_status', 'deal_ids', 'non_mercado_pago_payment_methods', 'variations', 'attributes', 'tags', 'coverage_areas', 'descriptions', 'pictures', 'shipping_methods', 'shipping_free_methods', 'shipping_tags']
```
--------
```python
# The lengths of the lists within the columns as this can provide us with a better understanding of how to handle the data.
for column in columns_w_list:
    lengths = set(df[column].apply(lambda x: len(x) if isinstance(x, list) else None))
    print(f"Lengths of {column}",lengths)
```
*Output:*
```
Lengths of sub_status {0, 1}
Lengths of deal_ids {0, 1}
Lengths of non_mercado_pago_payment_methods {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
Lengths of variations {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 34, 35, 36, 42, 50}
Lengths of attributes {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 31, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 52, 54, 63, 64, 65, 66, 67, 71, 73, 75, 77, 78, 81}
Lengths of tags {0, 1, 2}
Lengths of coverage_areas {0}
Lengths of descriptions {0, 1}
Lengths of pictures {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 36}
Lengths of shipping_methods {0.0, nan}
Lengths of shipping_free_methods {1.0, 2.0, nan}
Lengths of shipping_tags {0, 1}
```
--------
### 2.2.1 coverage_areas, sub_status, deal_ids, descriptions, shipping_tags and shipping_methods

```python
# Unpack ists with a length of up to 1.
columns_to_unpack = ["coverage_areas", "sub_status", "deal_ids", "descriptions", "shipping_tags","shipping_methods"]

for column in columns_to_unpack:
    df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
```
```python
# The 'descriptions' column contains dictionaries in string format. We need to convert them back to dictionaries for further processing.
df['descriptions'] = df['descriptions'].apply(lambda x: eval(x) if x is not None else None)
```
### 2.2.2 shipping_free_methods
--------
```python
# The 'shipping_free_methods' column is nearly empty and does not justify unpacking. 
count_column_unique_values(df["shipping_free_methods"])
```
*Output:*
```
{'None': 87311,
 "[{'rule': {'value': None, 'free_mode': 'country'}, 'id': 73328}]": 2641,
 "[{'rule': {'value': None, 'free_mode': 'country'}, 'id': 73330}]": 37,
 "[{'rule': {'value': None, 'free_mode': 'country'}, 'id': 501145}]": 6,
 "[{'rule': {'value': None, 'free_mode': 'country'}, 'id': 501146}]": 4,
 "[{'rule': {'value': None, 'free_mode': 'country'}, 'id': 73328}, {'rule': {'value': None, 'free_mode': 'country'}, 'id': 73330}]": 1}
```
--------
### 2.2.3 tags
```python
# Unpack tags in boolean columns and delete tags
unique_tags = set(item for sublist in df['tags'] for item in sublist)

for value in unique_tags:
    df[value] = df['tags'].apply(lambda x: value in x if x else False)

df = df.drop(columns=['tags'])
```
### 2.2.4 non_mercado_pago_payment_methods
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'non_mercado_pago_payment_methods'
print(f"Unique keys in non_mercado_pago_payment_methods: {column_unique_keys(df['non_mercado_pago_payment_methods'],True)}")
```
*Output:*
```
Unique keys in non_mercado_pago_payment_methods: {'id', 'type', 'description'}
```
--------
```python
# Unpack non_mercado_pago_payment_methods in boolean columns and delete non_mercado_pago_payment_methods
unique_payments = set(item for sublist in df['non_mercado_pago_payment_methods'] for item in sublist)

for value in unique_payments:
    df[value] = df['non_mercado_pago_payment_methods'].apply(lambda x: value in x if x else False)

df = df.drop(columns=['non_mercado_pago_payment_methods'])
```
### 2.2.5 pictures
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'pictures'
print(f"Unique keys in pictures: {column_unique_keys(df['pictures'],True)}")
```
*Output:*
```
Unique keys in pictures: {'max_size', 'quality', 'size', 'secure_url', 'url', 'id'}
```
--------
```python
# Get the number of pictures of the product.
df['num_pictures']  = df['pictures'].apply(lambda x: len(x) if isinstance(x, list) else None)
```
```python
# Delete pictures. None of the keys present in the dictionaries seem to contain relevant information, and several of them are devoid of any data.
df = df.drop(columns=["pictures"])
```
### 2.2.6 variations
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'variations'
print(f"Unique keys in variations: {column_unique_keys(df['variations'],True)}")
```
*Output:*
```
Unique keys in variations: {'price', 'attribute_combinations', 'seller_custom_field', 'available_quantity', 'sold_quantity', 'picture_ids', 'id'}
```
--------
```python
# Delete variations. Seems to contain irrelevant information, with the majority of its data being represented by None values.
df = df.drop(columns=["variations"])
```

### 2.2.7 attributes
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'attributes'
print(f"Unique keys in attributes: {column_unique_keys(X_train['attributes'],True)}")
```
*Output:*
```
Unique keys in attributes: {'value_name', 'value_id', 'attribute_group_name', 'attribute_group_id', 'name', 'id'}
```
--------
```python
# Delete attributes. seems to contain irrelevant information, with the majority of its data being represented by None values.
df = df.drop(columns=["attributes"])
```
## 2.3 Empty columns
```python
# Update the types of objects present in each column of the dataframe.
types_df = categorize_data_types(df)
```
```python
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
        df[col] = df[col].apply(lambda x: None if x is None or (isinstance(x, str) and x.strip() == "") else x)

        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, str) and x.replace(".", "", 1).isdigit() else x)

        df[col] = df[col].apply(lambda x: int(float(x)) if isinstance(x, str) and x.replace(".", "", 1).isdigit() and float(x) == int(float(x)) else x)
```
--------
```python
# Analysis of string-composed columns to detect empty strings, those solely comprising spaces, or numeric strings, aiming to enhance data usability through transformation
columns_w_str = types_df[types_df['basic_types'].str.contains("str")].index.tolist()
preprocess_str_columns(df, columns_w_str)
print(f"str columns:\t",columns_w_str)
```
*Output:*
```
str columns:	 ['warranty', 'sub_status', 'condition', 'deal_ids', 'site_id', 'listing_type_id', 'buying_mode', 'listing_source', 'parent_item_id', 'category_id', 'last_updated', 'international_delivery_mode', 'id', 'currency_id', 'thumbnail', 'title', 'date_created', 'secure_thumbnail', 'status', 'video_id', 'permalink', 'seller_country', 'seller_state', 'seller_city', 'shipping_dimensions', 'shipping_mode', 'shipping_tags']
```
--------
```python
# Get columns with missing data
missing_table = missing_values_table(df)
empty_columns = missing_table.index.tolist()
missing_table
```
*Output:*
```
| Missing Values           | % of Total Values | Data Type |
|--------------------------|-------------------|-----------|
| subtitle                 | 90000             | object    |
| listing_source           | 90000             | object    |
| coverage_areas           | 90000             | object    |
| shipping_methods         | 90000             | object    |
| differential_pricing     | 90000             | object    |
| catalog_product_id       | 89993             | float64   |
| shipping_dimensions      | 89978             | object    |
| shipping_tags            | 89941             | object    |
| original_price           | 89870             | float64   |
| deal_ids                 | 89783             | object    |
| official_store_id        | 89255             | float64   |
| sub_status               | 89109             | object    |
| video_id                 | 87324             | object    |
| shipping_free_methods    | 87311             | object    |
| warranty                 | 54786             | object    |
| parent_item_id           | 20690             | object    |
| descriptions             | 2417              | object    |
| thumbnail                | 703               | object    |
| secure_thumbnail         | 703               | object    |
| seller_country           | 1                 | object    |
| seller_state             | 1                 | object    |
| seller_city              | 2                 | object    |
```
--------
```python
# The 'warranty', 'parent_item_id' and 'descriptions' columns will be evaluated further later on, while the rest of the columns will be removed due to their high percentage of missing values
empty_columns_exceptions = ["warranty", "parent_item_id", "descriptions", "thumbnail", "secure_thumbnail", "seller_country", "seller_state", "seller_city"]

for column in empty_columns_exceptions:
    empty_columns.remove(column)

print(f"Columns to remove due to high count of missing data: {empty_columns}")
print(f"Columns to be analyzed due to the presence of missing data: {empty_columns_exceptions}")

for column in empty_columns:
    df = df.drop(columns=[column])
```
*Output:*
```
Columns to remove due to high count of missing data: ['subtitle', 'listing_source', 'coverage_areas', 'shipping_methods', 'differential_pricing', 'catalog_product_id', 'shipping_dimensions', 'shipping_tags', 'original_price', 'deal_ids', 'official_store_id', 'sub_status', 'video_id', 'shipping_free_methods']
Columns to be analyzed due to the presence of missing data: ['warranty', 'parent_item_id', 'descriptions', 'thumbnail', 'secure_thumbnail', 'seller_country', 'seller_state', 'seller_city']
```
--------
### 2.3.1 Columns with missing data 
```python
# The 'warranty', 'parent_item_id' and 'descriptions' columns will be evaluated further later on, while the rest of the columns will be removed due to their high percentage of missing values
empty_columns_exceptions = ["warranty", "parent_item_id", "descriptions", "thumbnail", "secure_thumbnail", "seller_country", "seller_state", "seller_city"]

for column in empty_columns_exceptions:
    empty_columns.remove(column)

print(f"Columns to remove due to high count of missing data: {empty_columns}")
print(f"Columns to be analyzed due to the presence of missing data: {empty_columns_exceptions}")

for column in empty_columns:
    df = df.drop(columns=[column])
```
### 2.3.2 descriptions
--------
```python
# Retrieve all the unique keys present in the dictionaries within the column 'descriptions'
print(f"Unique keys in descriptions: {column_unique_keys(df['descriptions'])}")
```
*Output:*
```
Unique keys in descriptions: {'id'}
```
--------
```python
# Compare the information stored in the 'descriptions' column with that stored in the 'id' column.
random_row = random.randint(0, df.shape[0])
print(f"Information for row {random_row} in column id: {df['id'].iloc[random_row]}, and in column descriptions: {df['descriptions'].iloc[random_row]['id']}")
```
*Output:*
```
Information for row 50198 in column id: MLA1923708022, and in column descriptions: MLA1923708022-906724134
```
--------
```python
# Considering the high similarity between the information in descriptions and that in id, descriptions is removed to avoid redundancy and streamline the dataset.
df = df.drop(columns=['descriptions'])
```
### 2.3.3 warranty
--------
```python
# Extract the unique values from the column 'warranty'
print(f"Number of unique values: {len(df['warranty'].value_counts())}")
```
*Output:*
```
Number of unique values: 9535
```
--------
Given its extensive range of categories and considerable missing data, the decision has been made to discard this column. The creation of the new column "warranty_info" serves to distill complex warranty information into a simplified binary variable, capturing whether any warranty details are provided. This transformation streamlines the feature space while retaining crucial insights about the presence or absence of warranty information. Moreover, it acknowledges the tendency for new items to frequently disclose warranty duration or status, contrasting with the relatively rare occurrence of warranty information for used products.


```python
# Create 'warranty_info' and delete 'warranty'
df['warranty_info'] = df['warranty'].apply(lambda x: True if x is not None else False)
df = df.drop(columns=["warranty"])
```
Now that we've handled columns containing lists and dictionaries, as well as empty columns, let's shift our attention to columns composed of basic Python types, we will work with groups of related columns.
## 2.4 Address columns
--------
```python
# Obtain the count of unique values for each column.
address_columns = ["seller_country", "seller_state", "seller_city"]

for column in address_columns:
    print(f"Unique values of {column}: {len(df[column].unique())}")
```
*Output:*
```
Unique values of seller_country: 2
Unique values of seller_state: 25
Unique values of seller_city: 3480
```
--------
```python
# Get the 'seller_country' value counts
df["seller_country"].value_counts()
```
*Output:*
```
Argentina    89999
Name: seller_country, dtype: int64
```
--------
```python
# Get the 'seller_country' value counts
df["seller_state"].value_counts()
```
*Output:*
```
Capital Federal        52143
Buenos Aires           31482
Santa Fe                2398
Córdoba                 1727
Mendoza                  400
Chubut                   335
Entre Ríos               249
Tucumán                  214
San Juan                 132
Salta                    131
Misiones                 122
Río Negro                119
Corrientes               110
Neuquén                   87
La Pampa                  70
Chaco                     69
San Luis                  56
Jujuy                     33
Formosa                   28
Santiago del Estero       26
Santa Cruz                22
Catamarca                 20
La Rioja                  16
Tierra del Fuego          10
Name: seller_state, dtype: int64
```
--------
Creating a new column "buenos_aires_seller" to combine sellers from Buenos Aires and the Federal Capital is justified due to their geographical proximity and socioeconomic similarities. This consolidation simplifies data analysis, ensures consistency, and facilitates meaningful comparisons. Additionally, sellers from these two regions constitute the vast majority of the dataset, making this grouping approach highly relevant and practical.
```python
# Create 'buenos_aires_seller'
df['buenos_aires_seller'] = df['seller_state'].isin(['Buenos Aires', 'Capital Federal'])
```
Given that all sellers are from Argentina, the information provided by the column seller_country is unnecessary. Furthermore, since the majority of sellers come from the Buenos Aires province or the Federal Capital, the detailed information in the seller_city column is also unnecessary, especially considering its numerous categories.
```python
# Delete 'seller_country', 'seller_state', 'seller_city'
for column in address_columns:
    df = df.drop(columns=[column])
```
## 2.5 Shipping columns

```python
shipping_columns = ["shipping_mode", "free_shipping", "local_pick_up", "international_delivery_mode"]
```
--------
```python
print(f"Number of unique clases in shipping_mode: {list(df['shipping_mode'].unique())}")
```
*Output:*
```
Number of unique clases in shipping_mode: ['not_specified', 'me2', 'custom', 'me1']
```
--------
```python
# Unpack the 'shipping_mode' column into 'shipping_not_specified', 'shipping_me2', 'shipping_custom', and 'shipping_me1'.
shipping_modes = list(df['shipping_mode'].unique())

for mode in shipping_modes:
    df[f'shipping_{mode}'] = df['shipping_mode'] == mode

df = df.drop(columns=['shipping_mode'])
```
--------
```python
# Check the values ​​present in the 'international_delivery_mode' column.
df["international_delivery_mode"].value_counts()
```
*Output:*
```
none    90000
Name: international_delivery_mode, dtype: int64
```
The 'international_delivery_mode' column has a single value, a string of 'none', so it will be removed.

--------
```python
# Delete 'international_delivery_mode'
df = df.drop(columns=['international_delivery_mode'])
```
The 'free_shipping' and 'local_pick_up' columns require no further processing.
## 2.6 Tag columns
```python
tag_columns = ['dragged_bids_and_visits','dragged_visits','free_relist','good_quality_thumbnail','poor_quality_thumbnail']
```
--------
```python
tags_true_counts = df[list(unique_tags)].sum(axis=0)
print("Number of True elements per column of tags:")
print(tags_true_counts)
```
*Output:*
```
Number of True elements per column of tags:
dragged_visits               723
good_quality_thumbnail      1537
free_relist                  259
dragged_bids_and_visits    66516
poor_quality_thumbnail        13
dtype: int64
```
--------
Given the similar nature of the 'dragged_bids_and_visits' and 'dragged_visits' columns, the information will be merged into a single column named 'dragged_bids_or_visits'
```python
# Create dragged_bids_or_visits
df["dragged_bids_or_visits"] = df["dragged_bids_and_visits"] | df["dragged_visits"]
```
Due to the low occurrence of True values in the 'good_quality_thumbnail', 'poor_quality_thumbnail', and 'free_relist' columns, they will be removed.
```python
# Delete dragged_bids_and_visits, dragged_visits, free_relist, good_quality_thumbnail, poor_quality_thumbnail
for column in tag_columns:
    df = df.drop(columns=[column])
```
## 2.7 Payment methods molumns
```python
payment_methods_columns = ['Acordar con el comprador','American Express','Cheque certificado','Contra reembolso','Diners','Efectivo','Giro postal','MasterCard','Mastercard Maestro','MercadoPago','Tarjeta de crédito','Transferencia bancaria','Visa','Visa Electron','accepts_mercadopago']
```
The payment methods were grouped as follows:

- **Agree_with_buyer_payment**: Only includes the payment method "Acordar con el comprador". This method involves agreeing on the payment directly with the buyer.
  
- **Card_payment**: Includes various credit and debit card brands such as American Express, MasterCard, Visa, etc. These methods involve payment using a credit or debit card.
  
- **Cash_payment**: Includes payment methods where cash is involved, such as "Efectivo" and "Giro postal". Also includes "Contra reembolso", which typically implies cash payment upon delivery.
  
- **Bank_payment**: Includes bank-related payment methods such as "Transferencia bancaria" and "Cheque certificado". These methods involve transferring funds between bank accounts.
  
- **Mercadopago_payment**: Specifically includes the payment method "MercadoPago".
```python
# Group the payment methods.
payment_mapping = {
    'agree_with_buyer_payment': ['Acordar con el comprador'],
    'card_payment': ['American Express', 'MasterCard', 'Mastercard Maestro', 'Tarjeta de crédito', 'Visa', 'Visa Electron', 'Diners'],
    'cash_payment': ['Efectivo', 'Giro postal', 'Contra reembolso'],
    'bank_payment': ['Transferencia bancaria', 'Cheque certificado'],
    'mercadopago_payment': ['MercadoPago']
}
for new_column, original_columns in payment_mapping.items():
    df[new_column] = df[original_columns].any(axis=1)
```
The column 'accepts_mercadopago' is conveying similar information to the previously created 'mercadopago_payment'. Therefore, the data from both columns will be merged.
```python
# Join 'accepts_mercadopago' and 'mercadopago_payment' in 'mercadopago_payment'
df["mercadopago_payment"] = df["accepts_mercadopago"] | df["mercadopago_payment"]
```
```python
# Delete payment_methods_columns
for column in payment_methods_columns:
    df = df.drop(columns=[column])
```
## 2.8 Price Columns
```python
price_columns = ["currency_id", "price", "base_price"]
```
--------
```python
# Check on the unique values present in the 'currency_id' column
df["currency_id"].value_counts()
```
*Output:*
```
ARS    89496
USD      504
Name: currency_id, dtype: int64
```
--------
The 'currency_id' column is transformed into 'price_in_usd' to indicate when the listing displays its price in USD.
```python
# Make 'price_in_usd' and 'delete currency_id'
df['price_in_usd'] = df['currency_id'].replace({'USD': True, 'ARS': False})
df = df.drop(columns=['currency_id'])
```
'price' and 'base_price' appear to have a similar nature. An evaluation will be conducted to determine if there are significant differences in their values.

--------
```python
# Check differences between 'base_price' and 'price differ'
print(f"Number of samples where base_price and price differ: {len(df[df['price'] != df['base_price']])}")
```
*Output:*
```
Number of samples where base_price and price differ: 26
```
--------
```python
# Perform data validation on 'price'
print(f"Number of inavlid values on price: {len(df[df['price']< 0])}")
```
*Output:*
```
Number of inavlid values on price: 0
```
--------
Due to the similarity between the data in 'price' and 'base_price', one of the two columns will be removed to avoid redundancy in the data.
```python
# Delete base_price
df = df.drop(columns=['base_price'])
```
## 2.9 Time columns
```python
time_columns = ['start_time', 'stop_time', 'date_created', 'last_updated']
```
```python
# Change columns to datetime object.
df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
df['stop_time'] = pd.to_datetime(df['stop_time'], unit='ms')
df['date_created'] = pd.to_datetime(df['date_created'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df['last_updated'] = pd.to_datetime(df['last_updated'], format='%Y-%m-%dT%H:%M:%S.%fZ')
```
'start_time' and 'date_created' appear to have a similar nature. An evaluation will be conducted to determine if there are significant differences in their values.

--------
```python
print(f"Number of samples where start_time and date_created differ: {len(df[df['start_time'] != df['date_created']])}")
```
*Output:*
```
Number of samples where start_time and date_created differ: 18238
```
--------
Upon thorough comparison of the columns 'start_time' and 'date_created', it was found that while there are indeed differences between the two columns, these discrepancies amount to just one second. Hence, for the purposes of this analysis, both columns effectively contain the same information.

--------
```python
df.loc[df['start_time'] != df['date_created'], ['start_time', 'date_created']].head(10)
```
*Output:*
```markdown
| start_time | date_created |
|------------|--------------|
| 2015-08-30 14:24:01 | 2015-08-30 14:24:02 |
| 2015-10-03 23:11:29 | 2015-10-03 23:11:30 |
| 2015-09-30 15:05:20 | 2015-09-30 15:05:21 |
| 2015-10-14 15:02:18 | 2015-10-14 15:02:19 |
| 2015-09-25 18:02:07 | 2015-09-25 18:02:08 |
| 2015-09-23 13:47:26 | 2015-09-23 13:47:27 |
| 2015-09-17 14:15:35 | 2015-09-17 14:15:36 |
| 2015-09-25 14:48:16 | 2015-09-25 14:48:17 |
| 2015-09-21 17:08:42 | 2015-09-21 17:08:43 |
| 2015-10-12 19:57:48 | 2015-10-12 19:57:49 |

```
--------
The corresponding columns for the week number and day of the week when the offer began, and likewise for when it ended, are created. Additionally, a column is generated to calculate the number of days that elapsed from the start to the end of the offer.
```python
# Make 'start_week', 'start_day', 'stop_week', 'stop_day', 'days_active'
df['start_week'] = df['start_time'].dt.isocalendar().week
df['start_day'] = df['start_time'].dt.dayofweek
df['stop_week'] = df['stop_time'].dt.isocalendar().week
df['stop_day'] = df['stop_time'].dt.dayofweek
df['days_active'] = (df['stop_time'] - df['start_time']).dt.days
```
The columns with redundant information are removed, and it was determined that the 'last_updated' column does not provide relevant information.
```python
# Delete time columns
for column in time_columns:
    df = df.drop(columns=[column])
```
## 2.10 Quantity columns
```python
quantity_columns = ["initial_quantity", "sold_quantity", "available_quantity"]
```
An integrity check is performed on the data for 'initial_quantity', 'sold_quantity', and 'available_quantity' columns.

--------
```python
print(f"Number of inavlid values on initial_quantity: {len(df[df['initial_quantity']< 1])}")
print(f"Number of inavlid values on sold_quantity: {len(df[df['sold_quantity']< 0])}")
print(f"Number of inavlid values on available_quantity: {len(df[df['available_quantity']< 0])}")
```
*Output:*
```
Number of inavlid values on initial_quantity: 0
Number of inavlid values on sold_quantity: 0
Number of inavlid values on available_quantity: 0
```
--------
The 'initial_quantity', 'sold_quantity' and 'available_quantity' columns require no further processing.

## 2.11 Listing control columns
```python
listing_control_columns =["status", "buying_mode", "listing_type_id", "automatic_relist"]
```
Analyze the value counts of column 'status' to gain insights into its distribution and frequency of occurrence.

--------
```python
df["status"].value_counts()
```
*Output:*
```
active            86116
paused             3863
closed               20
not_yet_active        1
Name: status, dtype: int64
```
--------
The unpacking of the "status" column into the variables "is_active" "is_paused" "is_closed" of "is_not_yet_active" was not performed due to the limited amount of meaningful information it could provide.
```python
# Unpack and delete status
status_values = list(df['status'].unique())
status_values.remove("not_yet_active")
status_values.remove("paused")
status_values.remove("closed")

for status in status_values:
    df[f'is_{status}'] = df['status'] == status

df = df.drop(columns=['status'])
```
We will analyze the value counts of column 'buying_mode' to gain insights into its distribution and frequency of occurrence.

--------
```python
buying_modes = df["buying_mode"].value_counts()
buying_modes
```
*Output:*
```
buy_it_now    87311
classified     1982
auction         707
Name: buying_mode, dtype: int64
```
--------
The unpacking of the 'status' column into the variables 'buy_it_now' 'classified' and 'auction''
```python
# Unpack and delete buying_mode
buying_modes = list(df['buying_mode'].unique())
for mode in buying_modes:
    df[f'mode_{mode}'] = df['buying_mode'] == mode

df = df.drop(columns=['buying_mode'])
```
We will analyze the value counts of column 'listing_type_id' to gain insights into its distribution and frequency of occurrence.

--------
```python
df['listing_type_id'].value_counts()
```
*Output:*
```
bronze          56904
free            19260
silver           8195
gold_special     2693
gold             2170
gold_premium      765
gold_pro           13
Name: listing_type_id, dtype: int64
```
--------
The column 'listing_type_id' will be unpacked to create new columns denoting each unique category present. Specifically, the categories 'gold_special', 'gold', 'gold_premium', and 'gold_pro' have been grouped under the listing_gold column. This grouping was done as these categories generally have a limited number of individual examples, allowing for better representation and generalization in the new combined column.

```python
# Unpack and delete listing_type_id
df['listing_free'] = df['listing_type_id'] == 'free'
df['listing_bronze'] = df['listing_type_id'] == 'bronze'
df['listing_silver'] = df['listing_type_id'] == 'silver'

gold_categories = ['gold_special', 'gold', 'gold_premium', 'gold_pro']
df['listing_gold'] = df['listing_type_id'].isin(gold_categories)

df = df.drop(columns=['listing_type_id'])
```
The 'automatic_relist' column require no further processing.

## 2.12 Links columns
```python
links_columns = ["thumbnail", "secure_thumbnail", "permalink"]
```
The columns 'thumbnail', 'secure_thumbnail', 'permalink' contain only links and do not contribute to the prediction task, hence they will be removed.
```python
# Delete link columns
for column in links_columns:
    df = df.drop(columns=[column])
```
## 2.13 Id columns
```python
id_columns = ["title","seller_id","id","parent_item_id","category_id","site_id" ]
```
We will analyze the unique values of the columns 'title', 'seller_id', 'id', 'parent_item_id', 'category_id' and 'site_id' to gain insights into its distribution and frequency of occurrence.

--------
```python
for i in id_columns:
    print(f"Unique values in {i}: {len(df[i].unique())}")
```
*Output:*
```
Unique values in title: 89008
Unique values in seller_id: 33281
Unique values in id: 90000
Unique values in parent_item_id: 69311
Unique values in category_id: 10491
Unique values in site_id: 1
```
--------
All columns exhibited a large number of categories, except for site_id, which had only one unique category. Additionally, it is worth noting that seller_id values are typically assigned in the order that sellers register on the platform. Regarding other ID columns such as id, parent_item_id, category_id, and site_id, their generation function is dependent on internal policies of the company and may or may not be related to the condition of the product being new or used.

```python
# Delete id columns
for column in id_columns:
    df = df.drop(columns=[column])
```
## 2.14 Tiding up
```python
COLUMNS_ORDER = [
    # product info
    "price","price_in_usd","initial_quantity","sold_quantity","available_quantity","warranty_info",
    # buying mode
    "mode_buy_it_now","mode_classified","mode_auction",
    # payment info
    "cash_payment","card_payment","bank_payment","mercadopago_payment","agree_with_buyer_payment",
    # shipping info
    "shipping_me1","shipping_me2","shipping_custom","shipping_not_specified","free_shipping","local_pick_up",
    # seller info
    "buenos_aires_seller","listing_free","listing_bronze","listing_silver","listing_gold",
    # publication info
    "start_week","start_day","stop_week","stop_day","days_active","is_active",
    "num_pictures","automatic_relist","dragged_bids_or_visits"
    ] 
```
```python
# Organize dataframe columns
    df = df.reindex(columns=COLUMNS_ORDER)
```
```python
# Transforms boolean column
bool_columns = df.select_dtypes(include=bool).columns
df[bool_columns] = df[bool_columns].astype(int)
```
An essential step forward: The function preprocess_dataset() has been coded and resides within src/features/data_preprocessing.py. This paves the way for streamlined data preprocessing in future endeavors.
# 3. Data Exploratory Analisys (DAE)