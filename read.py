from typing import Any

import pandas as pd
from numpy import ndarray
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Read csv file
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Split a given dataframe
def process_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[tuple[ndarray, ndarray], tuple[ndarray,ndarray], tuple[ndarray,ndarray], ndarray]:
    # Drop columns with too many missing values and problematic categorical columns
    cols_to_drop = ['hc', 'model']
    train_clean = train_df.drop(columns=cols_to_drop)
    x_test = test_df.drop(columns=cols_to_drop)

    print(train_clean.head())

    # Fill missing values for important columns
    num_cols = ['co', 'nox', 'hcnox', 'ptcl', 'urb_cons', 'exturb_cons']
    for col in num_cols:
        train_median_value = train_clean[col].median()
        train_clean[col] = train_clean[col].fillna(train_median_value)
        test_median_value = x_test[col].median()
        x_test[col] = x_test[col].fillna(test_median_value)

    # Replace categorical features with one-hot-encoded columns
    cat_cols = ['brand', 'car_class', 'range', 'fuel_type', 'hybrid', 'grbx_type_ratios']
    train_clean = pd.get_dummies(train_clean, columns=cat_cols)
    x_test = pd.get_dummies(x_test, columns=cat_cols)
    x_test = x_test.reindex(columns=train_clean.columns, fill_value=0)  # Ensure train and test have same columns

    # Define training target
    y_train_nosplit = train_clean['co2']

    # Remove unnecessary features from training data
    x_train = train_clean.drop(columns=['id', 'co2'])

    # Remove unnecessary feature from test
    x_test = x_test.drop(columns=['id', 'co2'])

    # Scale numerical features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Split data for validation
    x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_train_nosplit, test_size=0.2, random_state=31)
    return (x_train_scaled, y_train_nosplit), (x_train, y_train), (x_val, y_val), x_test_scaled,
