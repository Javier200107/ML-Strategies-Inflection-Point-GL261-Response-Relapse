from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
 
def col_standardisation(df, cols):
    """
    Standardise the columns of the dataset
    """
    scaler = StandardScaler()
    scaler.fit(df[cols])
    df[cols] = scaler.transform(df[cols])
    return df

 
def col_minmax_scaling(df, cols):
    """
    MinMax scale the columns of the dataset
    """
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

 
def col_normalisation(df, cols):
    """
    Normalise the columns of the dataset
    """
    df[cols] = normalize(df[cols], norm='l2', axis=0)
    return df

 
def label_encoding(df, cols):
    """
    Label encode the columns of the dataset
    """
    encoder = LabelEncoder()
    for col in cols:
        df[col] = encoder.fit_transform(df[col])
    return df

 
def one_hot_encoding(df, cols):
    """
    One hot encode the columns of the dataset
    """
    df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
    return df

 
def delete_rows_with_missing_values(df):
    """
    Delete rows with missing values
    """
    df.dropna(inplace=True)
    return df

 
def imputate_row_missing_values_with_median(df, cols):
    """
    Imputate missing values
    """
    for col in cols:
        df[col] = df[col].fillna(df[col].median())
    return df

 
def imputate_row_missing_values_with_mode(df, cols):
    """
    Imputate missing values
    """
    for col in cols:
        df[col] = df[col].fillna(df[col].mode())
    return df

 
def df_to_numeric(df, cols):
    """
    Convert columns to float
    """
    df.loc[:, cols] = df.loc[:, cols].astype(float)
    return df

 
def df_to_int(df, cols):
    """
    Convert columns to float
    """
    df.loc[:, cols] = df.loc[:, cols].astype(int)
    return df
 
def df_to_string(df, cols):
    """
    Convert columns to string
    """
    df.loc[:, cols] = df.loc[:, cols].astype(str)
    return df
