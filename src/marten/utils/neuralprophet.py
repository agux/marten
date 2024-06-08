import torch
import re

def layer_spec_to_list(spec):
    if spec is None:
        return []

    w, d = spec[0], spec[1]
    return [w] * d

def select_topk_features(df, ranked_features, k):
    """
    process df (dataframe): keep only the 'ds', 'y' columns, and columns with names 
    in top k elements in the ranked_features list.
    """
    top_k_features = ranked_features[:int(k)]
    columns_to_keep = ['ds', 'y'] + top_k_features
    return df[columns_to_keep]

def select_device(accelerator, util_threshold=80, vram_threshold=80):
    return (
        "gpu"
        if accelerator
        and torch.cuda.utilization() < util_threshold
        and torch.cuda.memory_usage() < vram_threshold
        else None
    )

def set_yhat_n(df):
    # Extract column names
    columns = df.columns

    # Filter columns that start with "yhat"
    yhat_columns = [col for col in columns if col.startswith('yhat')]

    # Sort columns based on the numerical part in ascending order
    yhat_columns_sorted = sorted(yhat_columns, key=lambda x: int(re.search(r'\d+', x).group()))

    # Initialize yhat_n with the values from the smallest yhat column
    df['yhat_n'] = df[yhat_columns_sorted[0]]

    # Iterate over the remaining yhat columns and fill in null/NA values in yhat_n
    for col in yhat_columns_sorted[1:]:
        df['yhat_n'].fillna(df[col], inplace=True)

def set_forecast_columns(forecast):
    # List of columns to keep
    columns_to_keep = ["ds", "y"]

    # Add columns that match the pattern 'yhat'
    columns_to_keep += [
        col
        for col in forecast.columns
        if col.startswith("yhat") and "%" not in col
    ]

    # Remove columns not in the list of columns to keep
    forecast.drop(
        columns=[col for col in forecast.columns if col not in columns_to_keep],
        inplace=True,
    )

    set_yhat_n(forecast)
