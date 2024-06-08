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

def max_yhat_col(df):
    # Extract column names
    columns = df.columns

    # Filter columns that start with "yhat"
    yhat_columns = [col for col in columns if col.startswith('yhat')]

    # Extract numbers from the column names and find the maximum
    max_number = max(int(re.search(r'\d+', col).group()) for col in yhat_columns)

    # Construct the column name with the largest number
    return f'yhat{max_number}'
