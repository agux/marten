def layer_spec_to_list(spec):
    if spec is None:
        return []

    w, d = spec[0], spec[1]
    return [w] * d

def select_topk_features(df, ranked_features, k):
    """
    process df (dataframe): keep only the 'ds', 'y' columns, and columns with names in top k elements in the ranked_features list.
    """
    top_k_features = ranked_features[:k]
    columns_to_keep = ['ds', 'y'] + top_k_features
    return df[columns_to_keep]
