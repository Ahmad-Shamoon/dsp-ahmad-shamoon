import pandas as pd

def clean_and_prepare_data(df: pd.DataFrame,
                           continuous_feats: list[str],
                           categorical_feats: list[str]) -> pd.DataFrame:
    """
    Cleans missing values for numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    continuous_feats : list of str
        Continuous feature column names.
    categorical_feats : list of str
        Categorical feature column names.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    for c in continuous_feats:
        df[c] = df[c].fillna(df[c].median())
    for c in categorical_feats:
        df[c] = df[c].fillna(df[c].mode()[0])
    return df
