import pandas as pd

def apply_column_renames(df: pd.DataFrame, rename_mapping: dict) -> pd.DataFrame:
    """
    Applies a dictionary of column renames to a Pandas DataFrame.

    This function accepts a DataFrame and a mapping, and returns a new DataFrame
    with the columns renamed, leaving the original DataFrame untouched.

    Args:
        df (pd.DataFrame): The input DataFrame containing the columns to rename.
        rename_mapping (dict): A dictionary mapping {'old_name': 'new_name'}.
                               For example: {'Annual Income (k$)': 'Annual Income'}

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns renamed.
    """
    # Use inplace=False and return the result to ensure a new DataFrame is created,
    # preventing potential SettingWithCopyWarning issues.
    return df.rename(columns=rename_mapping, inplace=False)

# Other data processing functions here (e.g., handle_missing_values)
# as the project grows.
