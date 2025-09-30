import pandas as pd
from pathlib import Path

def load_data(filename="Mall_Customers.csv") -> pd.DataFrame:
    """
    Loads the specified CSV data file from the relative 'data' directory.

    Args:
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded data as a Pandas DataFrame.
    """
    # Get the root directory (one level up from 'segmentum')
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / filename
    return pd.read_csv(data_path)