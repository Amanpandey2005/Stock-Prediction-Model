import pandas as pd

def load_data(csv_path):
    """
    Load CSV data and preprocess
    """
    df = pd.read_csv(csv_path)
    
    # Ensure Date column exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    
    # Keep only close price
    if "close" not in df.columns:
        raise ValueError("CSV must have 'close' column")
    
    return df[["close"]]
