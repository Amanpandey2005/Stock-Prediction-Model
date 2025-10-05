from src.dataloader import load_data
from src.features import create_lag_features
from src.model import train_linear_regression

if __name__ == "__main__":
    csv_path = "C:\\Users\\amanp\\OneDrive\\Documents\\Github\\Stock Prediction\\Data\\Raw\\aapl_2014_2023.csv"  # Tumhara Kaggle CSV

    # Step 1: Load data
    df = load_data(csv_path)

    # Step 2: Create features
    df_features = create_lag_features(df, n_lags=10)

    # Step 3: Train and evaluate model
    model = train_linear_regression(df_features)
