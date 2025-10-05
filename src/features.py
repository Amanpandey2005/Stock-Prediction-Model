def create_lag_features(df, n_lags=10):
    """
    Create lag features for past n days
    """
    data = df.copy()
    for i in range(1, n_lags+1):
        data[f"lag_{i}"] = data["close"].shift(i)
    data.dropna(inplace=True)
    return data
