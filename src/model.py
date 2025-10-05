import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib

def train_linear_regression(df, output_dir="outputs"):
    # Split features and target
    X = df.drop("close", axis=1).values
    y = df["close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Linear Regression - MSE: {mse:.2f}, R²: {r2:.2f}")

    # Save model
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    joblib.dump(model, f"{output_dir}/models/linear_model.pkl")
    print("✅ Model saved to outputs/models/linear_model.pkl")

    # Plot predictions
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction (Linear Regression)")
    plt.legend()
    plt.savefig(f"{output_dir}/plots/prediction_plot.png")
    plt.show()

    return model
