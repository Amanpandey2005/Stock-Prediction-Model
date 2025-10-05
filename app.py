import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# --- MOCK BACKEND FUNCTIONS ---
# Replace these with your actual model functions from main.py
# For demonstration, they will generate some dummy data.

def load_and_predict(ticker, start_date, end_date, prediction_days):
    """
    This is a mock function. In a real scenario, this would:
    1. Download the stock data for the given ticker and date range.
    2. Call create_lag_features().
    3. Call train_linear_regression().
    4. Predict the future 'prediction_days'.
    5. Return historical and forecasted dataframes.
    """
    # Create dummy historical data
    hist_dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date))
    hist_prices = pd.Series(range(len(hist_dates))) * 2 + 150 + (pd.Series(range(len(hist_dates))).apply(lambda x: x*0.1) * pd.Series(range(len(hist_dates))).apply(lambda x: x*0.1))
    historical_df = pd.DataFrame({'Date': hist_dates, 'Close': hist_prices})

    # Create dummy forecasted data
    last_date = hist_dates[-1]
    last_price = hist_prices.iloc[-1]
    forecast_dates = pd.to_datetime(pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days))
    forecast_prices = pd.Series(range(prediction_days)) * 1.5 + last_price
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': forecast_prices})

    return historical_df, forecast_df

# --- STREAMLIT UI CODE ---

# Set page configuration
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.title("📈 Stock Prediction")
    st.write("Configure the prediction model.")

    # Input for stock ticker
    ticker_symbol = st.text_input("**Enter Stock Ticker**", "AAPL")

    # Date range for historical data
    today = date.today()
    start_date = st.date_input("**Historical Start Date**", today - timedelta(days=365*2))
    end_date = st.date_input("**Historical End Date**", today)

    # Slider for prediction days
    prediction_days = st.slider("**Days to Predict**", min_value=7, max_value=90, value=30)

    # Action button
    predict_button = st.button("🚀 Predict Future Price", type="primary")


# --- Main Display Area ---
st.title(f"{ticker_symbol} Stock Price Prediction")

if predict_button:
    # Show a spinner while the model is running
    with st.spinner(f"Training model and predicting for {ticker_symbol}..."):
        # Call the backend function to get data
        historical_data, forecast_data = load_and_predict(ticker_symbol, start_date, end_date, prediction_days)

    st.success("Prediction complete!")

    # --- Display Key Metrics ---
    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    
    current_price = historical_data['Close'].iloc[-1]
    next_day_price = forecast_data['Predicted Close'].iloc[0]
    forecast_high = forecast_data['Predicted Close'].max()

    col1.metric("Current Price", f"${current_price:,.2f}")
    col2.metric("Predicted Price (Next Day)", f"${next_day_price:,.2f}", f"{((next_day_price - current_price) / current_price) * 100:.2f}%")
    col3.metric(f"Forecast High (+{prediction_days} Days)", f"${forecast_high:,.2f}")


    # --- Display Prediction Chart ---
    st.subheader("Prediction Chart")
    
    # Create the figure
    fig = px.line(historical_data, x='Date', y='Close', labels={'Close': 'Closing Price'})
    
    # Add the forecast line
    fig.add_scatter(x=forecast_data['Date'], y=forecast_data['Predicted Close'], mode='lines', name='Forecast', line=dict(dash='dash', color='green'))
    
    # Update layout for a cleaner look
    fig.update_layout(
        title=f'{ticker_symbol} Historical vs. Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        legend_title='Legend'
    )
    
    st.plotly_chart(fig, use_container_width=True)


    # --- Display Data Table ---
    with st.expander("View Forecast Data Table"):
        st.dataframe(forecast_data.style.format({"Predicted Close": "${:,.2f}"}))
else:
    st.info("Please configure the parameters on the left and click 'Predict'.")