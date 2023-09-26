import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

st.title('S&P 500 Investment Simulator')

# Function to calculate RSI
def calculate_RSI(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

# Download S&P 500 data
symbol = "^GSPC"
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
data = yf.download(symbol, start="2000-01-01", end=end_date)

# Feature engineering
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['Volatility'] = data['Close'].rolling(window=50).std()
data['RSI'] = calculate_RSI(data, 14)
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
data = data.dropna()

# Prepare for training
features = ['Open', 'High', 'Low', 'Volume', '50_MA', '200_MA', 'Volatility', 'RSI', 'EMA']
X = data[features]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Current S&P 500 value
current_sp500 = data['Close'].iloc[-1]
st.write(f"Current S&P 500 index value: {current_sp500}")

# Input for investment amount
investment = st.number_input('Enter your investment amount in €:', min_value=1000.0, max_value=100000.0, value=10000.0)

# Predict function
def predict_future(days, investment, volatility):
    future_date = datetime.datetime.now() + datetime.timedelta(days=days)
    last_data = pd.DataFrame(data[features].iloc[-1].values.reshape(1, -1), columns=features)
    future_price = model.predict(last_data)[0]
    future_price += np.random.normal(0, volatility)
    
    # Calculate investment outcome
    investment_multiplier = future_price / current_sp500
    future_investment = investment * investment_multiplier
    
    return future_date, future_price, future_investment

# Display predictions and investment outcomes
six_months = predict_future(180, investment, 50)
one_year = predict_future(365, investment, 100)
five_years = predict_future(1825, investment, 500)

st.write(f"In 6 months, the predicted S&P 500 index value is {six_months[1]:.2f} and your investment could be €{six_months[2]:.2f}.")
st.write(f"In 1 year, the predicted S&P 500 index value is {one_year[1]:.2f} and your investment could be €{one_year[2]:.2f}.")
st.write(f"In 5 years, the predicted S&P 500 index value is {five_years[1]:.2f} and your investment could be €{five_years[2]:.2f}.")

# Create a matplotlib figure
fig, ax1 = plt.subplots()

# Create lists to store the data
dates = [datetime.datetime.now(), six_months[0], one_year[0], five_years[0]]
sp500_values = [current_sp500, six_months[1], one_year[1], five_years[1]]
investment_values = [investment, six_months[2], one_year[2], five_years[2]]

# Plot S&P 500 index values
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P 500 Value', color='tab:blue')
ax1.plot(dates, sp500_values, label='S&P 500', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis to plot investment values
ax2 = ax1.twinx()
ax2.set_ylabel('Investment Value in €', color='tab:green')
ax2.plot(dates, investment_values, label='Investment', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Show legends and grid
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9))
ax1.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)
