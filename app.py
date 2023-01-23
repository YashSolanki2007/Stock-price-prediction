import streamlit as st
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import pandas_ta as pta
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression



st.title("Stock Analyser")
st.markdown("""
### The one place for all your stock needs
We not only provide several technical indicators to understand the movement of the stock better but also 
we have a Neural Network to predict the future price.
""")

ticker = st.text_input('Stock Ticker')

if ticker == "":
    ticker = "GOOG"


st.write('Ticker entered: ', ticker)


# Showing the close prices line chart
df = si.get_data(ticker)


# Extracting the usefull parts of the data
open_prices = df['open']
close_prices = df['close']
volumes = df['volume']
high_prices = df['high']
low_prices = df['low']
DATA_LEN = 300


close_prices = close_prices[len(
    close_prices) - DATA_LEN:len(close_prices)].to_list()
open_prices = open_prices[len(open_prices) -
                          DATA_LEN:len(open_prices)].to_list()
volumes = volumes[len(volumes) - DATA_LEN:len(volumes)].to_list()
high_prices = high_prices[len(high_prices) -
                          DATA_LEN:len(high_prices)].to_list()
low_prices = low_prices[len(low_prices) - DATA_LEN:len(low_prices)].to_list()

close_for_calc = df['close']
close_for_calc = close_for_calc[len(
    close_for_calc) - DATA_LEN:len(close_for_calc)]


st.text("");st.text("");st.text("")

st.markdown("## Technical Indicators")

# Close prices plot
fig = plt.figure()
plt.title(f"Close prices for: {ticker} currently at {round(close_prices[len(close_prices) - 1], 2)}", fontsize=15)
plt.xlabel("Days after", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.plot(close_prices, label='Close Price')
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.markdown("***")


# RSI
relative_strength_indexs = pta.rsi(close_for_calc, length = 14)
relative_strength_indexs = relative_strength_indexs.to_list()

fig = plt.figure()
plt.plot(relative_strength_indexs, label='Share Price')
plt.title(f"Relative Strength Index (RSI) every 14 days currently at: {round(relative_strength_indexs[-1], 2)}", fontsize=17)
plt.xlabel("Number of days after", fontsize=15)
plt.ylabel("RSI Value", fontsize=15)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.text("")
st.markdown("In the given chart an RSI > 70 indicates an overbought share and an RSI < 30 indicates an oversold share.")

st.markdown("***")

# Bollinger Bands
close_avg = close_for_calc.rolling(5).mean().to_list()
standard_deviations = close_for_calc.rolling(5).std().to_list()

upper_bollinger_band = []
lower_bollinger_band = []

for i in range(len(standard_deviations)):
    upper_bound = close_avg[i] + (standard_deviations[i] * 2)
    lower_bound = close_avg[i] - (standard_deviations[i] * 2)

    upper_bollinger_band.append(upper_bound)
    lower_bollinger_band.append(lower_bound)


fig = plt.figure()
plt.plot(close_avg, label='Simple Moving Average')
plt.plot(upper_bollinger_band, label='Upper Band')
plt.plot(lower_bollinger_band, label='Lower Band')
plt.plot(close_prices, 'r', label='Close Price')
plt.title("Bollinger Bands with standard deviation of 2 at a time perios of 5 days", fontsize=17)
plt.xlabel("Number of days after", fontsize=15)
plt.ylabel("Price", fontsize=15)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.markdown("***")


# OBV
on_balance_volumes = []
obv = 0

on_balance_volumes.append(obv)

for i in range(1, len(volumes)):
    if close_prices[i] > close_prices[i - 1]:
        obv += volumes[i]
        on_balance_volumes.append(obv)

    elif close_prices[i] < close_prices[i - 1]:
        obv -= volumes[i]
        on_balance_volumes.append(obv)

    else:
        obv += 0
        on_balance_volumes.append(obv)

NUM_OF_DAYS_2 = 5
obv_df = pd.DataFrame(on_balance_volumes)
obv_sma = obv_df.rolling(NUM_OF_DAYS_2).mean()

fig = plt.figure()
plt.plot(on_balance_volumes, label='OBV')
plt.plot(obv_sma, label='Simple Moving Average for OBV')
plt.title("OBV :- On Balance Volume", fontsize=17)
plt.xlabel("Number of days after", fontsize=15)
plt.ylabel("OBV", fontsize=15)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.markdown("***")

# MACD
ema12 = close_for_calc.ewm(span=12, adjust=False).mean()
ema26 = close_for_calc.ewm(span=26, adjust=False).mean()

macd = ema12 - ema26

# Signal line of macd
signal = macd.ewm(span=9, adjust=False).mean()

fig = plt.figure()
plt.plot(macd.to_list(), label='MACD')
plt.plot(signal.to_list(), label='Signal')
plt.title("Moving Average Convergence Divergence", fontsize=17)
plt.ylabel("MACD", fontsize=15)
plt.xlabel("Days After", fontsize=15)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.markdown("***")


# Momentum
MOMENTUM_PERIOD = 10

momentum_values = []
for i in range(MOMENTUM_PERIOD, len(close_prices)):
    curr_close_price = close_prices[i]
    period_start_close_price = close_prices[i - MOMENTUM_PERIOD]
    momentum_values.append(curr_close_price - (period_start_close_price))

momentum_sum = 0
for i in range(len(momentum_values)):
    momentum_sum += momentum_values[i]

avg_momentum = momentum_sum / len(momentum_values)

fig = plt.figure()
plt.plot(momentum_values, label='Momentum Values')
plt.title(f"Momentum of stock over: {MOMENTUM_PERIOD} days", fontsize=17)
plt.ylabel("Momentum", fontsize=15)
plt.xlabel("Days After", fontsize=15)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)


# Resistance and support lines
pivot_points = []

for i in range(len(close_for_calc)):
    if i == 0:
        pivot_points.append(float("nan"))
    else:
        prev_high = high_prices[i - 1]
        prev_low = low_prices[i - 1]
        prev_close = close_prices[i - 1]

        pivot_point = (prev_high + prev_low + prev_close) / 3
        pivot_points.append(pivot_point)

resistance_1 = []
support_1 = []
resistance_2 = []
support_2 = []

for i in range(len(pivot_points)):
    if i == 0:
        resistance_1.append(float("nan"))
        support_1.append(float("nan"))
    else:
        r1 = (2 * pivot_points[i]) - low_prices[i - 1]
        s1 = (2 * pivot_points[i]) - high_prices[i - 1]

        r2 = (pivot_points[i] - s1) + r1
        s2 = pivot_points[i] - (r1 - s1)

        resistance_1.append(r1)
        support_1.append(s1)
        resistance_2.append(r2)
        support_2.append(s2)



fig = plt.figure()
plt.plot(close_prices, label='Close Price')
plt.plot(resistance_1, label='Resistance (first)')
plt.plot(support_1, label='Support (first)')
plt.plot(resistance_2, label='Resistance (second)')
plt.plot(support_2, label='Support (second)')
plt.xlabel("Days After", fontsize=15)
plt.ylabel("Price", fontsize=15)
plt.title("Support And Resistance", fontsize=17)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.text("")
st.markdown("To get a better view of the support and resistances please zoom into the chart.")

st.markdown("***")


st.markdown("## Linear Regression Predictions")


# Linear Regression code
dataset = close_prices

dataset = np.array(dataset)
training = len(dataset)
dataset = np.reshape(dataset, (dataset.shape[0], 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training), :]
print(len(train_data))
# prepare feature and labels
x_train = []
y_train = []
prediction_days = 60

for i in range(prediction_days, len(train_data)):
    x_train.append(train_data[i-prediction_days:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))

reg = LinearRegression().fit(x_train, y_train)
x_tomm = close_prices[len(close_prices) - prediction_days:len(close_prices)]
x_tomm = np.array(x_tomm)
x_tomm = scaler.transform(x_tomm.reshape(1, -1))
prediction = reg.predict(x_tomm)
prediction = scaler.inverse_transform(prediction.reshape(1, -1))
st.markdown(f"#### Tomorrow's prediction for: {ticker} is {round(prediction[0][0], 2)}")


st.markdown("***")

# Future Predictions
FUTURE_DAYS = st.text_input('Enter number of future days (Recommended not to go over 15)')

try:
    FUTURE_DAYS = int(FUTURE_DAYS)
except:
    FUTURE_DAYS = 10


predicted_prices = []
tot_prices = list(close_prices)

for i in range(FUTURE_DAYS):
    x_prices = tot_prices[len(tot_prices) - prediction_days: len(tot_prices)]
    x_prices = np.array(x_prices)
    x_prices = scaler.transform(x_prices.reshape(1, -1))
    prediction = reg.predict(x_prices)
    prediction = scaler.inverse_transform(prediction.reshape(1, -1))
    tot_prices.append(prediction)
    predicted_prices.append(prediction)



tot_prices = np.array(tot_prices)
predicted_prices = np.array(predicted_prices)

tot_prices = np.reshape(tot_prices, (tot_prices.shape[0]))
predicted_prices = np.reshape(predicted_prices, (predicted_prices.shape[0]))

print(len(close_prices))
print(len(tot_prices))


fig = plt.figure()
plt.plot(tot_prices, label='Predicted Future Prices')
plt.plot(close_prices, label='Current Prices')
plt.xlabel("Days After", fontsize=15)
plt.ylabel("Price", fontsize=15)
plt.title("Future Price Predictions", fontsize=17)
plt.legend()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)
