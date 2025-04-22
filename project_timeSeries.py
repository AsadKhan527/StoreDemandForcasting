import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🛒 Store Item Demand Forecasting App")

# Create tabs
tab1, tab2 = st.tabs(["📄 Project Summary", "📈 Forecasting App"])

with tab1:
    st.header("Project Overview")
    st.markdown("""
        ## 🛒 Store-Item Demand Forecasting

    Welcome to the **Store-Item Demand Forecasting App**! This interactive tool is designed to help you analyze and forecast item-level sales across various stores using time series techniques and deep learning models.

    ---

    ### 🎯 **Objective**
    The goal of this project is to **predict daily sales** for individual store-item combinations. Accurate forecasts can significantly enhance:

    - 📦 **Inventory Management**
    - 🚚 **Supply Chain Optimization**
    - 💰 **Revenue Planning**

    ---

    ### 📂 **Dataset Overview**
    The dataset consists of historical daily sales records with the following structure:

    | Column   | Description                     |
    |----------|---------------------------------|
    | `date`   | Date of sale (daily granularity)|
    | `store`  | Store ID                        |
    | `item`   | Item ID                         |
    | `sales`  | Number of items sold            |

    - **Time Range:** Spanning multiple years on a daily basis  
    - **Granularity:** Per store, per item, per day

    ---

    ### 🧠 **Modeling Approach**

    #### 🔍 **Exploratory Analysis**
    - Interactive line plots using Plotly
    - Seasonal decomposition of sales patterns
    - Rolling statistics & stationarity checks (ADF test)

    #### 📈 **Forecasting Models**
    - **SARIMA**: Traditional statistical model for seasonal data
    - **LSTM**: Deep learning model trained on sequences of past sales

    #### ⚙️ **Key Steps**
    - Data filtering by store & item
    - Time series decomposition
    - Stationarity testing & differencing
    - Model training and prediction
    - Visualization with confidence intervals

    ---

    ### 📊 **Output Features**
    - Interactive time series plots for actual and forecasted sales
    - Seasonal breakdown (trend, seasonality, residual)
    - 90-day sales forecast using **SARIMA** and **LSTM**
    - Confidence intervals for SARIMA predictions

    ---

    ### 🛠️ **Technologies Used**
    - **Streamlit** for the interactive dashboard
    - **Statsmodels** for time series decomposition & SARIMA
    - **TensorFlow / Keras** for building LSTM model
    - **Plotly** for dynamic, zoomable plots
    - **Pandas / NumPy** for data manipulation

    ---

    ### ✅ **Why This App?**
    Whether you're a data scientist, supply chain analyst, or a retail manager, this app provides:

    - 📉 Easy visualization of sales trends
    - 🧠 Smart modeling for better forecasting
    - ⚡ Intuitive, fast, and interactive experience

    ---
    
    **Built with ❤️ by a data enthusiast.**
    """)

with tab2:
    uploaded_file = st.sidebar.file_uploader("Upload dataset (train.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload the dataset used in the original notebook to proceed.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

    def expand_df(df):
        data = df.copy()
        data['day'] = data.date.dt.day
        data['month'] = data.date.dt.month
        data['year'] = data.date.dt.year
        data['dayofweek'] = data.date.dt.dayofweek
        return data

    df = expand_df(df)

    st.sidebar.subheader("Filter Data")
    store = st.sidebar.selectbox("Store", sorted(df['store'].unique()))
    item = st.sidebar.selectbox("Item", sorted(df['item'].unique()))
    filtered_df = df[(df['store'] == store) & (df['item'] == item)].copy()
    filtered_df.set_index('date', inplace=True)
    filtered_df.sort_index(inplace=True)

    st.subheader(f"Data Preview: Store {store}, Item {item}")
    st.write(filtered_df.head())

    # Lineplot - Plotly
    st.subheader("Sales Over Time")
    fig1 = px.line(filtered_df, x=filtered_df.index, y='sales', title="Sales Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # Decomposition - Plotly
    st.subheader("Seasonal Decomposition")
    result = sm.tsa.seasonal_decompose(filtered_df['sales'], model='additive', period=365)

    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])

    fig2.add_trace(go.Scatter(x=result.observed.index, y=result.observed, name="Observed"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=result.trend.index, y=result.trend, name="Trend"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig2.add_trace(go.Scatter(x=result.resid.index, y=result.resid, name="Residual"), row=4, col=1)

    fig2.update_layout(height=800, title="Seasonal Decomposition", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Rolling stats with Plotly
    def plot_rolling_stats_plotly(ts, window=12):
        rolmean = ts.rolling(window).mean()
        rolstd = ts.rolling(window).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts, name='Original'))
        fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean, name='Rolling Mean'))
        fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd, name='Rolling Std'))
        fig.update_layout(title="Rolling Mean & Std Dev", height=400)
        return fig

    st.subheader("Rolling Statistics for Stationarity")
    st.plotly_chart(plot_rolling_stats_plotly(filtered_df['sales']), use_container_width=True)

    def dickey_fuller_test(timeseries):
        dftest = adfuller(timeseries, autolag='AIC')
        result = {
            'Test Statistic': dftest[0],
            'p-value': dftest[1],
            'Lags Used': dftest[2],
            'Number of Observations': dftest[3],
            'Critical Values': dftest[4]
        }
        return result

    st.write("Dickey-Fuller Test:")
    st.write(dickey_fuller_test(filtered_df['sales']))

    first_diff = filtered_df['sales'].diff().dropna()
    st.subheader("Differenced Series")
    st.plotly_chart(plot_rolling_stats_plotly(first_diff), use_container_width=True)
    st.write(dickey_fuller_test(first_diff))

    # Forecasting - SARIMA
    st.subheader("Forecasting with SARIMA")
    if st.button("Run SARIMA Forecast"):
        train = filtered_df['sales'][:-90]
        test = filtered_df['sales'][-90:]

        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit()

        forecast = results.get_forecast(steps=90)
        forecast_df = forecast.predicted_mean
        conf_int = forecast.conf_int()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig3.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig3.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df, name="Forecast", line=dict(color='green')))
        fig3.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 0],
            fill=None,
            mode='lines',
            line_color='lightgrey',
            name='Lower CI'
        ))
        fig3.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 1],
            fill='tonexty',
            mode='lines',
            line_color='lightgrey',
            name='Upper CI'
        ))
        fig3.update_layout(title="SARIMA Forecast", height=500)
        st.plotly_chart(fig3, use_container_width=True)

    # LSTM Forecast
    st.subheader("Forecasting with LSTM")

    def create_lstm_dataset(series, look_back=30):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back])
        return np.array(X), np.array(y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(filtered_df[['sales']].values)

    look_back = 30
    X_lstm, y_lstm = create_lstm_dataset(scaled_data, look_back)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    X_train_lstm = X_lstm[:-90]
    y_train_lstm = y_lstm[:-90]
    X_test_lstm = X_lstm[-90:]
    y_test_lstm = y_lstm[-90:]

    if st.button("Run LSTM Forecast"):
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)

        predictions = model.predict(X_test_lstm)
        predictions_rescaled = scaler.inverse_transform(predictions)

        forecast_dates = filtered_df.index[-90:]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=filtered_df.index[-180:], y=filtered_df['sales'].values[-180:], name="Actual"))
        fig4.add_trace(go.Scatter(x=forecast_dates, y=predictions_rescaled.flatten(), name="LSTM Forecast", line=dict(color='orange')))
        fig4.update_layout(title="LSTM Forecast vs Actual", height=500)
        st.plotly_chart(fig4, use_container_width=True)
