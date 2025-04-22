import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🛒 Store Item Demand Forecasting App")

# Create tabs
tab1, tab2 = st.tabs(["📄 Project Summary", "📈 Forecasting App"])

with tab1:
    st.header("Project Overview")
    st.markdown("""
    This project aims to forecast the future **item-level sales** for multiple stores using historical data.
    
    ### 🔍 Problem Statement:
    Predict future demand for each store-item combination to optimize inventory and supply chain management.

    ### 📦 Dataset Information:
    - **Date Range**: Daily sales from multiple stores and items
    - **Columns**: `date`, `store`, `item`, `sales`

    ### 🧠 Methodology:
    - Data Filtering by Store & Item
    - Time Series Decomposition
    - Dickey-Fuller Test for Stationarity
    - SARIMA Modeling for Forecasting
    - Visual Forecast Evaluation

    ### 📊 Output:
    - 90-Day Forecast with Confidence Interval
    - Visual breakdown of seasonality and trend
    """)

with tab2:
    # Upload CSV
    uploaded_file = st.sidebar.file_uploader("/Users/asadkhan/Desktop/PROJECTV_369/train.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload the dataset used in the original notebook to proceed.")
        st.stop()

    # Convert date
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

    # Expand features
    def expand_df(df):
        data = df.copy()
        data['day'] = data.date.dt.day
        data['month'] = data.date.dt.month
        data['year'] = data.date.dt.year
        data['dayofweek'] = data.date.dt.dayofweek
        return data

    df = expand_df(df)

    # Filters
    st.sidebar.subheader("Filter Data")
    store = st.sidebar.selectbox("Store", sorted(df['store'].unique()))
    item = st.sidebar.selectbox("Item", sorted(df['item'].unique()))
    filtered_df = df[(df['store'] == store) & (df['item'] == item)].copy()
    filtered_df.set_index('date', inplace=True)
    filtered_df.sort_index(inplace=True)

    # Show basic info
    st.subheader(f"Data Preview: Store {store}, Item {item}")
    st.write(filtered_df.head())

    # Lineplot
    st.subheader("Sales Over Time")
    fig1, ax1 = plt.subplots()
    sns.lineplot(x=filtered_df.index, y='sales', data=filtered_df, ax=ax1)
    st.pyplot(fig1)

    # Decomposition
    st.subheader("Seasonal Decomposition")
    result = sm.tsa.seasonal_decompose(filtered_df['sales'], model='additive', period=365)
    fig2 = result.plot()
    fig2.set_size_inches(14, 10)
    st.pyplot(fig2)

    # Rolling stats
    def roll_stats(timeseries, window=12):
        rolmean = timeseries.rolling(window).mean()
        rolstd = timeseries.rolling(window).std()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timeseries, color='blue', label='Original')
        ax.plot(rolmean, color='red', label='Rolling Mean')
        ax.plot(rolstd, color='black', label='Rolling Std')
        ax.legend()
        ax.set_title("Rolling Mean & Std Dev")
        return fig

    st.subheader("Rolling Statistics for Stationarity")
    st.pyplot(roll_stats(filtered_df['sales']))

    # Dickey-Fuller Test
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

    # First differencing
    first_diff = filtered_df['sales'].diff().dropna()
    st.subheader("Differenced Series")
    st.pyplot(roll_stats(first_diff))
    st.write(dickey_fuller_test(first_diff))

    # Forecasting
    st.subheader("Forecasting with SARIMA")
    if st.button("Run Forecast"):
        train = filtered_df['sales'][:-90]
        test = filtered_df['sales'][-90:]

        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
        results = model.fit()

        forecast = results.get_forecast(steps=90)
        forecast_df = forecast.predicted_mean
        conf_int = forecast.conf_int()

        fig3, ax3 = plt.subplots(figsize=(14, 5))
        train.plot(ax=ax3, label='Train')
        test.plot(ax=ax3, label='Test')
        forecast_df.plot(ax=ax3, label='Forecast', color='green')
        ax3.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.1)
        ax3.legend()
        st.pyplot(fig3)
