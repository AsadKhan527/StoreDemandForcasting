**🛍️ Store Item Demand Forecasting App**

Live Demo: https://storedemandforcastingts.streamlit.app

A powerful and interactive web application for forecasting store item demand using advanced time series models like SARIMA, ARIMA, Exponential Smoothing, ARCH/GARCH, and LSTM. Built with Streamlit and Plotly, this app enables you to analyze historical sales data, visualize trends, and generate accurate 90-day forecasts.

**📑 Table of Contents**
Project Overview
Features
Usage
Dataset
Methodology
Technologies Used

**📊 Project Overview**
The Store Item Demand Forecasting App is designed to predict future item-level sales across multiple stores, aiding businesses in smarter inventory management and operational efficiency.

**🎯 Problem Statement**
The app forecasts demand for each store-item combination over a 90-day horizon, helping minimize stockouts and overstocking.

✅ Key Objectives
Preprocess and filter data by store and item
Perform time series decomposition to reveal trends and seasonality
Evaluate stationarity using Dickey-Fuller test
Apply multiple forecasting models and compare their performance
Visualize forecasts alongside confidence intervals

**✨ Features**
Interactive UI with Streamlit
Data Filtering by store and item

Time Series Analysis:
Seasonal decomposition
Rolling statistics and stationarity test (ADF)

Multiple Forecasting Models:
ARIMA
SARIMA (for seasonal patterns)
Exponential Smoothing
ARCH/GARCH (for modeling volatility)
LSTM (neural network-based deep learning model)

Visualizations:
Plotly-based interactive charts
Forecasts with confidence intervals
Scalability: Handles large datasets efficiently

**🚀 Usage**
Upload Dataset
Upload a .csv file (train.csv) via the sidebar.
Required columns: date, store, item, sales.

Filter Data
Choose the desired store and item from dropdown menus.
Explore Visuals

View:
Sales trends
Seasonal decomposition
Rolling statistics
Dickey-Fuller test results
Run Forecasts
Click to generate and visualize forecasts using different models.

Interpret Results
Compare forecast vs. actual data
Use decomposition plots to understand underlying patterns
Analyze confidence intervals (especially for SARIMA)

**📦 Dataset**
Expected CSV structure:


date	store	item	sales
01-01-2013	1	1	13
02-01-2013	1	1	11
date: Sales date in DD-MM-YYYY format

store: Store ID
item: Item ID
sales: Units sold
Sample dataset available on Kaggle: Store Item Demand Forecasting Challenge

**🧠 Methodology**
🔧 Data Preprocessing
Convert date to datetime format
Extract features: day, month, year, day of week
Filter data by store and item

📊 Exploratory Data Analysis
Time series plots
Seasonal decomposition (additive model)
Rolling mean & std deviation
Dickey-Fuller test for stationarity

🧮 Forecasting Models
ARIMA: (5,1,0)
SARIMA: (1,1,1)(1,1,1,7)
Exponential Smoothing: Trend + Seasonality (365-day)
ARCH/GARCH: Model volatility in % returns
LSTM: Look-back window of 30 days

**📈 Evaluation**
Forecasts for a 90-day test window
Plot actual vs predicted
Show confidence intervals (e.g. SARIMA)

**🛠 Technologies Used**
Python
Streamlit – UI framework
Pandas, NumPy – Data manipulation
Statsmodels – ARIMA, SARIMA, Exponential Smoothing
ARCH – ARCH/GARCH volatility modeling
TensorFlow/Keras – LSTM
Plotly – Interactive visualizations
Scikit-learn – Preprocessing and scaling
