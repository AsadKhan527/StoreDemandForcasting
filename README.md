Store Item Demand Forecasting App

A powerful and interactive web application for forecasting store item demand using advanced time series models like SARIMA, ARIMA, Exponential Smoothing, ARCH/GARCH, and LSTM. Built with Streamlit and Plotly, this app provides a user-friendly interface for analyzing historical sales data, visualizing trends, and generating accurate 90-day demand forecasts.

📑 Table of Contents

Project Overview
Features
Installation
Usage
Dataset
Methodology
Technologies Used
Directory Structure
Contributing
License
Acknowledgements


📊 Project Overview
The Store Item Demand Forecasting App is designed to predict future item-level sales for multiple stores, helping businesses optimize inventory and streamline supply chain operations. By leveraging a variety of time series forecasting models, the app provides detailed insights into sales trends, seasonality, and volatility, all visualized through interactive Plotly charts.
Problem Statement
The goal is to accurately forecast demand for each store-item combination over a 90-day period, enabling better inventory planning and reducing overstock or stockouts.
Key Objectives

Filter and preprocess sales data by store and item.
Perform time series decomposition to identify trends and seasonality.
Evaluate stationarity using the Dickey-Fuller test.
Apply multiple forecasting models and compare their performance.
Visualize forecasts with confidence intervals and historical data.


✨ Features

Interactive UI: Built with Streamlit for seamless user interaction.
Data Filtering: Select specific stores and items for analysis.
Time Series Analysis:
Seasonal decomposition to uncover trends, seasonality, and residuals.
Rolling statistics and Dickey-Fuller test for stationarity.


Multiple Forecasting Models:
ARIMA: Autoregressive Integrated Moving Average.
SARIMA: Seasonal ARIMA for handling seasonality.
Exponential Smoothing: For trend and seasonal components.
ARCH/GARCH: For modeling volatility in sales data.
LSTM: Deep learning-based forecasting with neural networks.


Visualizations:
Interactive sales trends, decomposition, and forecast plots using Plotly.
Confidence intervals for SARIMA forecasts.


Scalable: Handles large datasets with efficient preprocessing.


🛠 Installation
Follow these steps to set up the project locally:
Prerequisites

Python 3.8 or higher
pip (Python package manager)
Git (optional, for cloning the repository)

Steps

Clone the Repository:
git clone https://github.com/your-username/store-item-demand-forecasting.git
cd store-item-demand-forecasting


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run the Streamlit App:
streamlit run app.py


Access the App:Open your browser and navigate to http://localhost:8501.


Requirements
The requirements.txt file includes all necessary dependencies. Key packages include:
streamlit
pandas
numpy
statsmodels
tensorflow
plotly
scikit-learn
arch

To generate the requirements.txt file:
pip freeze > requirements.txt


🚀 Usage

Upload Dataset:

Use the sidebar to upload a CSV file (train.csv) containing sales data.
Expected columns: date, store, item, sales.


Filter Data:

Select a store and item from the dropdown menus in the sidebar.


Explore Visualizations:

View sales trends, seasonal decomposition, and rolling statistics.
Check the Dickey-Fuller test results for stationarity.


Run Forecasts:

Click buttons to generate forecasts using ARIMA, SARIMA, Exponential Smoothing, ARCH/GARCH, or LSTM.
Visualize forecasts against actual test data.


Interpret Results:

Analyze forecast accuracy and confidence intervals.
Use decomposition plots to understand seasonality and trends.




📦 Dataset
The app expects a CSV file with the following structure:



date
store
item
sales



01-01-2013
1
1
13


02-01-2013
1
1
11



date: Date of sales (format: DD-MM-YYYY).
store: Store identifier (e.g., 1, 2, ...).
item: Item identifier (e.g., 1, 2, ...).
sales: Number of units sold.

You can use a sample dataset like the one from the Store Item Demand Forecasting Challenge on Kaggle.

🧠 Methodology
The app follows a structured approach to time series forecasting:

Data Preprocessing:

Convert dates to datetime format.
Extract features like day, month, year, and day of week.
Filter data by store and item.


Exploratory Data Analysis:

Visualize sales trends over time.
Perform seasonal decomposition (additive model, 365-day period).
Calculate rolling mean and standard deviation.
Conduct Dickey-Fuller test to check stationarity.


Forecasting Models:

ARIMA: Fits a (5,1,0) model to differenced data.
SARIMA: Uses (1,1,1)(1,1,1,7) for weekly seasonality.
Exponential Smoothing: Applies additive trend and 365-day seasonality.
ARCH/GARCH: Models volatility in percentage returns.
LSTM: Uses a 30-day look-back period with a single-layer LSTM network.


Evaluation:

Forecasts are generated for a 90-day test period.
Visual comparisons between actual and predicted values.
Confidence intervals for SARIMA forecasts.




🛠 Technologies Used

Python: Core programming language.
Streamlit: For building the interactive web app.
Pandas & NumPy: Data manipulation and numerical computations.
Statsmodels: For ARIMA, SARIMA, and Exponential Smoothing models.
ARCH: For ARCH/GARCH volatility modeling.
TensorFlow/Keras: For LSTM neural network implementation.
Plotly: For interactive visualizations.
Scikit-learn: For data scaling and preprocessing.


📁 Directory Structure
store-item-demand-forecasting/
│
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # Directory for datasets (not included in repo)
│   └── train.csv          # Sample dataset (user-provided)
├── screenshots/           # Directory for app screenshots
│   └── dashboard.png


🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure your code follows PEP 8 guidelines and includes relevant tests.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements

Streamlit for the amazing web app framework.
Plotly for interactive visualizations.
Statsmodels and TensorFlow for robust modeling tools.
Kaggle for inspiring datasets and challenges.


For any issues or suggestions, please open an issue on the GitHub repository. Happy forecasting! 🚀
