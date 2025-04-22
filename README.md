# 🛒 Store Item Demand Forecasting App

A Streamlit-based interactive dashboard for forecasting daily sales of items across multiple stores using classical time series models (**SARIMA**) and deep learning models (**LSTM**). Designed for analysts, retailers, and supply chain managers.

---


---

## 📦 Dataset Overview

The application expects a dataset with the following structure:

| Column   | Description                     |
|----------|---------------------------------|
| `date`   | Date of sale (daily granularity)|
| `store`  | Store ID                        |
| `item`   | Item ID                         |
| `sales`  | Number of items sold            |

- **Time Range:** Multiple years
- **Granularity:** Daily sales per store-item combination

---

## 🔍 Features

- 📊 **Interactive Data Exploration**
  - Filter by Store and Item
  - View sales trends using Plotly
- 🧠 **Time Series Analysis**
  - Seasonal decomposition (trend, seasonality, residual)
  - Dickey-Fuller test for stationarity
- 📈 **Forecasting Models**
  - **SARIMA** with confidence intervals
  - **LSTM** deep learning model with future predictions
- 📎 **Visual Output**
  - Zoomable forecast charts
  - Historical vs predicted overlays

---

## 🧪 Methodology

1. **Data Preprocessing**  
   - Date parsing, feature engineering  
   - Filtering by store/item

2. **Exploratory Analysis**  
   - Rolling mean & std  
   - Seasonal decomposition using statsmodels  

3. **Stationarity Check**  
   - Augmented Dickey-Fuller (ADF) test  
   - Differencing if needed

4. **Modeling & Forecasting**  
   - SARIMA: Statistical forecasting with seasonal cycles  
   - LSTM: Neural network forecasting with historical sequence learning  

5. **Visualization**  
   - Forecasted vs actual values  
   - Confidence bands and diagnostics

---

## 📂 Project Structure

store-demand-forecasting/
├── 📄 app.py              # Streamlit application script
├── 📂 data/
│   └── 📄 train.csv       # Input dataset (can be uploaded through the app)
├── 📄 requirements.txt    # Project dependencies
├── 📄 README.md           # Project overview and instructions
├── 📄 LICENSE             # License file (e.g., MIT)
└── 📂 models/             # (Optional) Saved models or future model extensions


---

🛠️ Tech Stack
Frontend: Streamlit, Plotly

Backend: Python, Statsmodels, TensorFlow (LSTM)

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

📈 Example Use Cases
Retail stock replenishment

Demand planning and logistics

Sales trend analysis

Model comparison and experimentation

---

🙌 Acknowledgements
This project is inspired by real-world retail forecasting challenges and built with ❤️ using open-source tools.

---

🧑‍💻 Author
Asad Khan
GitHub: https://github.com/AsadKhan527
LinkedIn: linkedin.com/in/asad-khan-0a526225b

---

📄 License
This project is licensed under the MIT License. See LICENSE for details.
