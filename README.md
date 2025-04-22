# 🛒 Store Item Demand Forecasting App

This project is a **Streamlit web application** built for forecasting item-level daily sales across multiple stores using historical time series data. It uses a SARIMA model for demand prediction and includes tools for visualizing trends, checking stationarity, and evaluating model performance.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [License](#license)

---

## 📖 Overview

Retailers face challenges in managing stock and optimizing supply chain operations. This app helps stakeholders forecast future demand for individual store-item pairs using past sales data. It enables smarter inventory decisions and reduces overstocking or stockouts.

---

## ✨ Features

✅ Upload and preprocess your own dataset  
✅ Filter by **Store** and **Item**  
✅ Visualize **sales trends** over time  
✅ Analyze **seasonality and trends** using decomposition  
✅ Perform **stationarity checks** (ADF Test, Rolling Stats)  
✅ Apply **SARIMA forecasting**  
✅ View **90-day forecasts** with confidence intervals  
✅ Intuitive **Streamlit dashboard interface**

---

## 🛠 Tech Stack

| Category        | Libraries/Tools                  |
|----------------|----------------------------------|
| Web App        | `Streamlit`                      |
| Data Handling  | `pandas`, `numpy`                |
| Visualization  | `matplotlib`, `seaborn`          |
| Time Series    | `statsmodels`, `SARIMAX`, `ADF`  |

---

## 🚀 How to Run

### 🧩 Step 1: Clone the Repo
```bash
git clone https://github.com/yourusername/store-item-demand-forecasting.git
cd store-item-demand-forecasting
