# Retail-Inventory-Demand-Forecasting-DeepLearning
This project applies deep learning to a critical supply chain problem: predicting daily product demand in retail stores. Using the Kaggle Retail Inventory Forecasting dataset, I developed an Artificial Neural Network (ANN) regression model to forecast the number of units sold for each product-store-day combination.




# 🏬 Retail Inventory Demand Forecasting with ANN

## 📌 Project Overview
Accurate inventory forecasting is a critical challenge in retail supply chains. Stock-outs lead to lost sales, while overstocking increases holding costs.  
In this project, we use **Artificial Neural Networks (ANNs)** to **predict daily product demand (`Units_Sold`)** based on store, product, and inventory data.  

The dataset comes from [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset).

---

## 🎯 Problem Statement
> **Given past sales, inventory levels, and product/store details, predict the number of units sold for each product in each store on a given day.**

This is formulated as a **supervised regression problem**.

---

## 📂 Dataset
- **Rows:** ~73,000  
- **Columns (examples):**
  - `Date`
  - `Store_ID`
  - `Product_ID`
  - `Product_Category`
  - `Inventory_Available`
  - `Units_Sold` (Target)
  - `Reorder_Flag`

---

## 🔧 Features
- **Time features:** Day of week, month, holidays  
- **Store attributes:** `Store_ID`  
- **Product attributes:** `Product_ID`, `Product_Category`  
- **Inventory info:** `Inventory_Available`  
- **Lag features:** Rolling averages & previous sales  

---

## 🎯 Target
- **`Units_Sold`** → continuous variable (demand forecast)

---

## 🧠 Model
- Framework: TensorFlow / PyTorch  
- Architecture (example):  
  - Input Layer: Encoded categorical + scaled numerical features  
  - Dense(128, ReLU)  
  - Dense(64, ReLU)  
  - Dropout(0.3)  
  - Output: Dense(1, Linear)  

---

## 📈 Evaluation Metrics
- **MAE (Mean Absolute Error):** measures average forecast error  
- **RMSE (Root Mean Squared Error):** penalizes larger errors  
- **R² Score:** explains variance captured by the model  

---

## 🚀 Project Pipeline
1. **Data Preprocessing**
   - Handle missing values & outliers  
   - Encode categorical variables (One-Hot / Embedding)  
   - Scale numerical features  
   - Create lag features for time-series context  

2. **Modeling**
   - Build baseline regression models for comparison  
   - Train ANN regression model  
   - Tune hyperparameters (batch size, learning rate, hidden units)  

3. **Evaluation**
   - Compare ANN vs baseline (Linear Regression, Random Forest, XGBoost)  
   - Plot Actual vs Predicted demand  

4. **Insights**
   - Identify demand patterns by product & store  
   - Show potential for **reducing stock-outs** and **minimizing holding costs**  

---

## 📊 Results (to be updated after training)
- ANN vs baseline comparison  
- Error metrics (MAE, RMSE, R²)  
- Forecast plots  

---








---

## 🔮 Future Work
- Incorporate external features (holidays, promotions, weather)  
- Experiment with sequence models (RNN/LSTM/GRU) for time-series demand  
- Deploy model as an API for real-time demand forecasting  

---

## 🙌 Acknowledgments
Dataset: [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)  
Author: *Your Name*  





Key highlights:

Data Preprocessing & Feature Engineering: Created time-based features, lag variables, and encoded categorical attributes.

Model Development: Designed and trained an ANN using TensorFlow/PyTorch with multiple hidden layers and dropout for regularization.

Evaluation: Benchmarked performance against classical regression models (Linear Regression, Random Forest, XGBoost) using MAE, RMSE, and R².

Business Insights: Forecasting results provide actionable insights to minimize stock-outs, reduce excess inventory, and optimize replenishment planning.

This project demonstrates the ability to combine deep learning expertise with real-world supply chain applications, producing solutions that directly impact operational efficiency and business profitability.












## 📁 Repository Structure
