This project applies deep learning to a critical supply chain problem: predicting daily product demand in retail stores. Using the Kaggle Retail Inventory Forecasting dataset, I developed an Artificial Neural Network (ANN) regression model to forecast the number of units sold for each product-store.

## Project Overview
Accurate  product demand (Units_Sold) is a critical challenge in retail supply chains due to its direct effect on lost sales if it is lower and excess cost due to overstocking if it is higher. In this project, I used  **Artificial Neural Networks (ANNs)**, one of the main methods in deep learning, to **predict daily product demand (Units_Sold)** based on store, product, and inventory data.
The dataset comes from [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset).

---

## Project Objectives

- Build a model to predict the number of units sold for each product daily
- The model is built using Artificial Neural Networks (ANNs) and trained by past sales, inventory levels, and product/store details given for each product in  a given day
- The model is formulated as a **supervised regression problem**
- The performance of the model is compared to trditional machine learning models (Linear Regression, XGboost, Random Forst)
- The model is built using  two main Python librairies **(PyTorch and TensorFlow)**

---
## Tools & Techniques used ???
- **Python Libraries:**
  - Numpy
  - Pandas
  - Matplotlib
  - PyTorch
  - Tensorflow
- **Techniques:**
  - Fully connected ANN

---

## Dataset & Variables ???
The dataset was provided in Kaggle (see Acknowledgments section) and includes detailed of retail store inventory as a table named **retail_store_inventory** with below varaibles.
- **number of Rows of the table:** ~73,000  
- **Columns (variabes):**
- Date –--------------------------- Unique identifier for each record  
- Case Number –------------------ Police department’s unique case ID  





   - `Date`
  - `Store_ID`
  - `Product_ID`
  - `Product_Category`
  - `Inventory_Available`
  - `Units_Sold` (Target)
  - `Reorder_Flag`

---

## Model ???
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

## Results (to be updated after training)
- ANN vs baseline comparison  
- Error metrics (MAE, RMSE, R²)  
- Forecast plots  

---








---

## Future Work
- Incorporate external features (holidays, promotions, weather)  
- Experiment with sequence models (RNN/LSTM/GRU) for time-series demand  
- Deploy model as an API for real-time demand forecasting  

---

## Acknowledgments
Dataset: [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)  






Key highlights:

Data Preprocessing & Feature Engineering: Created time-based features, lag variables, and encoded categorical attributes.

Model Development: Designed and trained an ANN using TensorFlow/PyTorch with multiple hidden layers and dropout for regularization.

Evaluation: Benchmarked performance against classical regression models (Linear Regression, Random Forest, XGBoost) using MAE, RMSE, and R².

Business Insights: Forecasting results provide actionable insights to minimize stock-outs, reduce excess inventory, and optimize replenishment planning.

This project demonstrates the ability to combine deep learning expertise with real-world supply chain applications, producing solutions that directly impact operational efficiency and business profitability.












## 📁 Repository Structure

Retail-Inventory-Demand-Forecasting/
│── data/                  # Kaggle dataset (not uploaded due to size)
│── pytorch_version/       # PyTorch implementation
│   ├── model.py           # ANN architecture
│   ├── train.py           # Training loop
│   └── evaluate.py        # Model evaluation
│── tensorflow_version/    # TensorFlow/Keras implementation
│   ├── model.py           # ANN architecture
│   ├── train.py           # Training & evaluation
│── traditional_ml/        # Baseline ML models (LR, RF, XGBoost)
│── notebooks/             # EDA, experiments, comparisons
│── results/               # Plots, metrics, reports
│── README.md              # Project documentation



🧠 Methodology

Data Preprocessing

Handling missing values

Feature engineering (time features, product/store encoding)

Normalization

Model Development

ANN in PyTorch

ANN in TensorFlow

Hyperparameter tuning

Baseline Comparisons

Linear Regression

Random Forest

XGBoost

Evaluation Metrics

RMSE, MAE, R² Score

🔎 Results
Model	RMSE ↓	MAE ↓	R² ↑
Linear Regression	...	...	...
Random Forest	...	...	...
XGBoost	...	...	...
ANN (PyTorch)	...	...	...
ANN (TensorFlow)	...	...	...














