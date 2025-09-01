This project applies deep learning to a critical supply chain problem: predicting daily product demand in retail stores. Using the Kaggle Retail Inventory Forecasting dataset, I developed an Artificial Neural Network (ANN) regression model to forecast the number of units sold for each product-store.

## Project Overview
Accurate  product demand (Units_Sold) is a critical challenge in retail supply chains due to its direct effect on lost sales if it is lower and excess cost due to overstocking if it is higher. In this project, I used  **Artificial Neural Networks (ANNs)**, one of the main methods in deep learning, to **predict daily product demand (Units_Sold)** based on store, product, and inventory data.
The dataset comes from [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset). 
**Business Insights:** Forecasting results provide actionable insights to minimize stock-outs, reduce excess inventory, and optimize replenishment planning.

---

## Project Objectives

- Build a model to predict the number of units sold for each product daily
- The model is built using Artificial Neural Networks (ANNs) and trained by past sales, inventory levels, and product/store details given for each product in  a given day
- The model is formulated as a **supervised regression problem**
- The performance of the model is compared to trditional machine learning models (Linear Regression, XGboost, Random Forst)
- The model is built using  two main deep learning Python librairies **(PyTorch and TensorFlow)**

---
## Tools & Techniques used ???
- **Programming Language:** Python
- **Python Libraries:** Numpy, Pandas, Matplotlib, Scikit-learn, XGBoost, PyTorch, TensorFlow
- **Tool:** Jupyter Notebook and Git/GitHub
- **Techniques:**
  - Fully connected ANN

---

## Dataset & Variables
The dataset was provided in Kaggle (see Acknowledgments section) and includes detailed of retail store inventory as a table named **retail_store_inventory** with below varaibles.
- **number of Rows of the table:** ~73,000  
- **Columns (variabes):**
- Date –--- ---------------- The specific day when the sales or demand data was recorded.  
- Store ID –---------------- A unique identifier for each store location.
- Product ID –-------------- A unique identifier for each product in the catalog.
- Category –---------------- The classification of the product (e.g., electronics, toys, groceries).
- Region –------------------ The geographical area where the store is located (e.g., North, South, West ).
- Units Sold –-------------- The actual number of product units sold on the given date.
- Units Ordered –----------- The number of product units requested by the store or customers (may differ from units sold if out of stock).
- Demand Forecas------------ The predicted number of units expected to be sold, based on forecasting models.
- Price –------------------- The selling price of the product per unit at the time of sale.
- Discount –---------------- Any reduction applied to the original product price (percentage or amount).
- Weather Condition –------- Environmental factors such as sunny, rainy, which can influence demand.
- Holiday/Promotion –------- Indicates if the date coincided with a holiday, festival, or promotional campaign (binary ).
- Competitor Pricing –------ Price of a similar product offered by competitors in the same region or market.
- Seasonality –------------- Cyclical or seasonal effects influencing sales patterns (e.g., Autumn, winter, spring).

---

## Modeling ???
- Framework: TensorFlow / PyTorch  
- Architecture:  
  - Input Layer: Encoded categorical + scaled numerical features  
  - Dense(128, ReLU)  
  - Dense(64, ReLU)  
  - Dropout(0.3)  
  - Output: Dense(1, Linear)  

---
## Modeling and hyperparamter tunning ???
- Framework: TensorFlow / PyTorch  
- Architecture (example):

---

## Evaluation Metrics
- **MAE (Mean Absolute Error):** measures average forecast error  
- **RMSE (Root Mean Squared Error):** penalizes larger errors  
- **R² Score:** explains variance captured by the model  

---
## Results ??? 
- ANN vs baseline ML comparison ()  
- Error metrics (MAE, RMSE, R²)  
- Forecast plots  

---

## Project Structure ???
```
Retail-Inventory-Demand-Forecasting-DeepLearning/
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
```

---
## Future Work
- Incorporate external features (holidays, promotions, weather)  
- Experiment with sequence models (RNN/LSTM/GRU) for time-series demand  
- Deploy model as an API for real-time demand forecasting  

---

## Acknowledgments
Dataset: [Kaggle – Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)  







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














