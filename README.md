# Retail-Demand-Forecasting-DeepLearning
This project applies deep learning - Artificial Neural Network (ANN) - to build a model for **prediction of daily product demand in retail stores**.

---

## Project Overview
Accurate  product demand forcast is a critical challenge in retail supply chains due to its direct effect on lost sales if it is lower and excess cost due to overstocking if it is higher. In this project, I used  Artificial Neural Networks (ANNs), one of the main methods in Deep Learning, to predict daily product demandin retail stores.
**Business Insights:** Forecasting results provide actionable insights to minimize stock-outs, reduce excess inventory, and optimize replenishment planning.

---
## Project Objectives and assumptions

- Build a model to predict daily demand for each product 
- The model is built using Artificial Neural Networks (ANNs) - Deep Learning
- The model is trained by a dataset containing past sales, inventory levels, and product/store details given for each product in a given day
- **Modeling Assumption1:** Each day is treated as independent, allowing the task to be framed as standard supervised learning.
- **Modeling Assumption2:** Daily features (past sales, inventory, product/store attributes) are assumed to capture relevant patterns, removing the need for explicit time-series modeling.
- **Problem Formulation:** The model is formulated as a supervised regression problem, where daily product/store features are used to predict the expert demand forecast.
- **Frameworks:** The model is implemented using two major deep learning Python libraries: **PyTorch** and **TensorFlow**.
---
## Tools & Libraries used ???
- **Programming Language:** Python
- **Python Libraries:** Numpy, Pandas, Matplotlib, Scikit-learn, PyTorch, TensorFlow
- **Tool:** Jupyter Notebook, Git/GitHub

---
## Dataset & Variables
The dataset was provided in Kaggle (see Acknowledgments section) and includes detailed of retail store inventory as a table named **retail_store_inventory** with below varaibles.
- **number of Rows of the table:** ~73,000  
- **Columns (variabes):**
- Date –--- ---------------- The specific day when the sales or demand data was recorded.  
- Store ID –---------------- A unique identifier for each store location.
- Product ID –-------------- A unique identifier for each product in the catalog.
- Category –---------------- The classification of the product (e.g., electronics, toys, groceries).
- Inventory Level ---------- the number of products in inventory
- Region –------------------ The geographical area where the store is located (e.g., North, South, West ).
- Units Sold –-------------- The actual number of product units sold on the given date.
- Units Ordered –----------- The number of product units requested by the store or customers (may differ from units sold if out of stock).
- Demand Forecas------------ The predicted number of units expected to be sold, based on forecasting models (by experts).
- Price –------------------- The selling price of the product per unit at the time of sale.
- Discount –---------------- Any reduction applied to the original product price (percentage or amount).
- Weather Condition –------- Environmental factors such as sunny, rainy, which can influence demand.
- Holiday/Promotion –------- Indicates if the date coincided with a holiday, festival, or promotional campaign (binary ).
- Competitor Pricing –------ Price of a similar product offered by competitors in the same region or market.
- Seasonality –------------- Cyclical or seasonal effects influencing sales patterns (e.g., Autumn, winter, spring).

---
## Methodology ???
1. **Data Preprocessing**  
   - Handling missing values  
   - Feature engineering (time features, product/store encoding)  
   - Normalization  

2. **Model Development**  
   - **ANN in PyTorch**  
   - **ANN in TensorFlow**
   - - Framework: TensorFlow / PyTorch  
   - **Architecture:**  
       - Input Layer: Encoded categorical + scaled numerical features  
       - Dense(128, ReLU)  
       - Dense(64, ReLU)  
       - Dropout(0.3)  
       - Output: Dense(1, Linear) 
   - Hyperparameter tuning  

3. **ML Baseline Comparisons**  
   - Linear Regression  
   - Random Forest  
   - XGBoost  

4. **Evaluation Metrics**  
   - **MAE (Mean Absolute Error):** measures average forecast error  
   - **RMSE (Root Mean Squared Error):** penalizes larger errors  
   - **R² Score:** explains variance captured by the model

---
## Results ???
| Model                 | RMSE ↓  | MAE ↓  | R² ↑ |
|-----------------------|---------|--------|------|
| Linear Regression     | ...     | ...    | ...  |
| Random Forest         | ...     | ...    | ...  |
| XGBoost               | ...     | ...    | ...  |
| ANN (PyTorch)         | ...     | ...    | ...  |
| ANN (TensorFlow)      | ...     | ...    | ...  |

---
## PyTorch vs TensorFlow – My Observations ???
- **PyTorch:** More intuitive training loop, easier debugging.  
- **TensorFlow/Keras:** More concise, production-ready, easier deployment with TensorFlow Serving.  

Both frameworks achieved **comparable performance**, but each had strengths depending on workflow needs. 

---
## Project Structure ???
```
Retail-Demand-Forecasting-DeepLearning/
├── data/
│ ├── raw/
│ │   └── retail_store_inventory.csv        #Dataset comes from Kaggle (see Acknowledgments for detail)
│ ├── interim/
│ │   ├── ?????.py           
│ │   ├── ?????.py           
│ │   └── ?????.xlsx
│ └── processed/
│     ├── ?????.parquet
│     ├── ?????.parquet
│     ├── ?????.parquet
│     └── ?????.parquet
├── jupyter_notebook_Scripts/
│     ├── 01_EDA_Feature_Engineering.ipynb
│     ├── 02_Hypothesis_tesing.ipynb 
│     └── 03_ML_modeling.ipynb
├── output/
│ ├── jupyter_notebook/
│ │   ├── 01_EDA_Feature_Engineering.docx
│ │   ├── 02_Hypothesis_tesing.docx
│ │   └── 03_ML_modeling.docx
│ ├── SQL/
│ │   ├── merged_cleaned_tables_Hospitalisation_details_Medical_Examinations.csv
│ │   ├── data_analysis_1.png
│ │   ├── data_analysis_2.png
│ │   ├── data_analysis_3.png
│ │   └── data_analysis_4.png
├── plots/
│     ├── Box_plot_whisker.png
│     ├── Cost_dist_hospital_tier_gender.png
│     ├── Histogram.png
│     ├── Median_cost_radar_plot.png
│     ├── stacked_plot.png
│     ├── Swarm_Plots.png
│     ├── Heat_map.png
│     ├── Decision_Tree_Regressor.png
│     ├── K-Nearest_Neighbors_(KNN)_Regression.png
│     ├── Lasso_Regression_(L1_regularization).png
│     ├── Learning_Curves.png
│     ├── Linear_Regression.png
│     ├── Predicted_vs_Actual.png
│     ├── Random_Forest_Regressor.png
│     ├── Residual_Plot.png
│     ├── Ridge_Regression_(L2_regularization).png
│     ├── Support_Vector_Regressor_(SVR).png
│     └── XGBoost_Regression.png
├── SQL/
│     └── SQL_Script.sql
├── tableau/
│     ├── Business_Insights.twbx
│     └── Business_Insights.twb
├── ML_model.pkl
└── README.md

```


---
## Future Work
- Incorporate external features (holidays, promotions, weather)  
- Experiment with sequence models (RNN/LSTM/GRU) for time-series demand  
- Deploy model as an API for real-time demand forecasting  

---
## Acknowledgments
Dataset: [Kaggle-Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)  


---

## Author

**Abolfazl Zolfaghari**  
[Email](ab.zolfaghari.abbasghaleh) | [GitHub](https://github.com/abolfazl6678)

---
