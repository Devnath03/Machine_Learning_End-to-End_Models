# Pharmacy Sales Prediction

This project aims to predict pharmacy sales using machine learning techniques, specifically linear regression. The repository contains datasets, code, and instructions for building, training, and evaluating predictive models for pharmacy sales data.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Contributing](#contributing)


## Project Overview
Pharmacy sales prediction helps businesses forecast future sales, optimize inventory, and improve decision-making. This project demonstrates how to use linear regression to predict sales based on historical data.

## Datasets
- `Pharmacy_Sales_Data.csv`: Main dataset containing historical sales data.
- `linear_regression_pharmacy_sales.csv`: Preprocessed dataset for linear regression modeling.

## Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   cd Pharmacy_Sales_Prediction
   ```
2. (Optional) Create a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
- Open `Sales_Prediction.ipynb` in Jupyter Notebook or VS Code.
- Run the notebook cells to:
  - Load and preprocess data
  - Train linear regression model
  - Evaluate model performance
  - Visualize results

## Project Structure
```
Pharmacy_Sales_Prediction/
├── Pharmacy_Sales/
│   └── linear_regression_pharmacy_sales.csv
├── Pharmacy_Sales_Data.csv
├── linear_regression_pharmacy_sales.csv
├── Sales_Prediction.ipynb
├── requirements.txt
└── README.md
```

## Model Training & Evaluation
- The notebook demonstrates:
  - Data cleaning and preprocessing
  - Feature selection
  - Model training using scikit-learn's LinearRegression
  - Performance metrics (R², MAE, RMSE)
  - Visualization of predictions vs actual sales

## Results
- Model performance metrics and visualizations are available in the notebook output.
- Example plots include sales trends and prediction accuracy.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.


