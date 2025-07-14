# Life Expectancy Prediction Model

This project predicts total life expectancy using female and male life expectancies.

## Overview
- **Data**: A dataset (`life_expectancy.csv`) with life expectancy values for females, males, and both sexes.
- **Process**:
  - Cleaned data by removing unrealistic values.
  - Explored data to understand relationships between features.
  - Added noise to the target variable to mimic real-world imperfections.
  - Split data into training (80%) and testing (20%) sets and scaled features.
  - Trained a Linear Regression model and a tuned Ridge Regression model to predict life expectancy.
  - Visualized results with plots showing prediction accuracy and errors.
  - Saved the best model for future use.
- **Results**: The Ridge model effectively handles noise and correlated features, achieving accurate predictions.
- **Files**:
  - `main.py`: Code for data processing, modeling, and visualization.
  - `correlation_matrix.png`: Shows feature relationships.
  - `regression_results.png`: Displays prediction accuracy and errors.

## How It Works
The model uses female and male life expectancies to predict total life expectancy, with regularization to ensure robust predictions despite noisy data.

## Visualizations
- Correlation matrix: Reveals strong relationships between male, female, and total life expectancies.
- Regression results: Compares predicted vs. actual values and shows prediction errors.