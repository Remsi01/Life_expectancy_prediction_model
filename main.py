import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import joblib

# Load the dataset from a CSV file
df = pd.read_csv("life_expectancy.csv")

# Clean column names for easier access and consistency
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('  ', ' ', regex=False)
df.columns = df.columns.str.replace(' ', '_')

# Filter out unrealistic age data
# Define reasonable bounds for average life expectancy values.
# Assuming these columns represent average life expectancy in years.
min_life_expectancy = 30
max_life_expectancy = 95

initial_rows = df.shape[0]

# Apply filters to all relevant life expectancy columns
# Check if values are within the plausible range
df = df[
    (df['Sum_of_Females_Life_Expectancy'] >= min_life_expectancy) &
    (df['Sum_of_Females_Life_Expectancy'] <= max_life_expectancy) &
    (df['Sum_of_Males_Life_Expectancy'] >= min_life_expectancy) &
    (df['Sum_of_Males_Life_Expectancy'] <= max_life_expectancy) &
    (df['Sum_of_Life_Expectancy_(both_sexes)'] >= min_life_expectancy) &
    (df['Sum_of_Life_Expectancy_(both_sexes)'] <= max_life_expectancy)
]

filtered_rows = df.shape[0]
print(f"\nFiltered out {initial_rows - filtered_rows} rows with unrealistic life expectancy values.")
print(f"Remaining rows: {filtered_rows}")

print("\nDataset Overview (After Filtering)")
# Display descriptive statistics of the dataset to understand its distribution
print(df.describe())

# Create a figure for the correlation heatmap
plt.figure(figsize=(8, 6))
# Calculate the correlation matrix for numerical columns
corr_matrix = df.corr(numeric_only=True)
# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Features (After Filtering)")
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Define features (X) and target (y) variables
X = df[['Sum_of_Females_Life_Expectancy', 'Sum_of_Males_Life_Expectancy']]
y = df['Sum_of_Life_Expectancy_(both_sexes)']

# Add noise to the target variable to simulate real-world data imperfections
# Noise strength is set to 35% of the target's standard deviation
np.random.seed(42)
noise_strength = 0.35 * y.std()
y_noisy = y + np.random.normal(0, noise_strength, size=y.shape)

# Split the data into training and testing sets
# 80% for training, 20% for testing
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

# Initialize and fit StandardScaler on the training data
# Scaling is crucial for models sensitive to feature scales (like Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Evaluate Linear Regression model performance
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Ridge Regression with GridSearch for Hyperparameter Tuning
# Initialize Ridge Regression model
ridge = Ridge()
# Define the parameter grid for alpha (regularization strength) to search
ridge_params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

# Initialize GridSearchCV for hyperparameter tuning
# cv=5 means 5-fold cross-validation
# scoring='r2' means R-squared is the metric to optimize
# n_jobs=-1 means use all available CPU cores for parallel processing
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train_scaled, y_train)

# Get the best Ridge model found by GridSearchCV
ridge_best = ridge_grid.best_estimator_
y_pred_ridge = ridge_best.predict(X_test_scaled)

# Evaluate Best Ridge Regression model performance
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

# Print performance metrics for both models
print("\nLinear Regression:")
print(f"R2: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
print("\nBest Ridge Alpha:", ridge_grid.best_params_)
print(f"Ridge Regression R2: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")

plt.figure(figsize=(12, 5))

# Actual vs Predicted Plot (for Ridge Regression)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge, alpha=0.6)
# Plot a red dashed line representing perfect prediction (actual = predicted)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted (Ridge Regression)")

# Residuals Plot (for Ridge Regression)
residuals = y_test - y_pred_ridge
plt.subplot(1, 2, 2)
plt.scatter(y_pred_ridge, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Life Expectancy")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot (Ridge Regression)")

plt.tight_layout()
plt.savefig('regression_results.png')
plt.show()

# Initialize PCA to reduce to 1 principal component
pca = PCA(n_components=1)
# Fit PCA on scaled training data and transform both training and test data
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train a Ridge model using the same best alpha on PCA-transformed data
ridge_pca = Ridge(alpha=ridge_grid.best_params_['alpha'])
ridge_pca.fit(X_train_pca, y_train)
y_pred_pca = ridge_pca.predict(X_test_pca)

# Evaluate Ridge model with PCA
r2_pca = r2_score(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))

print("\nRidge with PCA:")
print(f"R2: {r2_pca:.4f}, RMSE: {rmse_pca:.4f}")

# Bundle the best model, scaler, and feature names into a dictionary
model_data = {
    'model': ridge_best,
    'scaler': scaler,
    'features': X.columns.tolist()
}
# Save the bundled model data to a file using joblib
joblib.dump(model_data, 'life_expectancy_model.pkl')
print("\nModel saved as 'life_expectancy_model.pkl'")