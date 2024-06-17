#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure plotting
plt.style.use('seaborn')
# Load the dataset
df = pd.read_csv('advertising.csv')

# Display the first few rows of the dataset
print(df.head())
# Basic statistics and information about the dataset
print(df.describe())
print(df.info())

# Visualize the relationship between advertising budgets and sales
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='reg')
plt.show()
# Check for missing values
print(df.isnull().sum())

# If there are any missing values, you can handle them (e.g., fill with mean or drop rows)
df = df.dropna()  # Example: dropping rows with missing values
# Select features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Plotting actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Sales')
plt.show()
# Example: Predicting sales for a new advertising budget
new_ad_budget = pd.DataFrame({'TV': [150], 'Radio': [30], 'Newspaper': [20]})
predicted_sales = model.predict(new_ad_budget)
print(f'Predicted Sales: {predicted_sales[0]:.2f}')


# In[ ]:




