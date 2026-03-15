# %% [markdown]
# # Exploratory Data Analysis
# This notebook explores the historical sales data.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.config import RAW_DATA_FILE

# %%
# Load Data
df = pd.read_csv(RAW_DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df.head()

# %%
# Sales over time
plt.figure(figsize=(15, 6))
sns.lineplot(data=df, x='date', y='units_sold', hue='store_id')
plt.title('Sales over Time by Store')
plt.show()

# %%
# Sales Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['units_sold'], bins=30)
plt.title('Distribution of Daily Sales')
plt.show()

# %%
# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
