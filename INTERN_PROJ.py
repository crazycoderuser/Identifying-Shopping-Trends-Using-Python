import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Load the dataset
df = pd.read_csv('shopping_data.csv')

# Display first few rows
print(df.head())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical variables into numerical variables
categorical_cols = ['Gender', 'Category', 'Location', 'Size', 'Color', 'Season', 'Review Rating', 'Subscription Status', 'Payment Method', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Preferred Payment Method']
le = LabelEncoder()

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)




# Print the columns of the encoded DataFrame
print("Columns in df_encoded:", df_encoded.columns.tolist())


columns_to_drop = ['Customer ID', 'Review Rating']
existing_columns_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]

features = df_encoded.drop(columns=existing_columns_to_drop)  # Drop non-numeric columns



for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
# Plot age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution of Customers')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot purchase amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Purchase Amount (USD)'], bins=20, kde=True)
plt.title('Purchase Amount Distribution')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.show()

# Plot category distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Category', data=df)
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Feature selection for clustering
features = df[['Age', 'Purchase Amount (USD)', 'Frequency of Purchases']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Purchase Amount (USD)', y='Frequency of Purchases', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency of Purchases')
plt.show()
# Define features and target variable
X = df[['Age', 'Purchase Amount (USD)', 'Frequency of Purchases']]
y = df['Previous Purchases']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
