import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('/Users/aymen/Downloads/Expresso_churn_dataset.csv')

# Generate a profiling report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("churn_profiling_report.html")


# Display initial data information
print(df.shape)
print(df.head())
print(df.describe())
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(f"\nNumber of duplicate rows after removal: {df.duplicated().sum()}")

# Identify numeric columns for outlier detection
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Remove rows with Z-score > 3 (outliers)
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    df = df[z_scores < 3.5]

# Check if the dataset is empty after outlier removal
if df.empty:
    print("Dataset is empty after outlier removal. Adjusting threshold or skipping outlier removal.")
    df = pd.read_csv('/Users/aymen/Downloads/Expresso_churn_dataset.csv')  # Reload original dataset

print(f"Shape after outlier removal: {df.shape}")

# Drop the 'user_id' column
df.drop('user_id', axis=1, inplace=True)

# Drop columns with more than 40% missing values
threshold = 0.4
missing_percent = df.isnull().mean()

# Identify columns to drop based on the missing percentage
columns_to_drop = missing_percent[missing_percent > threshold].index

# Drop columns that have too many missing values
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Display which columns have been dropped
print(f"Columns dropped due to excessive missing values: {columns_to_drop}")

# Fill missing values in numerical columns with the median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values in categorical columns with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if not df[col].mode().empty:  # Ensure there is a mode available
        df[col] = df[col].fillna(df[col].mode()[0])

# After filling, check for any remaining missing values
print("\nMissing Values After Filling:")
print(df.isnull().sum())
print(df.shape)

# Encode categorical columns using LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('CHURN', axis=1)  # Assuming 'CHURN' is the target variable
y = df['CHURN']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model for later use in Streamlit
joblib.dump(model, 'model/churn_model.pkl')
