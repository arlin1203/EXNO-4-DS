# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
# EXNO-4-DS
# AIM: To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# STEP 1: Read the given Data
# Load the Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
print("Original Data (first 5 rows):\n", df.head())

# STEP 2: Data Cleaning
print("\nChecking for missing values:\n", df.isnull().sum())
df = df.dropna()
print("\nData cleaned successfully.")

# STEP 3: Feature Scaling

# Initialize different scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'RobustScaler': RobustScaler()
}

scaled_datasets = {}

# Apply each scaler
for name, scaler in scalers.items():
    scaled_data = scaler.fit_transform(df.drop('PRICE', axis=1))
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns[:-1])
    scaled_df['PRICE'] = df['PRICE']
    scaled_datasets[name] = scaled_df
    print(f"\n{name} applied successfully. First 3 rows:\n", scaled_df.head(3))

# STEP 4: Feature Selection

X = df.drop('PRICE', axis=1)
y = df['PRICE']

# (a) Filter Method - SelectKBest using f_regression
filter_selector = SelectKBest(score_func=f_regression, k=5)
filter_selector.fit(X, y)
filter_features = X.columns[filter_selector.get_support()]
print("\nFilter Method Selected Features:", list(filter_features))

# (b) Wrapper Method - Recursive Feature Elimination (RFE)
model = LinearRegression()
rfe_selector = RFE(model, n_features_to_select=5)
rfe_selector.fit(X, y)
wrapper_features = X.columns[rfe_selector.get_support()]
print("\nWrapper Method Selected Features:", list(wrapper_features))

# (c) Embedded Method - RandomForest Feature Importance
rf = RandomForestRegressor()
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
embedded_features = list(importance.head(5).index)
print("\nEmbedded Method Selected Features:", embedded_features)

# STEP 5: Save the final scaled and selected dataset
final_df = scaled_datasets['StandardScaler'][filter_features]
final_df['PRICE'] = df['PRICE']
final_df.to_csv("Scaled_FeatureSelected_Data.csv", index=False)

print("\nData saved to 'Scaled_FeatureSelected_Data.csv'")



<img width="1410" height="495" alt="image" src="https://github.com/user-attachments/assets/928c3336-a156-4f35-a3ab-f2f231775124" />
<img width="1410" height="495" alt="Screenshot 2025-10-16 105258" src="https://github.com/user-attachments/assets/ffdfa9f0-a034-4477-af27-7cdbe2991512" />

# RESULT:
     The above code executed sucessfully
