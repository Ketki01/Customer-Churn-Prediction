# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load and Prepare Data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop customerID (not a feature)
df = df.drop(columns=['customerID'])

# Convert TotalCharges to numeric (handle errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert Churn to binary (0/1)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 2. Encode Categorical Features
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future use

# 3. Split Data
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# 6. Save Artifacts
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('preprocessor.pkl', 'wb') as f:
    # Save both label encoders and SMOTE for preprocessing new data
    preprocessing_artifacts = {
        'label_encoders': label_encoders,
        'smote': smote
    }
    pickle.dump(preprocessing_artifacts, f)

print("Model and preprocessor saved successfully!")
