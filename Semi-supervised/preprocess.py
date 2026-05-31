# preprocess.py
# Run this script once to get a clean, fully preprocessed version of the Bank Marketing dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
INPUT_FILE  = "/home/abdelkader/Downloads/bank+marketing/bank-additional/bank-additional/bank-additional-full.csv"   # <-- your original file
OUTPUT_FILE = "bank_additional_processed.csv"

# Target column name (the column that says whether the client subscribed)
TARGET_COL = "y"

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, sep=";")   # the original file uses semicolon separator
print(f"Original shape: {df.shape}")

# ------------------------------------------------------------------
# 2. Basic cleaning
# ------------------------------------------------------------------
# Remove duplicate rows (if any)
df = df.drop_duplicates()
print(f"After duplicate removal: {df.shape}")

# The column "duration" is data leakage for real prediction tasks (it is only known after the call)
# Most papers remove it when the goal is to predict before making the call
if "duration" in df.columns:
    df = df.drop(columns=["duration"])
    print("Dropped 'duration' column (data leakage)")

# ------------------------------------------------------------------
# 3. Handle the unknown / missing values
# ------------------------------------------------------------------
# In this dataset "unknown" is used as a missing-value marker for several categorical columns
df = df.replace("unknown", np.nan)

# ------------------------------------------------------------------
# 4. Separate features and target
# ------------------------------------------------------------------
y_raw = df[TARGET_COL]                     # "yes" / "no"
X = df.drop(columns=[TARGET_COL])

# ------------------------------------------------------------------
# 5. Encode the target variable (yes→1, no→0)
# ------------------------------------------------------------------
le_target = LabelEncoder()
y = pd.Series(le_target.fit_transform(y_raw), name=TARGET_COL)
print(f"Target classes: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# ------------------------------------------------------------------
# 6. Identify column types
# ------------------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns   ({len(numerical_cols)}): {numerical_cols}")

# ------------------------------------------------------------------
# 7. One-hot encode categorical variables
# ------------------------------------------------------------------
print("One-hot encoding categorical features ...")
X_cat_encoded = pd.get_dummies(X[categorical_cols], drop_first=True, dtype=int)  # drop_first avoids multicollinearity
X = pd.concat([X[numerical_cols], X_cat_encoded], axis=1)

print(f"Shape after one-hot encoding: {X.shape}")

# ------------------------------------------------------------------
# 8. Impute any remaining missing values (should be very few after "unknown" → NaN)
# ------------------------------------------------------------------
if X.isnull().any().any():
    print("Imputing missing numerical values with the column mean ...")
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ------------------------------------------------------------------
# 9. Feature scaling (StandardScaler – zero mean, unit variance)
# ------------------------------------------------------------------
print("Standardizing numerical features ...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ------------------------------------------------------------------
# 10. Put everything back together
# ------------------------------------------------------------------
df_processed = pd.concat([X_scaled, y], axis=1)

print(f"Final processed shape: {df_processed.shape}")
print(f"Final columns: {list(df_processed.columns[:10])} ... (total {df_processed.shape[1]})")

# ------------------------------------------------------------------
# 11. Save to CSV (no index, ready to be loaded by the GUI app)
# ------------------------------------------------------------------
df_processed.to_csv(OUTPUT_FILE, index=False)
print(f"\nPreprocessed dataset saved to '{OUTPUT_FILE}'")
print("You can now load this file directly in the PyQt5 Self-Training Demo app.")