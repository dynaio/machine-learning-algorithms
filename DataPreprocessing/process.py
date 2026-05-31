# fix_bank_once.py — RUN THIS ONLY ONCE
import pandas as pd

# CHANGE THIS TO YOUR EXACT FILE PATH
file_path = "/home/abdelkader/VSCdium/Workspace/S1/AI/DataPreprocessing/UniversalBank.xls"   # or .xlsx

print("Reading file...")
try:
    # Force engine for old .xls files
    df = pd.read_excel(file_path, engine='xlrd')   # xlrd handles .xls
except:
    try:
        df = pd.read_excel(file_path, engine='openpyxl')  # openpyxl for .xlsx
    except Exception as e:
        print("ERROR:", e)
        exit()

print("Original columns:", df.columns.tolist())

# Fix column names (remove spaces)
df.columns = [col.strip() for col in df.columns]
print("Clean columns:", df.columns.tolist())

# Check target
if 'Personal Loan' not in df.columns:
    print("ERROR: 'Personal Loan' still missing!")
    print("Available:", df.columns.tolist())
    exit()

# Drop junk
df = df.drop(columns=['ID', 'ZIP Code'], errors='ignore')

# Save perfect CSV
df.to_csv("bank_ready.csv", index=False)
print("\nSUCCESS! Saved as: bank_ready.csv")
print(f"Shape: {df.shape}")
print("Target distribution:")
print(df['Personal Loan'].value_counts())
print("\nNow run: python svm_bank_final.py")