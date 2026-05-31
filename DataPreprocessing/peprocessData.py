import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class BMWDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def find_data_file(self):
        """Find the BMW data file in common locations"""
        possible_paths = [
            'BMW sales data (2010-2024).csv',
            './BMW sales data (2010-2024).csv',
            'S1/AI/KNN/BMW sales data (2010-2024).csv',
            '../BMW sales data (2010-2024).csv',
            'KNN/BMW sales data (2010-2024).csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found data file: {path}")
                return path
        return None
    
    def load_data(self, file_path=None):
        """Load the BMW dataset"""
        if file_path is None:
            file_path = self.find_data_file()
            
        if file_path is None:
            print("Could not find BMW data file. Please specify the path.")
            return False
            
        try:
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"File: {file_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_data(self):
        """Analyze the dataset structure"""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nCOLUMNS AND DATA TYPES:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            print(f"  {col:20} | {str(dtype):10} | {unique_count:4} unique values")
        
        print("\nMISSING VALUES:")
        missing_data = self.df.isnull().sum()
        has_missing = False
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(self.df)) * 100
                print(f"  {col:20} | {missing_count:4} ({percentage:.2f}%)")
                has_missing = True
        if not has_missing:
            print("  No missing values found!")
    
    def detect_column_types(self):
        """Detect numerical and categorical columns"""
        numerical_cols = []
        categorical_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return numerical_cols, categorical_cols
    
    def preprocess_data(self, target_column='Sales_Classification'):
        """Preprocess the BMW dataset for KNN"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle missing values
        initial_shape = df_processed.shape
        df_processed = df_processed.dropna()
        removed_count = initial_shape[0] - df_processed.shape[0]
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing values")
        
        # Detect column types
        numerical_cols, categorical_cols = self.detect_column_types()
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Encode categorical features
        print("\nENCODING CATEGORICAL FEATURES:")
        for col in categorical_cols:
            if col in df_processed.columns and col != target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col:20} | {len(le.classes_):2} classes")
        
        # Encode target if categorical
        if target_column in categorical_cols:
            le = LabelEncoder()
            df_processed[target_column] = le.fit_transform(df_processed[target_column].astype(str))
            self.label_encoders[target_column] = le
            print(f"  {target_column:20} | {len(le.classes_):2} classes (TARGET)")
        
        # Define feature columns (exclude target)
        self.feature_columns = [col for col in df_processed.columns if col != target_column]
        
        # Scale numerical features (excluding target)
        numerical_to_scale = [col for col in numerical_cols if col != target_column and col in df_processed.columns]
        
        if numerical_to_scale:
            df_processed[numerical_to_scale] = self.scaler.fit_transform(df_processed[numerical_to_scale])
            print(f"\nSCALED NUMERICAL FEATURES: {numerical_to_scale}")
        
        # Split features and target
        X = df_processed[self.feature_columns]
        y = df_processed[target_column]
        
        print(f"\nFINAL DATASET:")
        print(f"  Features: {X.shape[1]} columns")
        print(f"  Samples: {X.shape[0]} rows")
        print(f"  Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y, df_processed
    
    def get_target_options(self):
        """Get all possible target columns for classification"""
        numerical_cols, categorical_cols = self.detect_column_types()
        
        # Good targets for classification are categorical columns with reasonable number of classes
        good_targets = []
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if 2 <= unique_count <= 10:  # Binary or multi-class with reasonable classes
                good_targets.append((col, unique_count))
        
        return good_targets
    
    def save_processed_data(self, X, y, file_path='bmw_processed.csv'):
        """Save processed data for KNN"""
        processed_df = X.copy()
        processed_df['Target'] = y
        processed_df.to_csv(file_path, index=False)
        print(f"\nProcessed data saved to: {file_path}")
        print(f"Saved shape: {processed_df.shape}")

def main():
    # Initialize preprocessor
    preprocessor = BMWDataPreprocessor()
    
    # Load data
    if preprocessor.load_data():
        # Analyze data
        preprocessor.analyze_data()
        
        # Show target options
        target_options = preprocessor.get_target_options()
        print("\n" + "="*50)
        print("RECOMMENDED TARGETS FOR CLASSIFICATION")
        print("="*50)
        for target, count in target_options:
            print(f"  {target:20} | {count:2} classes")
        
        # Preprocess data with recommended target
        if target_options:
            recommended_target = target_options[0][0]  # Use first recommended target
            print(f"\nUsing target: {recommended_target}")
            
            X, y, df_processed = preprocessor.preprocess_data(target_column=recommended_target)
            
            # Save processed data
            preprocessor.save_processed_data(X, y)
            
            print("\n" + "="*50)
            print("PREPROCESSING COMPLETE!")
            print("="*50)
            print("The data is now ready for KNN classification.")
        else:
            print("No suitable target columns found for classification.")
    else:
        print("Please make sure the BMW data file is in one of these locations:")
        print("1. Current directory")
        print("2. S1/AI/KNN/ directory")
        print("3. KNN/ directory")

if __name__ == '__main__':
    main()