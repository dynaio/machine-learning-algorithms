import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QPushButton, 
                             QTextEdit, QGroupBox, QProgressBar, QCheckBox,
                             QFileDialog, QMessageBox, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal

class DataPreprocessor:
    """Advanced Data Preprocessing Class"""
    
    def __init__(self):
        self.original_df = None
        self.processed_df = None
        self.label_encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        try:
            self.original_df = pd.read_csv(file_path)
            return True, f"Data loaded successfully! Shape: {self.original_df.shape}"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def analyze_data(self):
        """Comprehensive data analysis"""
        if self.original_df is None:
            return "No data loaded!"
        
        analysis_report = "=== DATA ANALYSIS REPORT ===\n\n"
        
        # Basic info
        analysis_report += f"Dataset Shape: {self.original_df.shape}\n"
        analysis_report += f"Total Samples: {self.original_df.shape[0]}\n"
        analysis_report += f"Total Features: {self.original_df.shape[1]}\n\n"
        
        # Data types
        analysis_report += "=== DATA TYPES ===\n"
        for col in self.original_df.columns:
            dtype = self.original_df[col].dtype
            unique_count = self.original_df[col].nunique()
            analysis_report += f"{col:25} | {str(dtype):10} | Unique: {unique_count:4}\n"
        
        # Missing values
        analysis_report += "\n=== MISSING VALUES ===\n"
        missing_data = self.original_df.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(self.original_df)) * 100
                analysis_report += f"{col:25} | {missing_count:4} ({percentage:.2f}%)\n"
        
        if missing_data.sum() == 0:
            analysis_report += "No missing values found! ✅\n"
        
        # Data quality issues
        analysis_report += "\n=== DATA QUALITY ISSUES ===\n"
        issues_found = False
        
        for col in self.original_df.select_dtypes(include=['object']).columns:
            # Check for leading/trailing spaces
            has_spaces = self.original_df[col].astype(str).str.contains(r'^\s+|\s+$').any()
            if has_spaces:
                analysis_report += f"⚠️  '{col}' has leading/trailing spaces\n"
                issues_found = True
            
            # Check for inconsistent casing
            unique_vals = self.original_df[col].astype(str).unique()
            if len(unique_vals) > 0:
                lower_vals = [str(val).lower() for val in unique_vals]
                if len(set(lower_vals)) != len(unique_vals):
                    analysis_report += f"⚠️  '{col}' has inconsistent casing\n"
                    issues_found = True
        
        if not issues_found:
            analysis_report += "No major data quality issues found! ✅\n"
        
        return analysis_report
    
    def clean_data(self):
        """Clean the dataset"""
        if self.original_df is None:
            return False, "No data loaded!"
        
        try:
            df = self.original_df.copy()
            
            # Remove leading/trailing spaces from all string columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            # Fix specific issues (like "Single " with space)
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].replace({'Single ': 'Single'})
            
            self.original_df = df
            return True, "Data cleaned successfully! ✅"
            
        except Exception as e:
            return False, f"Error cleaning data: {str(e)}"
    
    def handle_missing_values(self, strategy='mean'):
        """Handle missing values"""
        if self.original_df is None:
            return False, "No data loaded!"
        
        try:
            df = self.original_df.copy()
            
            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                if strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif strategy == 'mode':
                    imputer = SimpleImputer(strategy='most_frequent')
                
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                self.imputers['numerical'] = imputer
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
                self.imputers['categorical'] = cat_imputer
            
            self.original_df = df
            return True, f"Missing values handled using {strategy} strategy! ✅"
            
        except Exception as e:
            return False, f"Error handling missing values: {str(e)}"
    
    def encode_categorical_features(self, encoding_type='label'):
        """Encode categorical features"""
        if self.original_df is None:
            return False, "No data loaded!"
        
        try:
            df = self.original_df.copy()
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if encoding_type == 'label':
                # Label Encoding
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
            
            elif encoding_type == 'onehot':
                # One-Hot Encoding
                df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
            
            self.original_df = df
            return True, f"Categorical features encoded using {encoding_type} encoding! ✅"
            
        except Exception as e:
            return False, f"Error encoding categorical features: {str(e)}"
    
    def scale_features(self, scaling_type='standard'):
        """Scale numerical features"""
        if self.original_df is None:
            return False, "No data loaded!"
        
        try:
            df = self.original_df.copy()
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            if scaling_type == 'standard':
                scaler = StandardScaler()
            elif scaling_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers[scaling_type] = scaler
            
            self.processed_df = df
            return True, f"Features scaled using {scaling_type} scaling! ✅"
            
        except Exception as e:
            return False, f"Error scaling features: {str(e)}"
    
    def save_processed_data(self, file_path):
        """Save processed data to CSV"""
        try:
            if self.processed_df is not None:
                self.processed_df.to_csv(file_path, index=False)
                return True, f"Processed data saved to: {file_path} ✅"
            else:
                return False, "No processed data available!"
        except Exception as e:
            return False, f"Error saving data: {str(e)}"

class PreprocessingThread(QThread):
    update_progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    
    def __init__(self, preprocessor, steps):
        super().__init__()
        self.preprocessor = preprocessor
        self.steps = steps
        
    def run(self):
        try:
            total_steps = len(self.steps)
            current_step = 0
            
            for step in self.steps:
                current_step += 1
                progress = int((current_step / total_steps) * 100)
                
                if step['type'] == 'clean':
                    self.update_progress.emit("Cleaning data...", progress)
                    success, message = self.preprocessor.clean_data()
                    
                elif step['type'] == 'missing':
                    self.update_progress.emit("Handling missing values...", progress)
                    success, message = self.preprocessor.handle_missing_values(step['strategy'])
                    
                elif step['type'] == 'encode':
                    self.update_progress.emit("Encoding categorical features...", progress)
                    success, message = self.preprocessor.encode_categorical_features(step['encoding'])
                    
                elif step['type'] == 'scale':
                    self.update_progress.emit("Scaling features...", progress)
                    success, message = self.preprocessor.scale_features(step['scaling'])
                
                if not success:
                    self.finished.emit(f"Error: {message}")
                    return
                
                # Small delay for smooth progress update
                self.msleep(300)
            
            self.update_progress.emit("Processing complete!", 100)
            self.finished.emit("Data preprocessing completed successfully! ✅")
            
        except Exception as e:
            self.finished.emit(f"Processing error: {str(e)}")

class DataPreprocessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.preprocessor = DataPreprocessor()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Advanced Data Preprocessing Tool - ML Ready')
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Data Loading Tab
        load_tab = QWidget()
        load_layout = QVBoxLayout(load_tab)
        
        # File selection
        file_group = QGroupBox("Data File Selection")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load CSV File")
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_label)
        
        load_layout.addWidget(file_group)
        
        # Analysis display
        analysis_group = QGroupBox("Data Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analyze_btn = QPushButton("Analyze Data")
        self.analyze_btn.clicked.connect(self.analyze_data)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        load_layout.addWidget(analysis_group)
        tabs.addTab(load_tab, "Data Loading & Analysis")
        
        # Preprocessing Tab
        prep_tab = QWidget()
        prep_layout = QVBoxLayout(prep_tab)
        
        # Preprocessing options
        options_group = QGroupBox("Preprocessing Options")
        options_layout = QVBoxLayout(options_group)
        
        # Data cleaning
        self.clean_cb = QCheckBox("Clean Data (remove spaces, fix inconsistencies)")
        self.clean_cb.setChecked(True)
        options_layout.addWidget(self.clean_cb)
        
        # Missing values
        missing_layout = QHBoxLayout()
        missing_layout.addWidget(QLabel("Handle Missing Values:"))
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(['mean', 'median', 'mode', 'skip'])
        missing_layout.addWidget(self.missing_combo)
        missing_layout.addStretch()
        options_layout.addLayout(missing_layout)
        
        # Encoding
        encode_layout = QHBoxLayout()
        encode_layout.addWidget(QLabel("Categorical Encoding:"))
        self.encode_combo = QComboBox()
        self.encode_combo.addItems(['label', 'onehot', 'skip'])
        encode_layout.addWidget(self.encode_combo)
        encode_layout.addStretch()
        options_layout.addLayout(encode_layout)
        
        # Scaling
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Feature Scaling:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['standard', 'minmax', 'robust', 'skip'])
        scale_layout.addWidget(self.scale_combo)
        scale_layout.addStretch()
        options_layout.addLayout(scale_layout)
        
        prep_layout.addWidget(options_group)
        
        # Process buttons
        process_group = QGroupBox("Processing Controls")
        process_layout = QVBoxLayout(process_group)
        
        self.process_btn = QPushButton("Start Preprocessing")
        self.process_btn.clicked.connect(self.start_preprocessing)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to process")
        process_layout.addWidget(self.progress_label)
        
        prep_layout.addWidget(process_group)
        
        # Results
        results_group = QGroupBox("Processing Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        self.save_btn = QPushButton("Save Processed Data")
        self.save_btn.clicked.connect(self.save_data)
        self.save_btn.setEnabled(False)
        results_layout.addWidget(self.save_btn)
        
        prep_layout.addWidget(results_group)
        tabs.addTab(prep_tab, "Preprocessing")
        
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
            success, message = self.preprocessor.load_data(file_path)
            
            if success:
                self.analyze_btn.setEnabled(True)
                self.process_btn.setEnabled(True)
                self.analysis_text.setText(message)
            else:
                QMessageBox.critical(self, "Error", message)
    
    def analyze_data(self):
        analysis_report = self.preprocessor.analyze_data()
        self.analysis_text.setText(analysis_report)
    
    def start_preprocessing(self):
        # Build processing steps
        steps = []
        
        if self.clean_cb.isChecked():
            steps.append({'type': 'clean'})
        
        missing_strategy = self.missing_combo.currentText()
        if missing_strategy != 'skip':
            steps.append({'type': 'missing', 'strategy': missing_strategy})
        
        encoding_type = self.encode_combo.currentText()
        if encoding_type != 'skip':
            steps.append({'type': 'encode', 'encoding': encoding_type})
        
        scaling_type = self.scale_combo.currentText()
        if scaling_type != 'skip':
            steps.append({'type': 'scale', 'scaling': scaling_type})
        
        if not steps:
            QMessageBox.warning(self, "Warning", "No preprocessing steps selected!")
            return
        
        # Start processing thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        
        self.thread = PreprocessingThread(self.preprocessor, steps)
        self.thread.update_progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()
    
    def update_progress(self, message, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_processing_finished(self, message):
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        if message.startswith("Error"):
            QMessageBox.critical(self, "Error", message)
        else:
            self.results_text.setText(message)
            QMessageBox.information(self, "Success", message)
    
    def save_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Data", "processedData.csv", "CSV Files (*.csv)")
        if file_path:
            success, message = self.preprocessor.save_processed_data(file_path)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)

def main():
    app = QApplication(sys.argv)
    window = DataPreprocessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()