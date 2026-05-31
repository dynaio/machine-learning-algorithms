import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QPushButton, 
                             QGroupBox, QTextEdit, QProgressBar,
                             QSpinBox, QDoubleSpinBox, QFileDialog,
                             QMessageBox, QLineEdit, QFormLayout, QDialog)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import warnings
import os
warnings.filterwarnings('ignore')

class RandomForestWorker(QThread):
    update_progress = pyqtSignal(int, str, dict)
    finished = pyqtSignal(dict)
    
    def __init__(self, X_train, X_test, y_train, y_test, algorithm, criterion, max_depth, min_samples_split, 
                 min_samples_leaf, task_type, n_estimators, max_features, bootstrap):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.algorithm = algorithm
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.forest = None
        
    def run(self):
        try:
            criterion_map = {
                'Gini Impurity': 'gini',
                'Information Gain (Entropy)': 'entropy',
                'Variance Reduction': 'squared_error',
                'Mean Squared Error': 'squared_error'
            }
            
            sklearn_criterion = criterion_map.get(self.criterion, 'gini')
            
            if self.task_type == 'classification':
                self.forest = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    criterion=sklearn_criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.forest = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    criterion=sklearn_criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    random_state=42,
                    n_jobs=-1
                )
            
            total_trees = self.n_estimators
            
            for i in range(total_trees):
                progress = 20 + (i / total_trees) * 70
                self.update_progress.emit(int(progress), f"Training tree {i+1}/{total_trees}...", {})
                
            self.forest.fit(self.X_train, self.y_train)
            
            y_pred = self.forest.predict(self.X_test)
            metrics = self.calculate_metrics(self.y_test, y_pred)
            feature_importance = self.forest.feature_importances_
            
            self.update_progress.emit(100, "Training Complete!", metrics)
            self.finished.emit({
                'forest': self.forest,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'X_train': self.X_train, 'X_test': self.X_test,
                'y_train': self.y_train, 'y_test': self.y_test,
                'y_pred': y_pred
            })
            
        except Exception as e:
            self.update_progress.emit(0, f"Error: {str(e)}", {})

    def calculate_metrics(self, y_true, y_pred):
        metrics = {}
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = accuracy
            metrics['accuracy_percent'] = f"{accuracy:.2%}"
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            target_names = [f'Class {cls}' for cls in unique_classes]
            metrics['report'] = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        else:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            metrics['mse'] = mse
            metrics['rmse'] = rmse
        return metrics

class ForestVisualizer:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
    def draw_feature_importance(self, feature_importance, feature_names):
        self.ax.clear()
        self.ax.set_title('Random Forest - Feature Importance', fontsize=12, fontweight='bold')
        
        indices = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        bars = self.ax.barh(range(len(sorted_features)), sorted_importance, color=self.colors)
        
        self.ax.set_yticks(range(len(sorted_features)))
        self.ax.set_yticklabels(sorted_features, fontsize=8)
        self.ax.set_xlabel('Importance Score')
        self.ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
            width = bar.get_width()
            self.ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', ha='left', va='center', fontsize=7)

class RandomForestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.forest_worker = None
        self.visualizer = None
        self.current_forest = None
        self.feature_names = []
        self.original_df = None
        self.target_encoder = None
        self.init_ui()
        self.load_initial_data()
        
    def init_ui(self):
        self.setWindowTitle('Random Forest Simulator')
        
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        window_width = min(1200, screen_size.width() - 100)
        window_height = min(800, screen_size.height() - 100)
        self.setGeometry(50, 50, window_width, window_height)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        control_panel = QWidget()
        control_panel.setMaximumWidth(400)
        control_layout = QVBoxLayout(control_panel)
        
        data_group = QGroupBox("Dataset Configuration")
        data_layout = QVBoxLayout(data_group)
        
        self.load_btn = QPushButton("Select CSV Dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        data_layout.addWidget(self.load_btn)
        
        self.data_info = QLabel("Loading default dataset...")
        self.data_info.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 8px; }")
        data_layout.addWidget(self.data_info)
        
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Training Size:"))
        self.train_size_spin = QSpinBox()
        self.train_size_spin.setRange(50, 95)
        self.train_size_spin.setValue(70)
        self.train_size_spin.setSuffix("%")
        train_layout.addWidget(self.train_size_spin)
        data_layout.addLayout(train_layout)
        
        control_layout.addWidget(data_group)
        
        forest_group = QGroupBox("Forest Configuration")
        forest_layout = QVBoxLayout(forest_group)
        
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(['classification', 'regression'])
        task_layout.addWidget(self.task_combo)
        forest_layout.addLayout(task_layout)
        
        estimators_layout = QHBoxLayout()
        estimators_layout.addWidget(QLabel("Number of Trees:"))
        self.estimators_spin = QSpinBox()
        self.estimators_spin.setRange(1, 200)
        self.estimators_spin.setValue(50)
        estimators_layout.addWidget(self.estimators_spin)
        forest_layout.addLayout(estimators_layout)
        
        features_layout = QHBoxLayout()
        features_layout.addWidget(QLabel("Max Features:"))
        self.features_combo = QComboBox()
        self.features_combo.addItems(['sqrt', 'log2', 'all'])
        features_layout.addWidget(self.features_combo)
        forest_layout.addLayout(features_layout)
        
        bootstrap_layout = QHBoxLayout()
        bootstrap_layout.addWidget(QLabel("Bootstrap:"))
        self.bootstrap_combo = QComboBox()
        self.bootstrap_combo.addItems(['True', 'False'])
        bootstrap_layout.addWidget(self.bootstrap_combo)
        forest_layout.addLayout(bootstrap_layout)
        
        control_layout.addWidget(forest_group)
        
        tree_group = QGroupBox("Tree Parameters")
        tree_layout = QVBoxLayout(tree_group)
        
        criterion_layout = QHBoxLayout()
        criterion_layout.addWidget(QLabel("Criterion:"))
        self.criteria_combo = QComboBox()
        self.update_criteria_options()
        criterion_layout.addWidget(self.criteria_combo)
        tree_layout.addLayout(criterion_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(10)
        depth_layout.addWidget(self.depth_spin)
        tree_layout.addLayout(depth_layout)
        
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Min Split:"))
        self.split_spin = QSpinBox()
        self.split_spin.setRange(2, 20)
        self.split_spin.setValue(2)
        split_layout.addWidget(self.split_spin)
        tree_layout.addLayout(split_layout)
        
        leaf_layout = QHBoxLayout()
        leaf_layout.addWidget(QLabel("Min Leaf:"))
        self.leaf_spin = QSpinBox()
        self.leaf_spin.setRange(1, 10)
        self.leaf_spin.setValue(1)
        leaf_layout.addWidget(self.leaf_spin)
        tree_layout.addLayout(leaf_layout)
        
        control_layout.addWidget(tree_group)
        
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout(feature_group)
        
        feature_layout.addWidget(QLabel("Target Variable:"))
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self.on_target_changed)
        feature_layout.addWidget(self.target_combo)
        
        control_layout.addWidget(feature_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to train Random Forest")
        control_layout.addWidget(self.progress_label)
        
        btn_group = QGroupBox("Controls")
        btn_layout = QHBoxLayout(btn_group)
        
        self.train_btn = QPushButton("Train Forest")
        self.train_btn.clicked.connect(self.train_forest)
        btn_layout.addWidget(self.train_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_parameters)
        btn_layout.addWidget(self.reset_btn)
        
        control_layout.addWidget(btn_group)
        
        test_group = QGroupBox("Testing")
        test_layout = QVBoxLayout(test_group)
        
        self.test_btn = QPushButton("Manual Test")
        self.test_btn.clicked.connect(self.show_test_dialog)
        self.test_btn.setEnabled(False)
        test_layout.addWidget(self.test_btn)
        
        self.test_result = QLabel("Train a model first")
        test_layout.addWidget(self.test_result)
        
        control_layout.addWidget(test_group)
        
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        control_layout.addWidget(results_group)
        
        main_layout.addWidget(control_panel)
        
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.visualizer = ForestVisualizer(self.fig, self.ax)
        
        self.ax.text(0.5, 0.5, 'Load dataset and train Random Forest\n to see feature importance', 
                    ha='center', va='center', transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()
        
        viz_layout.addWidget(self.canvas)
        main_layout.addWidget(viz_panel)
        
    def load_initial_data(self):
        default_path = 'S1/AI/DecisionTree/Breast_Cancer.csv'
        if os.path.exists(default_path):
            self.load_dataset_file(default_path)
        else:
            current_path = 'Breast_Cancer.csv'
            if os.path.exists(current_path):
                self.load_dataset_file(current_path)
    
    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV Dataset", "", "CSV Files (*.csv)")
        if file_path:
            self.load_dataset_file(file_path)
    
    def load_dataset_file(self, file_path):
        try:
            self.original_df = pd.read_csv(file_path)
            self.df = self.original_df.copy()
            
            for col in self.df.select_dtypes(include=['object']).columns:
                if self.df[col].nunique() < 20:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
            
            self.data_info.setText(f"Dataset: {os.path.basename(file_path)}\nSamples: {self.df.shape[0]}")
            
            self.target_combo.clear()
            self.target_combo.addItems(self.df.columns)
            
            if 'Status' in self.df.columns:
                self.target_combo.setCurrentText('Status')
            
            self.train_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def on_target_changed(self):
        if self.df is not None and self.target_combo.currentText():
            current_target = self.target_combo.currentText()
            self.feature_names = [col for col in self.df.columns if col != current_target]
            
            if self.original_df is not None and current_target in self.original_df.columns:
                if self.original_df[current_target].dtype == 'object':
                    self.target_encoder = LabelEncoder()
                    self.target_encoder.fit(self.original_df[current_target].astype(str))
    
    def update_criteria_options(self):
        self.criteria_combo.clear()
        task_type = self.task_combo.currentText()
        if task_type == 'classification':
            self.criteria_combo.addItems(['Gini Impurity', 'Information Gain (Entropy)'])
        else:
            self.criteria_combo.addItems(['Variance Reduction', 'Mean Squared Error'])
    
    def reset_parameters(self):
        self.train_size_spin.setValue(70)
        self.task_combo.setCurrentText('classification')
        self.estimators_spin.setValue(50)
        self.features_combo.setCurrentText('sqrt')
        self.bootstrap_combo.setCurrentText('True')
        self.criteria_combo.setCurrentText('Gini Impurity')
        self.depth_spin.setValue(10)
        self.split_spin.setValue(2)
        self.leaf_spin.setValue(1)
        
        if self.df is not None and 'Status' in self.df.columns:
            self.target_combo.setCurrentText('Status')
        
        self.results_text.setText("Parameters reset")
        self.test_btn.setEnabled(False)
        
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Parameters reset\nClick Train Forest to start', 
                    ha='center', va='center', transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()
    
    def train_forest(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        try:
            target_col = self.target_combo.currentText()
            X = self.df[self.feature_names].values
            y = self.df[target_col].values
            
            train_size = self.train_size_spin.value() / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
            
            algorithm = "Random Forest"
            criterion = self.criteria_combo.currentText()
            max_depth = self.depth_spin.value()
            min_samples_split = self.split_spin.value()
            min_samples_leaf = self.leaf_spin.value()
            task_type = self.task_combo.currentText()
            n_estimators = self.estimators_spin.value()
            max_features = self.features_combo.currentText()
            bootstrap = self.bootstrap_combo.currentText() == 'True'
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.train_btn.setEnabled(False)
            
            self.forest_worker = RandomForestWorker(
                X_train, X_test, y_train, y_test,
                algorithm, criterion, max_depth, min_samples_split,
                min_samples_leaf, task_type, n_estimators, max_features, bootstrap
            )
            self.forest_worker.update_progress.connect(self.on_training_update)
            self.forest_worker.finished.connect(self.on_training_finished)
            self.forest_worker.start()
            
            self.results_text.setText("Training Random Forest...")
            
        except Exception as e:
            self.results_text.setText(f"Error: {str(e)}")
            self.train_btn.setEnabled(True)
    
    def on_training_update(self, progress, message, metrics):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"{message} - {progress}%")
    
    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Training Complete")
        
        self.current_forest = results['forest']
        metrics = results['metrics']
        feature_importance = results['feature_importance']
        
        self.visualizer.draw_feature_importance(feature_importance, self.feature_names)
        self.canvas.draw()
        
        results_text = "Random Forest Training Complete\n\n"
        results_text += f"Number of Trees: {self.estimators_spin.value()}\n"
        results_text += f"Max Features: {self.features_combo.currentText()}\n"
        results_text += f"Bootstrap: {self.bootstrap_combo.currentText()}\n\n"
        
        if self.task_combo.currentText() == 'classification':
            results_text += f"Accuracy: {metrics['accuracy_percent']}\n"
        else:
            results_text += f"MSE: {metrics['mse']:.4f}\nRMSE: {metrics['rmse']:.4f}\n"
        
        results_text += f"\nTop Features:\n"
        indices = np.argsort(feature_importance)[::-1][:5]
        for i, idx in enumerate(indices):
            results_text += f"  {i+1}. {self.feature_names[idx]}: {feature_importance[idx]:.3f}\n"
        
        self.results_text.setText(results_text)
    
    def show_test_dialog(self):
        if self.current_forest is None:
            QMessageBox.warning(self, "Warning", "Please train a model first!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Test")
        dialog.setFixedSize(400, 500)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Enter feature values:"))
        
        form_layout = QFormLayout()
        self.test_inputs = {}
        
        for feature in self.feature_names:
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Enter {feature}")
            
            if self.original_df is not None and feature in self.original_df.columns:
                sample_value = self.original_df[feature].iloc[0]
                if isinstance(sample_value, (int, float)):
                    input_field.setText("0")
                else:
                    input_field.setText(str(sample_value))
            else:
                input_field.setText("0")
                
            form_layout.addRow(f"{feature}:", input_field)
            self.test_inputs[feature] = input_field
        
        layout.addLayout(form_layout)
        
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(lambda: self.update_prediction())
        layout.addWidget(predict_btn)
        
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        layout.addWidget(self.test_output)
        
        dialog.exec_()
    
    def update_prediction(self):
        try:
            input_values = []
            original_values = {}
            
            for feature in self.feature_names:
                value_str = self.test_inputs[feature].text().strip()
                if not value_str:
                    self.test_output.setText("Please fill all fields")
                    return
                    
                original_values[feature] = value_str
                
                if self.original_df is not None and feature in self.original_df.columns:
                    sample_value = self.original_df[feature].iloc[0]
                    if isinstance(sample_value, (int, float)):
                        value = float(value_str)
                    else:
                        le = LabelEncoder()
                        le.fit(self.original_df[feature].astype(str))
                        value = le.transform([value_str])[0]
                else:
                    value = float(value_str)
                    
                input_values.append(value)
            
            input_array = np.array([input_values])
            prediction = self.current_forest.predict(input_array)[0]
            
            task_type = self.task_combo.currentText()
            target_name = self.target_combo.currentText()
            
            if task_type == 'classification':
                if self.target_encoder is not None:
                    prediction_label = self.target_encoder.inverse_transform([int(prediction)])[0]
                    result_text = f"Predicted {target_name}: {prediction_label}"
                else:
                    result_text = f"Predicted {target_name}: Class {int(prediction)}"
            else:
                result_text = f"Predicted {target_name}: {prediction:.2f}"
            
            input_details = "Input Values:\n"
            for feature, value in original_values.items():
                input_details += f"  {feature}: {value}\n"
            
            self.test_output.setText(f"{result_text}\n\n{input_details}")
            
        except Exception as e:
            self.test_output.setText(f"Error: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = RandomForestApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()