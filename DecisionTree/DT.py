import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

class DecisionTreeWorker(QThread):
    update_progress = pyqtSignal(int, str, dict)
    finished = pyqtSignal(dict)
    
    def __init__(self, X_train, X_test, y_train, y_test, algorithm, criterion, max_depth, min_samples_split, 
                 min_samples_leaf, task_type):
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
        self.tree = None
        
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
                self.tree = DecisionTreeClassifier(
                    criterion=sklearn_criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42
                )
            else:
                self.tree = DecisionTreeRegressor(
                    criterion=sklearn_criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42
                )
            
            self.update_progress.emit(50, "Training Decision Tree...", {})
            self.tree.fit(self.X_train, self.y_train)
            
            y_pred = self.tree.predict(self.X_test)
            metrics = self.calculate_metrics(self.y_test, y_pred)
            tree_structure = self.extract_tree_structure()
            
            self.update_progress.emit(100, "Training Complete!", metrics)
            self.finished.emit({
                'tree': self.tree,
                'metrics': metrics,
                'tree_structure': tree_structure,
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

    def extract_tree_structure(self):
        tree = self.tree.tree_
        structure = {
            'node_count': tree.node_count,
            'children_left': tree.children_left,
            'children_right': tree.children_right,
            'feature': tree.feature,
            'threshold': tree.threshold,
            'value': tree.value,
            'impurity': tree.impurity,
            'n_node_samples': tree.n_node_samples
        }
        return structure

class TreeVisualizer:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.node_positions = {}
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
    def calculate_layout(self, tree_structure):
        children_left = tree_structure['children_left']
        children_right = tree_structure['children_right']
        
        node_depths = np.zeros(shape=tree_structure['node_count'], dtype=np.int64)
        stack = [(0, -1)]
        while stack:
            node_id, parent_depth = stack.pop()
            node_depths[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
        
        max_depth = np.max(node_depths)
        if max_depth == 0: max_depth = 1
        
        y_positions = np.linspace(0.9, 0.1, max_depth + 1)
        
        for depth in range(max_depth + 1):
            nodes_at_depth = np.where(node_depths == depth)[0]
            if len(nodes_at_depth) > 0:
                x_positions = np.linspace(0.1, 0.9, len(nodes_at_depth) + 2)[1:-1]
                for i, node_id in enumerate(nodes_at_depth):
                    self.node_positions[node_id] = (x_positions[i], y_positions[depth])

    def draw_tree(self, tree_structure, feature_names, task_type):
        self.ax.clear()
        self.ax.set_title('Decision Tree Visualization', fontsize=12, fontweight='bold')
        
        children_left = tree_structure['children_left']
        children_right = tree_structure['children_right']
        features = tree_structure['feature']
        thresholds = tree_structure['threshold']
        values = tree_structure['value']
        
        legend_elements = []
        used_features = set()
        
        for node_id in range(tree_structure['node_count']):
            if node_id not in self.node_positions:
                continue
                
            x, y = self.node_positions[node_id]
            
            if children_left[node_id] != children_right[node_id]:
                left_child = children_left[node_id]
                right_child = children_right[node_id]
                
                if left_child in self.node_positions:
                    x_left, y_left = self.node_positions[left_child]
                    self.ax.plot([x, x_left], [y, y_left], 'k-', alpha=0.6, linewidth=1)
                if right_child in self.node_positions:
                    x_right, y_right = self.node_positions[right_child]
                    self.ax.plot([x, x_right], [y, y_right], 'k-', alpha=0.6, linewidth=1)
                
                feature_idx = features[node_id]
                if feature_names and feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                    color = self.colors[feature_idx % len(self.colors)]
                    
                    if feature_name not in used_features:
                        legend_elements.append(patches.Patch(color=color, label=feature_name))
                        used_features.add(feature_name)
                else:
                    feature_name = f"F{feature_idx}"
                    color = 'gray'
                
                threshold = thresholds[node_id]
                
                circle = patches.Circle((x, y), 0.03, facecolor=color, edgecolor='black', linewidth=1)
                self.ax.add_patch(circle)
                
                self.ax.text(x, y, f"{threshold:.2f}", ha='center', va='center', fontsize=6, fontweight='bold')
                self.ax.text(x, y+0.03, feature_name, ha='center', va='bottom', fontsize=6, fontweight='bold')
                
            else:
                if task_type == 'classification':
                    class_idx = np.argmax(values[node_id])
                    samples = np.sum(values[node_id])
                    node_text = f"{class_idx}"
                else:
                    pred_value = values[node_id].mean()
                    samples = tree_structure['n_node_samples'][node_id]
                    node_text = f"{pred_value:.2f}"
                
                circle = patches.Circle((x, y), 0.02, facecolor='lightgreen', edgecolor='darkgreen', linewidth=1)
                self.ax.add_patch(circle)
                self.ax.text(x, y, node_text, ha='center', va='center', fontsize=6, fontweight='bold')
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                          ncol=3, fontsize=8)
        
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

class DecisionTreeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.tree_worker = None
        self.visualizer = None
        self.current_tree = None
        self.feature_names = []
        self.original_df = None
        self.target_encoder = None
        self.init_ui()
        self.load_initial_data()
        
    def init_ui(self):
        self.setWindowTitle('Decision Tree Simulator')
        
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        window_width = min(1200, screen_size.width() - 100)
        window_height = min(800, screen_size.height() - 100)
        self.setGeometry(50, 50, window_width, window_height)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        control_panel = QWidget()
        control_panel.setMaximumWidth(350)
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
        
        algo_group = QGroupBox("Tree Configuration")
        algo_layout = QVBoxLayout(algo_group)
        
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(['classification', 'regression'])
        task_layout.addWidget(self.task_combo)
        algo_layout.addLayout(task_layout)
        
        algo_type_layout = QHBoxLayout()
        algo_type_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['CART', 'ID3', 'C4.5', 'CHAID', 'MARS'])
        algo_type_layout.addWidget(self.algo_combo)
        algo_layout.addLayout(algo_type_layout)
        
        criterion_layout = QHBoxLayout()
        criterion_layout.addWidget(QLabel("Criterion:"))
        self.criteria_combo = QComboBox()
        self.update_criteria_options()
        criterion_layout.addWidget(self.criteria_combo)
        algo_layout.addLayout(criterion_layout)
        
        control_layout.addWidget(algo_group)
        
        params_group = QGroupBox("Tree Parameters")
        params_layout = QVBoxLayout(params_group)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setValue(4)
        depth_layout.addWidget(self.depth_spin)
        params_layout.addLayout(depth_layout)
        
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Min Split:"))
        self.split_spin = QSpinBox()
        self.split_spin.setRange(2, 20)
        self.split_spin.setValue(5)
        split_layout.addWidget(self.split_spin)
        params_layout.addLayout(split_layout)
        
        leaf_layout = QHBoxLayout()
        leaf_layout.addWidget(QLabel("Min Leaf:"))
        self.leaf_spin = QSpinBox()
        self.leaf_spin.setRange(1, 10)
        self.leaf_spin.setValue(2)
        leaf_layout.addWidget(self.leaf_spin)
        params_layout.addLayout(leaf_layout)
        
        control_layout.addWidget(params_group)
        
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
        
        self.progress_label = QLabel("Ready to train")
        control_layout.addWidget(self.progress_label)
        
        btn_group = QGroupBox("Controls")
        btn_layout = QHBoxLayout(btn_group)
        
        self.train_btn = QPushButton("Train Tree")
        self.train_btn.clicked.connect(self.train_tree)
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
        self.visualizer = TreeVisualizer(self.fig, self.ax)
        
        self.ax.text(0.5, 0.5, 'Load dataset and train tree', 
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
        self.algo_combo.setCurrentText('CART')
        self.criteria_combo.setCurrentText('Gini Impurity')
        self.depth_spin.setValue(4)
        self.split_spin.setValue(5)
        self.leaf_spin.setValue(2)
        
        if self.df is not None and 'Status' in self.df.columns:
            self.target_combo.setCurrentText('Status')
        
        self.results_text.setText("Parameters reset")
        self.test_btn.setEnabled(False)
        
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Parameters reset\nClick Train Tree to start', 
                    ha='center', va='center', transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()
    
    def train_tree(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        try:
            target_col = self.target_combo.currentText()
            X = self.df[self.feature_names].values
            y = self.df[target_col].values
            
            train_size = self.train_size_spin.value() / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
            
            algorithm = self.algo_combo.currentText()
            criterion = self.criteria_combo.currentText()
            max_depth = self.depth_spin.value()
            min_samples_split = self.split_spin.value()
            min_samples_leaf = self.leaf_spin.value()
            task_type = self.task_combo.currentText()
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.train_btn.setEnabled(False)
            
            self.tree_worker = DecisionTreeWorker(
                X_train, X_test, y_train, y_test,
                algorithm, criterion, max_depth, min_samples_split,
                min_samples_leaf, task_type
            )
            self.tree_worker.update_progress.connect(self.on_training_update)
            self.tree_worker.finished.connect(self.on_training_finished)
            self.tree_worker.start()
            
            self.results_text.setText("Training started...")
            
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
        
        self.current_tree = results['tree']
        metrics = results['metrics']
        
        self.visualizer.calculate_layout(results['tree_structure'])
        self.visualizer.draw_tree(results['tree_structure'], self.feature_names, self.task_combo.currentText())
        self.canvas.draw()
        
        results_text = "Training Complete\n\n"
        if self.task_combo.currentText() == 'classification':
            results_text += f"Accuracy: {metrics['accuracy_percent']}\n"
        else:
            results_text += f"MSE: {metrics['mse']:.4f}\nRMSE: {metrics['rmse']:.4f}\n"
        
        self.results_text.setText(results_text)
    
    def show_test_dialog(self):
        if self.current_tree is None:
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
            prediction = self.current_tree.predict(input_array)[0]
            
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
            pass

def main():
    app = QApplication(sys.argv)
    window = DecisionTreeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()