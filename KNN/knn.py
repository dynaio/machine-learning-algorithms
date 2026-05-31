import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QPushButton, 
                             QGroupBox, QTextEdit, QProgressBar,
                             QSpinBox, QDoubleSpinBox, QFileDialog,
                             QMessageBox, QLineEdit, QFormLayout, QDialog,
                             QTabWidget, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import warnings
import os
import random
from collections import Counter
warnings.filterwarnings('ignore')

class KNNWorker(QThread):
    update_progress = pyqtSignal(int, str, dict)
    finished = pyqtSignal(dict)
    
    def __init__(self, X_train, X_test, y_train, y_test, n_neighbors, weights, 
                 algorithm, metric, leaf_size, task_type, optimize_k=False):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.task_type = task_type
        self.optimize_k = optimize_k
        self.knn = None
        self.optimal_k = n_neighbors
        
    def run(self):
        try:
            if self.optimize_k:
                self.update_progress.emit(10, "Finding optimal K value...", {})
                self.optimal_k = self.find_optimal_k()
                self.n_neighbors = self.optimal_k
            
            self.update_progress.emit(30, f"Initializing KNN with K={self.n_neighbors}...", {})
            
            self.knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                metric=self.metric,
                leaf_size=self.leaf_size
            )
            
            self.update_progress.emit(50, "Training KNN model...", {})
            self.knn.fit(self.X_train, self.y_train)
            
            self.update_progress.emit(70, "Making predictions...", {})
            y_pred = self.knn.predict(self.X_test)
            y_pred_proba = self.knn.predict_proba(self.X_test) if hasattr(self.knn, 'predict_proba') else None
            
            self.update_progress.emit(90, "Calculating metrics...", {})
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['optimal_k'] = self.optimal_k
            
            self.update_progress.emit(100, "Training Complete!", metrics)
            self.finished.emit({
                'knn': self.knn,
                'metrics': metrics,
                'X_train': self.X_train, 'X_test': self.X_test,
                'y_train': self.y_train, 'y_test': self.y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'optimal_k': self.optimal_k
            })
            
        except Exception as e:
            self.update_progress.emit(0, f"Error: {str(e)}", {})
    
    def find_optimal_k(self):
        """Find the optimal K value using cross-validation"""
        k_range = range(1, min(50, len(self.X_train) // 2))
        k_scores = []
        
        for k in k_range:
            if k >= len(self.X_train):
                break
                
            knn = KNeighborsClassifier(n_neighbors=k)
            try:
                scores = cross_val_score(knn, self.X_train, self.y_train, cv=min(5, len(self.X_train)), scoring='accuracy')
                k_scores.append(scores.mean())
            except:
                k_scores.append(0)
        
        if k_scores:
            optimal_k = k_range[np.argmax(k_scores)]
            return optimal_k
        return self.n_neighbors

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        metrics = {}
        accuracy = accuracy_score(y_true, y_pred)
        metrics['accuracy'] = accuracy
        metrics['accuracy_percent'] = f"{accuracy:.2%}"
        
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [f'Class {cls}' for cls in unique_classes]
        metrics['report'] = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Calculate confidence scores if probabilities are available
        if y_pred_proba is not None:
            max_proba = np.max(y_pred_proba, axis=1)
            metrics['avg_confidence'] = np.mean(max_proba)
            metrics['min_confidence'] = np.min(max_proba)
            metrics['max_confidence'] = np.max(max_proba)
        else:
            metrics['avg_confidence'] = 0.0
            metrics['min_confidence'] = 0.0
            metrics['max_confidence'] = 0.0
        
        return metrics

class KOptimizationWorker(QThread):
    update_progress = pyqtSignal(int, str, list)
    finished = pyqtSignal(dict)
    
    def __init__(self, X_train, y_train, max_k=50):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.max_k = min(max_k, len(X_train) - 1)
        
    def run(self):
        try:
            k_range = range(1, self.max_k + 1)
            k_scores = []
            
            for i, k in enumerate(k_range):
                progress = int((i / len(k_range)) * 100)
                self.update_progress.emit(progress, f"Testing K={k}...", [])
                
                if k >= len(self.X_train):
                    break
                    
                knn = KNeighborsClassifier(n_neighbors=k)
                try:
                    scores = cross_val_score(knn, self.X_train, self.y_train, 
                                           cv=min(5, len(self.X_train)), 
                                           scoring='accuracy')
                    k_scores.append(scores.mean())
                except Exception as e:
                    k_scores.append(0)
            
            optimal_k = k_range[np.argmax(k_scores)]
            optimal_score = max(k_scores)
            
            self.finished.emit({
                'k_range': list(k_range)[:len(k_scores)],
                'k_scores': k_scores,
                'optimal_k': optimal_k,
                'optimal_score': optimal_score
            })
            
        except Exception as e:
            self.update_progress.emit(0, f"Error: {str(e)}", [])

class KNNVisualizer:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.color_palette = plt.cm.Set3(np.linspace(0, 1, 12))
        self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
    def get_random_color(self, class_label):
        """Get a random but consistent color for each class"""
        random.seed(class_label * 42)  # Consistent randomness
        return random.choice(self.color_palette)
    
    def get_marker_style(self, class_label):
        """Get different marker for each class"""
        return self.marker_styles[class_label % len(self.marker_styles)]
    
    def draw_feature_space(self, X, y, feature_names, feature_indices=[0, 1], title="Feature Space"):
        self.ax.clear()
        
        if X.shape[1] < 2:
            self.ax.text(0.5, 0.5, 'Need at least 2 features for visualization', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.axis('off')
            return
        
        # Use only the selected features for visualization
        X_vis = X[:, feature_indices]
        
        # Add some jitter for better visualization
        jitter_strength = (X_vis.max() - X_vis.min()) * 0.01
        X_vis_jittered = X_vis + np.random.normal(0, jitter_strength, X_vis.shape)
        
        # Plot each class with different color and marker
        for class_label in np.unique(y):
            class_mask = y == class_label
            if len(X_vis_jittered[class_mask]) > 0:
                color = self.get_random_color(class_label)
                marker = self.get_marker_style(class_label)
                
                self.ax.scatter(X_vis_jittered[class_mask, 0], X_vis_jittered[class_mask, 1], 
                              c=[color], marker=marker, label=f'Class {class_label}',
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.8)
        
        self.ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
        self.ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        
    def draw_decision_boundary(self, knn, X, y, feature_names, feature_indices=[0, 1]):
        self.ax.clear()
        
        if X.shape[1] < 2:
            self.ax.text(0.5, 0.5, 'Need at least 2 features for visualization', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.axis('off')
            return
            
        X_vis = X[:, feature_indices]
        
        # Create mesh grid
        x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
        y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                           np.linspace(y_min, y_max, 80))
        
        # Create full feature array for prediction
        mesh_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            if i == feature_indices[0]:
                mesh_points[:, i] = xx.ravel()
            elif i == feature_indices[1]:
                mesh_points[:, i] = yy.ravel()
            else:
                mesh_points[:, i] = np.mean(X[:, i])
        
        try:
            Z = knn.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Create colorful decision regions
            n_classes = len(np.unique(y))
            colors = [self.get_random_color(i) for i in range(n_classes)]
            
            # Plot decision boundary with transparency
            self.ax.contourf(xx, yy, Z, alpha=0.4, colors=colors, levels=n_classes-1)
            
            # Plot data points with jitter
            jitter_strength = (X_vis.max() - X_vis.min()) * 0.01
            X_vis_jittered = X_vis + np.random.normal(0, jitter_strength, X_vis.shape)
            
            for class_label in np.unique(y):
                class_mask = y == class_label
                if len(X_vis_jittered[class_mask]) > 0:
                    color = self.get_random_color(class_label)
                    marker = self.get_marker_style(class_label)
                    
                    self.ax.scatter(X_vis_jittered[class_mask, 0], X_vis_jittered[class_mask, 1], 
                                  c=[color], marker=marker, label=f'Class {class_label}',
                                  alpha=0.8, s=40, edgecolors='black', linewidth=0.8)
            
            self.ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
            self.ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
            self.ax.set_title(f'KNN Decision Boundary (K={knn.n_neighbors})', fontsize=14, fontweight='bold')
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.grid(True, alpha=0.3)
            self.fig.tight_layout()
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f'Error drawing boundary: {str(e)}', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.axis('off')
        
    def draw_neighbors_example(self, knn, X, y, feature_names, feature_indices=[0, 1], sample_idx=0):
        self.ax.clear()
        
        if X.shape[1] < 2:
            self.ax.text(0.5, 0.5, 'Need at least 2 features for visualization', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.axis('off')
            return
            
        X_vis = X[:, feature_indices]
        
        # Add jitter for better visualization
        jitter_strength = (X_vis.max() - X_vis.min()) * 0.01
        X_vis_jittered = X_vis + np.random.normal(0, jitter_strength, X_vis.shape)
        
        # Plot all points with colors and markers
        for class_label in np.unique(y):
            class_mask = y == class_label
            if len(X_vis_jittered[class_mask]) > 0:
                color = self.get_random_color(class_label)
                marker = self.get_marker_style(class_label)
                
                self.ax.scatter(X_vis_jittered[class_mask, 0], X_vis_jittered[class_mask, 1], 
                              c=[color], marker=marker, label=f'Class {class_label}',
                              alpha=0.5, s=30)
        
        # Ensure sample index is valid
        sample_idx = min(sample_idx, len(X) - 1)
        
        # Highlight test sample
        test_point = X[sample_idx:sample_idx+1]
        test_point_vis = test_point[:, feature_indices]
        
        # Add jitter to test point for consistency
        test_point_jittered = test_point_vis + np.random.normal(0, jitter_strength, test_point_vis.shape)
        
        self.ax.scatter(test_point_jittered[:, 0], test_point_jittered[:, 1], 
                       c='red', s=300, marker='*', edgecolors='black', 
                       linewidth=3, label='Test point', zorder=5)
        
        try:
            # Find and plot k nearest neighbors
            distances, indices = knn.kneighbors(test_point)
            neighbors = X[indices[0]]
            neighbors_vis = neighbors[:, feature_indices]
            
            # Add jitter to neighbors
            neighbors_jittered = neighbors_vis + np.random.normal(0, jitter_strength, neighbors_vis.shape)
            
            # Plot neighbors with special styling
            self.ax.scatter(neighbors_jittered[:, 0], neighbors_jittered[:, 1], 
                           c='yellow', s=200, edgecolors='black', linewidth=2, 
                           label=f'K={knn.n_neighbors} neighbors', zorder=4)
            
            # Draw lines to neighbors with different colors
            for i, neighbor_idx in enumerate(indices[0]):
                neighbor_class = y[neighbor_idx]
                line_color = self.get_random_color(neighbor_class)
                
                self.ax.plot([test_point_jittered[0, 0], neighbors_jittered[i, 0]],
                            [test_point_jittered[0, 1], neighbors_jittered[i, 1]],
                            color=line_color, alpha=0.7, linewidth=2, linestyle='--')
            
            self.ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
            self.ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
            self.ax.set_title(f'KNN Neighbors Example (K={knn.n_neighbors})', fontsize=14, fontweight='bold')
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.grid(True, alpha=0.3)
            self.fig.tight_layout()
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f'Error drawing neighbors: {str(e)}', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.axis('off')
    
    def draw_k_optimization(self, k_range, k_scores, optimal_k):
        """Draw K optimization results"""
        self.ax.clear()
        
        self.ax.plot(k_range, k_scores, 'b-', linewidth=2, marker='o', markersize=6)
        self.ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Optimal K = {optimal_k}')
        
        self.ax.set_xlabel('K Value', fontsize=12)
        self.ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
        self.ax.set_title('K Value Optimization', fontsize=14, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xticks(k_range[::max(1, len(k_range)//10)])  # Show some x-ticks
        
        self.fig.tight_layout()

class KNNApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.knn_worker = None
        self.k_optimizer = None
        self.visualizer = None
        self.current_knn = None
        self.feature_names = []
        self.original_df = None
        self.target_encoder = None
        self.region_mapping = {}
        self.init_ui()
        self.load_initial_data()
        
    def init_ui(self):
        self.setWindowTitle('Advanced KNN Explorer - Smart K Optimization')
        
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        window_width = min(1400, screen_size.width() - 50)
        window_height = min(900, screen_size.height() - 50)
        self.setGeometry(50, 50, window_width, window_height)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(450)
        control_layout = QVBoxLayout(control_panel)
        
        # Dataset info
        data_group = QGroupBox("Dataset Configuration")
        data_layout = QVBoxLayout(data_group)
        
        self.load_btn = QPushButton("Load Processed Data")
        self.load_btn.clicked.connect(self.load_dataset)
        data_layout.addWidget(self.load_btn)
        
        self.data_info = QLabel("Loading bmw_processed.csv...")
        self.data_info.setStyleSheet("QLabel { background-color: #e8f4fd; padding: 10px; border-radius: 5px; }")
        self.data_info.setWordWrap(True)
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
        
        # KNN Parameters
        knn_group = QGroupBox("KNN Parameters")
        knn_layout = QVBoxLayout(knn_group)
        
        # K value with optimization
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K (Neighbors):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 100)
        self.k_spin.setValue(5)
        k_layout.addWidget(self.k_spin)
        
        self.optimize_k_btn = QPushButton("Find Optimal K")
        self.optimize_k_btn.clicked.connect(self.find_optimal_k)
        self.optimize_k_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        k_layout.addWidget(self.optimize_k_btn)
        knn_layout.addLayout(k_layout)
        
        k_info_layout = QHBoxLayout()
        self.k_info_label = QLabel("Click 'Find Optimal K' for smart K selection")
        self.k_info_label.setStyleSheet("QLabel { color: #2196F3; font-size: 11px; font-weight: bold; }")
        k_info_layout.addWidget(self.k_info_label)
        knn_layout.addLayout(k_info_layout)
        
        # Other parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Weights:"))
        self.weights_combo = QComboBox()
        self.weights_combo.addItems(['uniform', 'distance'])
        params_layout.addWidget(self.weights_combo)
        
        params_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['auto', 'ball_tree', 'kd_tree', 'brute'])
        params_layout.addWidget(self.algorithm_combo)
        knn_layout.addLayout(params_layout)
        
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Distance:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        metric_layout.addWidget(self.metric_combo)
        knn_layout.addLayout(metric_layout)
        
        control_layout.addWidget(knn_group)
        
        # Feature Selection
        feature_group = QGroupBox("Feature Selection for Visualization")
        feature_layout = QVBoxLayout(feature_group)
        
        feature_layout.addWidget(QLabel("Feature 1 (X-axis):"))
        self.feature1_combo = QComboBox()
        self.feature1_combo.currentTextChanged.connect(self.on_feature_change)
        feature_layout.addWidget(self.feature1_combo)
        
        feature_layout.addWidget(QLabel("Feature 2 (Y-axis):"))
        self.feature2_combo = QComboBox()
        self.feature2_combo.currentTextChanged.connect(self.on_feature_change)
        feature_layout.addWidget(self.feature2_combo)
        
        control_layout.addWidget(feature_group)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to explore KNN")
        self.progress_label.setStyleSheet("QLabel { background-color: #fff3cd; padding: 8px; border-radius: 5px; }")
        control_layout.addWidget(self.progress_label)
        
        # Control buttons
        btn_group = QGroupBox("Controls")
        btn_layout = QVBoxLayout(btn_group)
        
        self.train_btn = QPushButton("🚀 Train KNN Model")
        self.train_btn.clicked.connect(self.train_knn)
        self.train_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        btn_layout.addWidget(self.train_btn)
        
        viz_btn_layout = QHBoxLayout()
        self.viz_mode_btn = QPushButton("Switch to Decision Boundary")
        self.viz_mode_btn.clicked.connect(self.switch_visualization)
        viz_btn_layout.addWidget(self.viz_mode_btn)
        
        self.next_sample_btn = QPushButton("Next Sample")
        self.next_sample_btn.clicked.connect(self.next_sample)
        self.next_sample_btn.setEnabled(False)
        viz_btn_layout.addWidget(self.next_sample_btn)
        btn_layout.addLayout(viz_btn_layout)
        
        control_layout.addWidget(btn_group)
        
        # Results
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(250)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        control_layout.addWidget(results_group)
        
        main_layout.addWidget(control_panel)
        
        # Right visualization area with tabs
        viz_tabs = QTabWidget()
        
        # Main visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.visualizer = KNNVisualizer(self.fig, self.ax)
        
        self.ax.text(0.5, 0.5, 'Load dataset and train KNN\n to see interactive visualizations', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.axis('off')
        self.canvas.draw()
        
        viz_layout.addWidget(self.canvas)
        viz_tabs.addTab(viz_tab, "KNN Visualization")
        
        # Results details tab
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        viz_tabs.addTab(details_tab, "Detailed Results")
        
        main_layout.addWidget(viz_tabs)
        
        self.visualization_mode = 'feature_space'
        self.sample_index = 0

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV Dataset", "", "CSV Files (*.csv)")
        if file_path:
            self.load_dataset_file(file_path)

    def load_initial_data(self):
        default_path = 'bmw_processed.csv'
        if os.path.exists(default_path):
            self.load_dataset_file(default_path)
        else:
            self.data_info.setText("bmw_processed.csv not found - Click 'Load Processed Data' to load a dataset")

    def load_dataset_file(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            self.original_df = self.df.copy()
            
            # Check if dataset has required columns
            if 'Target' not in self.df.columns:
                QMessageBox.critical(self, "Error", "Dataset must contain 'Target' column")
                return
            
            # Separate features and target
            self.feature_names = [col for col in self.df.columns if col != 'Target']
            
            if not self.feature_names:
                QMessageBox.critical(self, "Error", "No features found in dataset")
                return
            
            target_values = self.df['Target'].values
            
            # Define region mapping based on our preprocessing
            self.region_mapping = {
                0: 'Asia', 1: 'Europe', 2: 'Middle East', 
                3: 'North America', 4: 'South America', 5: 'Oceania'
            }
            
            n_classes = len(np.unique(target_values))
            max_recommended_k = min(15, len(self.df) // 10)
            self.k_info_label.setText(f"Classes: {n_classes} | Recommended K: 3-{max_recommended_k}")
            
            # Adjust K spin range based on dataset size
            max_k = min(100, len(self.df) - 1)
            self.k_spin.setRange(1, max_k)
            if self.k_spin.value() > max_k:
                self.k_spin.setValue(max_k)
            
            self.data_info.setText(f"Dataset: {os.path.basename(file_path)}\nSamples: {self.df.shape[0]:,}\nFeatures: {len(self.feature_names)}\nTarget classes: {n_classes}")
            
            # Update feature combos
            self.feature1_combo.clear()
            self.feature2_combo.clear()
            self.feature1_combo.addItems(self.feature_names)
            self.feature2_combo.addItems(self.feature_names)
            
            if len(self.feature_names) >= 2:
                self.feature1_combo.setCurrentIndex(0)
                self.feature2_combo.setCurrentIndex(1 if len(self.feature_names) > 1 else 0)
            
            self.train_btn.setEnabled(True)
            self.optimize_k_btn.setEnabled(True)
            
            # Show initial feature space
            self.show_initial_feature_space()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def show_initial_feature_space(self):
        if self.df is not None and len(self.feature_names) >= 2:
            X = self.df[self.feature_names].values
            y = self.df['Target'].values
            
            feature1_idx = self.feature1_combo.currentIndex()
            feature2_idx = self.feature2_combo.currentIndex()
            
            if feature1_idx < len(self.feature_names) and feature2_idx < len(self.feature_names):
                self.visualizer.draw_feature_space(
                    X, y, self.feature_names, [feature1_idx, feature2_idx],
                    "Feature Space Visualization"
                )
                self.canvas.draw()

    def on_feature_change(self):
        """Handle feature selection changes"""
        if (self.feature1_combo.currentText() == self.feature2_combo.currentText() and 
            self.feature1_combo.count() > 1):
            # Prevent same feature selection
            features = [self.feature1_combo.itemText(i) for i in range(self.feature1_combo.count())]
            current_feature1 = self.feature1_combo.currentText()
            other_features = [f for f in features if f != current_feature1]
            if other_features:
                self.feature2_combo.setCurrentText(other_features[0])
        
        self.update_visualization()

    def update_visualization(self):
        if self.df is None:
            return
            
        feature1_idx = self.feature1_combo.currentIndex()
        feature2_idx = self.feature2_combo.currentIndex()
        
        if feature1_idx >= len(self.feature_names) or feature2_idx >= len(self.feature_names):
            return
            
        if self.visualization_mode == 'feature_space':
            self.show_initial_feature_space()
        elif self.current_knn is not None:
            if self.visualization_mode == 'boundary':
                self.visualizer.draw_decision_boundary(
                    self.current_knn, self.X_test, self.y_test, 
                    self.feature_names, [feature1_idx, feature2_idx]
                )
            elif self.visualization_mode == 'neighbors':
                self.visualizer.draw_neighbors_example(
                    self.current_knn, self.X_test, self.y_test, 
                    self.feature_names, [feature1_idx, feature2_idx], self.sample_index
                )
            
            self.canvas.draw()

    def find_optimal_k(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        try:
            X = self.df[self.feature_names].values
            y = self.df['Target'].values
            
            train_size = self.train_size_spin.value() / 100.0
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=1-train_size, random_state=42, stratify=y
            )
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.optimize_k_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            
            self.k_optimizer = KOptimizationWorker(X_train, y_train)
            self.k_optimizer.update_progress.connect(self.on_optimization_update)
            self.k_optimizer.finished.connect(self.on_optimization_finished)
            self.k_optimizer.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"K optimization failed: {str(e)}")

    def on_optimization_update(self, progress, message, k_scores):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"{message} - {progress}%")

    def on_optimization_finished(self, results):
        self.optimize_k_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        optimal_k = results['optimal_k']
        optimal_score = results['optimal_score']
        
        self.k_spin.setValue(optimal_k)
        self.k_info_label.setText(f"Optimal K: {optimal_k} (Accuracy: {optimal_score:.3f})")
        
        # Show optimization results
        self.visualizer.draw_k_optimization(
            results['k_range'], results['k_scores'], optimal_k
        )
        self.canvas.draw()
        
        self.progress_label.setText(f"✅ Optimal K found: {optimal_k} with accuracy {optimal_score:.3f}")

    def next_sample(self):
        if hasattr(self, 'X_test') and len(self.X_test) > 0:
            self.sample_index = (self.sample_index + 1) % len(self.X_test)
            self.update_visualization()

    def train_knn(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        try:
            X = self.df[self.feature_names].values
            y = self.df['Target'].values
            
            if len(X) == 0:
                QMessageBox.warning(self, "Warning", "No data available for training!")
                return
            
            k_value = self.k_spin.value()
            
            # Validate K value
            if k_value >= len(X):
                QMessageBox.warning(self, "Warning", 
                                  f"K value ({k_value}) cannot be greater than or equal to number of samples ({len(X)})")
                return
            
            train_size = self.train_size_spin.value() / 100.0
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=1-train_size, random_state=42, stratify=y
            )
            
            # Ensure we have enough training samples
            if len(self.X_train) < k_value:
                QMessageBox.warning(self, "Warning", 
                                  f"Not enough training samples ({len(self.X_train)}) for K={k_value}")
                return
            
            n_neighbors = self.k_spin.value()
            weights = self.weights_combo.currentText()
            algorithm = self.algorithm_combo.currentText()
            metric = self.metric_combo.currentText()
            leaf_size = 30
            task_type = 'classification'
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.train_btn.setEnabled(False)
            self.optimize_k_btn.setEnabled(False)
            
            self.knn_worker = KNNWorker(
                self.X_train, self.X_test, self.y_train, self.y_test,
                n_neighbors, weights, algorithm, metric, leaf_size, task_type
            )
            self.knn_worker.update_progress.connect(self.on_training_update)
            self.knn_worker.finished.connect(self.on_training_finished)
            self.knn_worker.start()
            
            self.results_text.setText("🚀 Training KNN Model...\n\nThis may take a moment for large datasets.")
            
        except Exception as e:
            self.results_text.setText(f"❌ Error: {str(e)}")
            self.train_btn.setEnabled(True)
            self.optimize_k_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_training_update(self, progress, message, metrics):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"{message} - {progress}%")

    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        self.optimize_k_btn.setEnabled(True)
        self.next_sample_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("✅ Training Complete!")
        
        self.current_knn = results['knn']
        metrics = results['metrics']
        
        self.visualization_mode = 'boundary'
        self.update_visualization()
        
        # Enhanced results display
        results_text = "🎉 KNN Training Complete!\n\n"
        results_text += "📊 Model Parameters:\n"
        results_text += f"   • K Value: {self.k_spin.value()}\n"
        results_text += f"   • Weights: {self.weights_combo.currentText()}\n"
        results_text += f"   • Algorithm: {self.algorithm_combo.currentText()}\n"
        results_text += f"   • Distance: {self.metric_combo.currentText()}\n\n"
        
        results_text += "📈 Performance Metrics:\n"
        results_text += f"   • Accuracy: {metrics['accuracy_percent']}\n"
        results_text += f"   • Avg Confidence: {metrics['avg_confidence']:.3f}\n"
        results_text += f"   • Training Samples: {len(self.X_train):,}\n"
        results_text += f"   • Test Samples: {len(self.X_test):,}\n\n"
        
        # K value analysis
        k_value = self.k_spin.value()
        n_classes = len(np.unique(self.y_test))
        if k_value > len(self.X_train) // n_classes:
            results_text += f"⚠️  Note: K={k_value} may be too large (risk of underfitting)\n"
        elif k_value == 1:
            results_text += f"⚠️  Note: K=1 may overfit to noise\n"
        else:
            results_text += "✅ K value appears reasonable\n"
        
        results_text += f"\n💡 Try different K values or use 'Find Optimal K' for better performance!"
        
        self.results_text.setText(results_text)
        
        # Update details tab
        details = "Detailed Classification Report:\n\n"
        details += metrics['report']
        details += f"\nConfusion Matrix:\n{metrics['confusion_matrix']}"
        self.details_text.setText(details)

    def switch_visualization(self):
        if self.current_knn is None:
            self.visualization_mode = 'feature_space'
            self.show_initial_feature_space()
            self.viz_mode_btn.setText("Switch to Decision Boundary")
        else:
            if self.visualization_mode == 'feature_space':
                self.visualization_mode = 'boundary'
                self.viz_mode_btn.setText("Switch to Neighbors View")
            elif self.visualization_mode == 'boundary':
                self.visualization_mode = 'neighbors'
                self.viz_mode_btn.setText("Switch to Feature Space")
            else:
                self.visualization_mode = 'feature_space'
                self.viz_mode_btn.setText("Switch to Decision Boundary")
            
            self.update_visualization()

    def closeEvent(self, event):
        """Ensure threads are stopped when closing the application"""
        if self.knn_worker and self.knn_worker.isRunning():
            self.knn_worker.quit()
            self.knn_worker.wait()
        
        if self.k_optimizer and self.k_optimizer.isRunning():
            self.k_optimizer.quit()
            self.k_optimizer.wait()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = KNNApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()