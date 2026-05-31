# main.py
import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 instead of Wayland for compatibility

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ============================================
# SELF-TRAINING CLASSIFIER
# ============================================
class SelfTrainingClassifier:
    """Self-training semi-supervised classifier"""
    
    def __init__(self, base_classifier, threshold=0.85, max_iter=50):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.max_iter = max_iter
        self.final_classifier = None
        self.history = []
        self.labeled_indices = None
        self.unlabeled_indices = None
        
    def fit(self, X_labeled, y_labeled, X_unlabeled, y_true_unlabeled=None):
        """Self-training algorithm implementation"""
        X_labeled_current = X_labeled.copy()
        y_labeled_current = y_labeled.copy()
        X_unlabeled_current = X_unlabeled.copy()
        
        # Store original indices for tracking
        self.labeled_indices = list(range(len(X_labeled_current)))
        self.unlabeled_indices = list(range(len(X_labeled_current), 
                                           len(X_labeled_current) + len(X_unlabeled_current)))
        
        # Store true labels for evaluation if available
        self.y_true_unlabeled = y_true_unlabeled
        
        # Training history storage
        self.history = []
        
        for iteration in range(self.max_iter):
            if len(X_unlabeled_current) == 0:
                self.history.append({
                    'iteration': iteration,
                    'labeled_samples': len(X_labeled_current),
                    'unlabeled_samples': 0,
                    'new_labels_added': 0,
                    'avg_confidence': 0,
                    'accuracy_on_unlabeled': None
                })
                break
            
            # Train classifier on current labeled data
            self.base_classifier.fit(X_labeled_current, y_labeled_current)
            
            # Predict probabilities for unlabeled data
            if hasattr(self.base_classifier, 'predict_proba'):
                proba = self.base_classifier.predict_proba(X_unlabeled_current)
                max_proba = np.max(proba, axis=1)
                
                # Find high confidence predictions
                high_conf_idx = np.where(max_proba >= self.threshold)[0]
                
                # Calculate accuracy on unlabeled data if true labels are available
                accuracy_on_unlabeled = None
                if y_true_unlabeled is not None:
                    predictions = self.base_classifier.predict(X_unlabeled_current)
                    accuracy_on_unlabeled = accuracy_score(y_true_unlabeled, predictions)
                
                if len(high_conf_idx) > 0:
                    # Get predictions for high confidence samples
                    preds = self.base_classifier.predict(X_unlabeled_current[high_conf_idx])
                    
                    # Add to labeled set
                    X_labeled_current = np.vstack([X_labeled_current, 
                                                  X_unlabeled_current[high_conf_idx]])
                    y_labeled_current = np.hstack([y_labeled_current, preds])
                    
                    # Remove from unlabeled set
                    X_unlabeled_current = np.delete(X_unlabeled_current, high_conf_idx, axis=0)
                    
                    # Update indices
                    moved_indices = [self.unlabeled_indices[i] for i in high_conf_idx]
                    self.labeled_indices.extend(moved_indices)
                    self.unlabeled_indices = [i for i in self.unlabeled_indices 
                                             if i not in moved_indices]
                    
                    # Store history
                    self.history.append({
                        'iteration': iteration,
                        'labeled_samples': len(X_labeled_current),
                        'unlabeled_samples': len(X_unlabeled_current),
                        'new_labels_added': len(high_conf_idx),
                        'avg_confidence': np.mean(max_proba[high_conf_idx]),
                        'max_confidence': np.max(max_proba[high_conf_idx]),
                        'min_confidence': np.min(max_proba[high_conf_idx]),
                        'accuracy_on_unlabeled': accuracy_on_unlabeled
                    })
                else:
                    # No high confidence samples found
                    self.history.append({
                        'iteration': iteration,
                        'labeled_samples': len(X_labeled_current),
                        'unlabeled_samples': len(X_unlabeled_current),
                        'new_labels_added': 0,
                        'avg_confidence': 0,
                        'max_confidence': 0,
                        'min_confidence': 0,
                        'accuracy_on_unlabeled': accuracy_on_unlabeled
                    })
                    
                    # Stop if no progress for several iterations
                    if iteration > 5 and all(h['new_labels_added'] == 0 for h in self.history[-3:]):
                        break
            else:
                # For classifiers without predict_proba
                break
        
        # Train final classifier on all labeled data
        self.final_classifier = self.base_classifier
        self.final_classifier.fit(X_labeled_current, y_labeled_current)
        
        return self
    
    def predict(self, X):
        if self.final_classifier is not None:
            return self.final_classifier.predict(X)
        return None
    
    def predict_proba(self, X):
        if self.final_classifier is not None and hasattr(self.final_classifier, 'predict_proba'):
            return self.final_classifier.predict_proba(X)
        return None

# ============================================
# MAIN APPLICATION WINDOW
# ============================================
class SelfTrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.processed_data = None
        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.y_true_unlabeled = None
        self.target_column = None
        self.original_labels = None
        self.predicted_labels = None
        self.feature_names = None
        
        # Preprocessing objects
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Models
        self.self_training_model = None
        self.supervised_model = None
        
        # UI initialization
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Semi-Supervised Learning: Self-Training Demo")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                font-size: 12px;
                color: #333333;
            }
            QPushButton {
                background-color: #4a6fa5;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3a5a8c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton#danger {
                background-color: #e74c3c;
            }
            QPushButton#danger:hover {
                background-color: #c0392b;
            }
            QPushButton#success {
                background-color: #27ae60;
            }
            QPushButton#success:hover {
                background-color: #219653;
            }
            QPushButton#warning {
                background-color: #f39c12;
            }
            QPushButton#warning:hover {
                background-color: #d68910;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #dcdde1;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                background-color: #ecf0f1;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #dcdde1;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #7f8c8d;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover {
                background-color: #dfe6e9;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                min-height: 30px;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #3498db;
            }
            QTextEdit, QPlainTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Monospace', 'Courier New';
                font-size: 11px;
            }
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #e3f2fd;
                gridline-color: #dee2e6;
            }
            QHeaderView::section {
                background-color: #4a6fa5;
                color: white;
                padding: 8px;
                border: 1px solid #3a5a8c;
                font-weight: bold;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 6px;
                background: #ecf0f1;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 20px;
                height: 20px;
                margin: -8px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_dataset_tab()
        self.create_preprocessing_tab()
        self.create_training_tab()
        self.create_results_tab()
        self.create_comparison_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to load dataset")
        
        # Show window
        self.show()
        
    def create_dataset_tab(self):
        """Create dataset selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Semi-Supervised Learning: Self-Training Algorithm")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Dataset selection group
        dataset_group = QGroupBox("1. Dataset Selection")
        dataset_layout = QVBoxLayout()
        
        # Dataset description
        desc_label = QLabel(
            "This application demonstrates self-training semi-supervised learning. "
            "We use the Bank Marketing dataset to predict if clients subscribe to term deposits. "
            "You can remove labels to simulate unlabeled data and recover them using self-training."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("padding: 5px; background-color: #e8f4fd; border-radius: 4px;")
        dataset_layout.addWidget(desc_label)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Dataset file:"))
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select bank-additional-full.csv or bank-full.csv")
        file_layout.addWidget(self.file_path_edit, 1)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_dataset)
        file_layout.addWidget(self.browse_button)
        
        dataset_layout.addLayout(file_layout)
        
        # Or use default dataset
        default_button_layout = QHBoxLayout()
        default_button_layout.addStretch()
        
        self.load_bank_additional_button = QPushButton("Load Bank-Additional Dataset")
        self.load_bank_additional_button.clicked.connect(lambda: self.load_default_dataset("additional"))
        default_button_layout.addWidget(self.load_bank_additional_button)
        
        self.load_bank_button = QPushButton("Load Bank Dataset")
        self.load_bank_button.clicked.connect(lambda: self.load_default_dataset("bank"))
        default_button_layout.addWidget(self.load_bank_button)
        
        dataset_layout.addLayout(default_button_layout)
        
        # Dataset info
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 4px;")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_info_label)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Label removal settings
        removal_group = QGroupBox("2. Label Removal Settings")
        removal_layout = QGridLayout()
        
        # Label removal percentage
        removal_layout.addWidget(QLabel("Percentage of labels to remove:"), 0, 0)
        
        self.unlabeled_slider = QSlider(Qt.Horizontal)
        self.unlabeled_slider.setRange(10, 95)
        self.unlabeled_slider.setValue(70)
        self.unlabeled_slider.setTickPosition(QSlider.TicksBelow)
        self.unlabeled_slider.setTickInterval(10)
        self.unlabeled_slider.valueChanged.connect(self.update_removal_percentage)
        removal_layout.addWidget(self.unlabeled_slider, 0, 1)
        
        self.unlabeled_percentage_label = QLabel("70%")
        self.unlabeled_percentage_label.setStyleSheet("font-weight: bold; color: #e74c3c; min-width: 50px;")
        removal_layout.addWidget(self.unlabeled_percentage_label, 0, 2)
        
        # Explanation
        explanation_label = QLabel(
            f"With 70% removal: From 41,188 samples → {int(41188*0.3):,} labeled, {int(41188*0.7):,} unlabeled"
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        removal_layout.addWidget(explanation_label, 1, 0, 1, 3)
        
        removal_group.setLayout(removal_layout)
        layout.addWidget(removal_group)
        
        # Dataset preview
        preview_group = QGroupBox("3. Dataset Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(300)
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Next button
        next_button_layout = QHBoxLayout()
        next_button_layout.addStretch()
        
        self.next_to_preprocess_button = QPushButton("Next: Preprocessing →")
        self.next_to_preprocess_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.next_to_preprocess_button.setEnabled(False)
        next_button_layout.addWidget(self.next_to_preprocess_button)
        
        layout.addLayout(next_button_layout)
        
        self.tab_widget.addTab(tab, "📂 Dataset")
        
    def create_preprocessing_tab(self):
        """Create data preprocessing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Preprocessing controls
        controls_group = QGroupBox("1. Preprocessing Settings")
        controls_layout = QGridLayout()
        
        # Target column selection
        controls_layout.addWidget(QLabel("Target Column:"), 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self.update_preprocess_status)
        controls_layout.addWidget(self.target_combo, 0, 1)
        
        # Show target distribution button
        self.show_distribution_button = QPushButton("Show Distribution")
        self.show_distribution_button.clicked.connect(self.show_target_distribution)
        self.show_distribution_button.setEnabled(False)
        controls_layout.addWidget(self.show_distribution_button, 0, 2)
        
        # Preprocessing options
        self.scale_checkbox = QCheckBox("Scale numerical features (StandardScaler)")
        self.scale_checkbox.setChecked(True)
        controls_layout.addWidget(self.scale_checkbox, 1, 0)
        
        self.encode_checkbox = QCheckBox("Encode categorical variables (OneHot/Label)")
        self.encode_checkbox.setChecked(True)
        controls_layout.addWidget(self.encode_checkbox, 1, 1)
        
        self.impute_checkbox = QCheckBox("Impute missing values (Mean/Mode)")
        self.impute_checkbox.setChecked(True)
        controls_layout.addWidget(self.impute_checkbox, 1, 2)
        
        # Status indicator
        self.preprocess_status = QLabel("Status: Load dataset first")
        self.preprocess_status.setStyleSheet("padding: 10px; font-weight: bold;")
        controls_layout.addWidget(self.preprocess_status, 2, 0, 1, 3)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Dataset")
        self.analyze_button.clicked.connect(self.analyze_dataset)
        self.analyze_button.setEnabled(False)
        button_layout.addWidget(self.analyze_button)
        
        self.preprocess_button = QPushButton("Apply Preprocessing")
        self.preprocess_button.clicked.connect(self.apply_preprocessing)
        self.preprocess_button.setEnabled(False)
        self.preprocess_button.setObjectName("success")
        button_layout.addWidget(self.preprocess_button)
        
        controls_layout.addLayout(button_layout, 3, 0, 1, 3)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Data analysis tabs
        analysis_group = QGroupBox("2. Data Analysis")
        analysis_layout = QVBoxLayout()
        
        analysis_tabs = QTabWidget()
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        analysis_tabs.addTab(stats_tab, "📊 Statistics")
        
        # Missing values tab
        missing_tab = QWidget()
        missing_layout = QVBoxLayout(missing_tab)
        self.missing_table = QTableWidget()
        missing_layout.addWidget(self.missing_table)
        analysis_tabs.addTab(missing_tab, "⚠️ Missing Values")
        
        # Data types tab
        types_tab = QWidget()
        types_layout = QVBoxLayout(types_tab)
        self.types_table = QTableWidget()
        types_layout.addWidget(self.types_table)
        analysis_tabs.addTab(types_tab, "🔧 Data Types")
        
        analysis_layout.addWidget(analysis_tabs)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        
        self.prev_to_dataset_button = QPushButton("← Back to Dataset")
        self.prev_to_dataset_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        nav_layout.addWidget(self.prev_to_dataset_button)
        
        self.next_to_training_button = QPushButton("Next: Training →")
        self.next_to_training_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        self.next_to_training_button.setEnabled(False)
        nav_layout.addWidget(self.next_to_training_button)
        
        layout.addLayout(nav_layout)
        
        self.tab_widget.addTab(tab, "🔧 Preprocessing")
        
    def create_training_tab(self):
        """Create model training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Supervised model selection
        supervised_group = QGroupBox("1. Supervised Model Selection (Baseline)")
        supervised_layout = QGridLayout()
        
        supervised_layout.addWidget(QLabel("Select supervised model:"), 0, 0)
        self.supervised_model_combo = QComboBox()
        self.supervised_model_combo.addItems([
            "Random Forest",
            "Logistic Regression", 
            "Support Vector Machine",
            "Gradient Boosting",
            "Decision Tree"
        ])
        supervised_layout.addWidget(self.supervised_model_combo, 0, 1)
        
        self.train_supervised_button = QPushButton("Train Supervised Model")
        self.train_supervised_button.clicked.connect(self.train_supervised_model)
        self.train_supervised_button.setEnabled(False)
        supervised_layout.addWidget(self.train_supervised_button, 0, 2)
        
        self.supervised_results_label = QLabel("Not trained yet")
        self.supervised_results_label.setStyleSheet("padding: 5px; background-color: #fff3cd; border-radius: 4px;")
        supervised_layout.addWidget(self.supervised_results_label, 1, 0, 1, 3)
        
        supervised_group.setLayout(supervised_layout)
        layout.addWidget(supervised_group)
        
        # Self-training parameters
        self_training_group = QGroupBox("2. Self-Training Parameters")
        self_training_layout = QGridLayout()
        
        # Base classifier
        self_training_layout.addWidget(QLabel("Base Classifier:"), 0, 0)
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems([
            "Random Forest",
            "Logistic Regression", 
            "Support Vector Machine",
            "Gradient Boosting",
            "Decision Tree"
        ])
        self.classifier_combo.setCurrentText("Random Forest")
        self_training_layout.addWidget(self.classifier_combo, 0, 1)
        
        # Confidence threshold
        self_training_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 99)
        self.threshold_slider.setValue(85)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.85")
        self.threshold_label.setStyleSheet("font-weight: bold; color: #3498db; min-width: 50px;")
        threshold_layout.addWidget(self.threshold_label)
        
        self_training_layout.addLayout(threshold_layout, 1, 1)
        
        # Max iterations
        self_training_layout.addWidget(QLabel("Max Iterations:"), 2, 0)
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(1, 200)
        self.iter_spinbox.setValue(50)
        self.iter_spinbox.setSuffix(" iterations")
        self_training_layout.addWidget(self.iter_spinbox, 2, 1)
        
        # Training status
        self.training_status_label = QLabel("Preprocessing required before training")
        self.training_status_label.setStyleSheet("padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;")
        self_training_layout.addWidget(self.training_status_label, 3, 0, 1, 2)
        
        # Start button
        self.start_button = QPushButton("🚀 Start Self-Training")
        self.start_button.clicked.connect(self.start_self_training)
        self.start_button.setEnabled(False)
        self.start_button.setObjectName("warning")
        self_training_layout.addWidget(self.start_button, 4, 0, 1, 2)
        
        self_training_group.setLayout(self_training_layout)
        layout.addWidget(self_training_group)
        
        # Training progress
        progress_group = QGroupBox("3. Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        progress_layout.addWidget(self.progress_text)
        
        # Progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        progress_layout.addWidget(self.training_progress)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Visualization
        viz_group = QGroupBox("4. Training Visualization")
        viz_layout = QVBoxLayout()
        
        # Matplotlib figure for training progress
        self.training_figure = plt.figure(figsize=(10, 4))
        self.training_canvas = FigureCanvas(self.training_figure)
        self.training_toolbar = NavigationToolbar(self.training_canvas, self)
        
        viz_layout.addWidget(self.training_toolbar)
        viz_layout.addWidget(self.training_canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        
        self.prev_to_preprocess_button = QPushButton("← Back to Preprocessing")
        self.prev_to_preprocess_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        nav_layout.addWidget(self.prev_to_preprocess_button)
        
        self.next_to_results_button = QPushButton("Next: Results →")
        self.next_to_results_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        self.next_to_results_button.setEnabled(False)
        nav_layout.addWidget(self.next_to_results_button)
        
        layout.addLayout(next_button_layout)
        
        self.tab_widget.addTab(tab, "⚙️ Training")
        
    def create_results_tab(self):
        """Create results visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Results summary
        summary_group = QGroupBox("1. Results Summary")
        summary_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        summary_layout.addWidget(self.results_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Detailed visualizations
        viz_group = QGroupBox("2. Detailed Analysis")
        viz_layout = QVBoxLayout()
        
        results_tabs = QTabWidget()
        
        # Confusion Matrix tab
        cm_tab = QWidget()
        cm_layout = QVBoxLayout(cm_tab)
        self.cm_figure = plt.figure(figsize=(6, 5))
        self.cm_canvas = FigureCanvas(self.cm_figure)
        cm_layout.addWidget(self.cm_canvas)
        results_tabs.addTab(cm_tab, "📉 Confusion Matrix")
        
        # Training Progress tab
        progress_tab = QWidget()
        progress_layout = QVBoxLayout(progress_tab)
        self.progress_figure = plt.figure(figsize=(6, 5))
        self.progress_canvas = FigureCanvas(self.progress_figure)
        progress_layout.addWidget(self.progress_canvas)
        results_tabs.addTab(progress_tab, "📈 Training Progress")
        
        # Confidence Distribution tab
        conf_tab = QWidget()
        conf_layout = QVBoxLayout(conf_tab)
        self.conf_figure = plt.figure(figsize=(6, 5))
        self.conf_canvas = FigureCanvas(self.conf_figure)
        conf_layout.addWidget(self.conf_canvas)
        results_tabs.addTab(conf_tab, "📊 Confidence Distribution")
        
        viz_layout.addWidget(results_tabs)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_results_button = QPushButton("💾 Export Results to CSV")
        self.export_results_button.clicked.connect(self.export_results)
        self.export_results_button.setEnabled(False)
        export_layout.addWidget(self.export_results_button)
        
        self.export_report_button = QPushButton("📄 Export Full Report")
        self.export_report_button.clicked.connect(self.export_full_report)
        self.export_report_button.setEnabled(False)
        export_layout.addWidget(self.export_report_button)
        
        export_layout.addStretch()
        
        self.next_to_comparison_button = QPushButton("Next: Comparison →")
        self.next_to_comparison_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(4))
        self.next_to_comparison_button.setEnabled(False)
        export_layout.addWidget(self.next_to_comparison_button)
        
        layout.addLayout(export_layout)
        
        self.tab_widget.addTab(tab, "📈 Results")
        
    def create_comparison_tab(self):
        """Create comparison tab between true and predicted labels"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Comparison title
        title_label = QLabel("Label Recovery Comparison")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; text-align: center;")
        layout.addWidget(title_label)
        
        # Comparison metrics
        metrics_group = QGroupBox("Recovery Metrics")
        metrics_layout = QGridLayout()
        
        self.accuracy_label = QLabel("Accuracy: N/A")
        self.accuracy_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.accuracy_label, 0, 0)
        
        self.precision_label = QLabel("Precision: N/A")
        self.precision_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.precision_label, 0, 1)
        
        self.recall_label = QLabel("Recall: N/A")
        self.recall_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.recall_label, 0, 2)
        
        self.f1_label = QLabel("F1-Score: N/A")
        self.f1_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_layout.addWidget(self.f1_label, 0, 3)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Comparison table
        table_group = QGroupBox("Label Comparison (First 100 Samples)")
        table_layout = QVBoxLayout()
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setMaximumHeight(400)
        table_layout.addWidget(self.comparison_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Visual comparison
        viz_group = QGroupBox("Visual Comparison")
        viz_layout = QVBoxLayout()
        
        self.comparison_figure = plt.figure(figsize=(10, 6))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        viz_layout.addWidget(self.comparison_canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Navigation
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        
        self.prev_to_results_button = QPushButton("← Back to Results")
        self.prev_to_results_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        nav_layout.addWidget(self.prev_to_results_button)
        
        layout.addLayout(nav_layout)
        
        self.tab_widget.addTab(tab, "🔍 Comparison")
        
    # ============================================
    # DATASET METHODS
    # ============================================
    
    def browse_dataset(self):
        """Browse for dataset file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_dataset(file_path)
    
    def load_default_dataset(self, dataset_type):
        """Load default bank marketing dataset"""
        if dataset_type == "additional":
            # Try to find bank-additional-full.csv in common locations
            possible_paths = [
                "bank-additional-full.csv",
                "bank+marketing/bank-additional/bank-additional-full.csv",
                "bank+marketing/bank-additional/bank-additional/bank-additional-full.csv",
                "../bank+marketing/bank-additional/bank-additional-full.csv"
            ]
        else:  # "bank"
            possible_paths = [
                "bank-full.csv",
                "bank+marketing/bank/bank-full.csv",
                "../bank+marketing/bank/bank-full.csv"
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.file_path_edit.setText(path)
                self.load_dataset(path)
                return
        
        # If not found, show error
        QMessageBox.warning(
            self, "File Not Found",
            f"Could not find {dataset_type} dataset. Please browse for the file manually."
        )
    
    def load_dataset(self, file_path):
        """Load dataset from file"""
        try:
            # Read CSV file
            if 'bank-full' in file_path:
                self.dataset = pd.read_csv(file_path, sep=';')
            elif 'bank-additional-full' in file_path:
                self.dataset = pd.read_csv(file_path, sep=';')
            else:
                self.dataset = pd.read_csv(file_path)
            
            # Update UI
            self.update_dataset_info()
            self.update_preview_table()
            self.update_target_combo()
            
            # Enable next button
            self.next_to_preprocess_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            
            self.status_bar.showMessage(f"Dataset loaded: {len(self.dataset)} samples, {len(self.dataset.columns)} features")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def update_dataset_info(self):
        """Update dataset information display"""
        info = f"""
        <b>Dataset Information:</b><br>
        • File: {os.path.basename(self.file_path_edit.text())}<br>
        • Samples: {len(self.dataset):,}<br>
        • Features: {len(self.dataset.columns)}<br>
        • Memory usage: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB<br>
        <br>
        <b>Target variable should be:</b><br>
        • For bank datasets: 'y' (yes/no for subscription)<br>
        <br>
        <b>First few columns:</b> {', '.join(self.dataset.columns[:5])}...
        """
        self.dataset_info_label.setText(info)
    
    def update_preview_table(self):
        """Update preview table with dataset head"""
        self.preview_table.clear()
        self.preview_table.setRowCount(min(10, len(self.dataset)))
        self.preview_table.setColumnCount(min(8, len(self.dataset.columns)))
        
        # Set headers
        headers = list(self.dataset.columns[:self.preview_table.columnCount()])
        self.preview_table.setHorizontalHeaderLabels(headers)
        
        # Fill data
        for i in range(self.preview_table.rowCount()):
            for j in range(self.preview_table.columnCount()):
                value = str(self.dataset.iloc[i, j])
                item = QTableWidgetItem(value)
                self.preview_table.setItem(i, j, item)
        
        self.preview_table.resizeColumnsToContents()
    
    def update_target_combo(self):
        """Update target column combo box"""
        self.target_combo.clear()
        self.target_combo.addItems(self.dataset.columns.tolist())
        
        # Auto-select likely target columns
        if 'y' in self.dataset.columns:
            self.target_combo.setCurrentText('y')
        elif 'target' in self.dataset.columns:
            self.target_combo.setCurrentText('target')
        elif 'class' in self.dataset.columns:
            self.target_combo.setCurrentText('class')
    
    def update_removal_percentage(self, value):
        """Update removal percentage label"""
        self.unlabeled_percentage_label.setText(f"{value}%")
        
        if self.dataset is not None:
            total_samples = len(self.dataset)
            labeled = int(total_samples * (100 - value) / 100)
            unlabeled = total_samples - labeled
            
            # Find and update explanation label
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "📂 Dataset":
                    widget = self.tab_widget.widget(i)
                    for child in widget.findChildren(QLabel):
                        if "With" in child.text() and "removal" in child.text():
                            child.setText(f"With {value}% removal: From {total_samples:,} samples → {labeled:,} labeled, {unlabeled:,} unlabeled")
                            break
    
    def update_threshold_label(self, value):
        """Update threshold label"""
        self.threshold_label.setText(f"{value/100:.2f}")
    
    def update_preprocess_status(self):
        """Update preprocessing status based on target column selection"""
        if self.dataset is None:
            self.preprocess_status.setText("Status: Load dataset first")
            self.preprocess_status.setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 10px;")
            return
        
        target = self.target_combo.currentText()
        if target:
            # Check if target column exists
            if target in self.dataset.columns:
                # Check data type
                dtype = self.dataset[target].dtype
                unique_count = self.dataset[target].nunique()
                
                status_text = f"Status: Target column '{target}' selected\n"
                status_text += f"• Data type: {dtype}\n"
                status_text += f"• Unique values: {unique_count}\n"
                
                if unique_count <= 10:
                    values = self.dataset[target].unique()[:10]
                    status_text += f"• Values: {', '.join(map(str, values))}"
                    if unique_count > 10:
                        status_text += "..."
                else:
                    status_text += "• Too many unique values for classification"
                
                if unique_count == 2:
                    status_text += "\n• ✅ Perfect for binary classification"
                    style = "background-color: #d4edda; color: #155724; padding: 10px;"
                elif 3 <= unique_count <= 10:
                    status_text += "\n• ⚠️ Suitable for multi-class classification"
                    style = "background-color: #fff3cd; color: #856404; padding: 10px;"
                else:
                    status_text += "\n• ❌ May not be suitable for classification"
                    style = "background-color: #f8d7da; color: #721c24; padding: 10px;"
                
                self.preprocess_status.setText(status_text)
                self.preprocess_status.setStyleSheet(style)
                self.show_distribution_button.setEnabled(True)
            else:
                self.preprocess_status.setText(f"Status: Column '{target}' not found in dataset")
                self.preprocess_status.setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 10px;")
        else:
            self.preprocess_status.setText("Status: Please select a target column")
            self.preprocess_status.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 10px;")
    
    # ============================================
    # ANALYSIS METHODS
    # ============================================
    
    def show_target_distribution(self):
        """Show distribution of target variable"""
        if self.dataset is None or self.target_combo.currentText() == "":
            return
        
        target = self.target_combo.currentText()
        distribution = self.dataset[target].value_counts()
        
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if distribution.shape[0] <= 10:
            # Bar plot for categorical
            bars = ax.bar([str(x) for x in distribution.index], distribution.values)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of Target Variable: {target}')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height):,}', ha='center', va='bottom')
        else:
            # Histogram for numerical
            ax.hist(self.dataset[target].dropna(), bins=30, edgecolor='black')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of Target Variable: {target}')
        
        plt.tight_layout()
        
        # Show in dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Target Distribution: {target}")
        dialog.setGeometry(100, 100, 900, 600)
        
        layout = QVBoxLayout(dialog)
        
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        
        dialog.exec_()
    
    def analyze_dataset(self):
        """Analyze dataset and show statistics"""
        if self.dataset is None:
            return
        
        try:
            # Update statistics
            stats_text = "<b>Dataset Statistics:</b><br><br>"
            stats_text += f"<b>Shape:</b> {self.dataset.shape[0]} rows × {self.dataset.shape[1]} columns<br><br>"
            
            # Basic info for each column
            stats_text += "<b>Column Information:</b><br>"
            for col in self.dataset.columns:
                dtype = self.dataset[col].dtype
                non_null = self.dataset[col].count()
                null_count = self.dataset[col].isnull().sum()
                unique = self.dataset[col].nunique()
                
                stats_text += f"• <b>{col}</b>: {dtype}, {non_null:,} non-null"
                if null_count > 0:
                    stats_text += f", {null_count:,} null ({null_count/len(self.dataset)*100:.1f}%)"
                if unique <= 20:
                    stats_text += f", {unique} unique"
                else:
                    stats_text += f", {unique:,} unique"
                stats_text += "<br>"
            
            # Numerical columns statistics
            numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_text += "<br><b>Numerical Columns Summary:</b><br>"
                numeric_stats = self.dataset[numeric_cols].describe().T
                stats_text += numeric_stats.to_html()
            
            self.stats_text.setHtml(stats_text)
            
            # Update missing values table
            missing_data = self.dataset.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            self.missing_table.clear()
            if len(missing_data) > 0:
                self.missing_table.setRowCount(len(missing_data))
                self.missing_table.setColumnCount(3)
                self.missing_table.setHorizontalHeaderLabels(['Column', 'Missing Count', 'Missing %'])
                
                for i, (col, count) in enumerate(missing_data.items()):
                    percentage = count / len(self.dataset) * 100
                    
                    self.missing_table.setItem(i, 0, QTableWidgetItem(col))
                    self.missing_table.setItem(i, 1, QTableWidgetItem(f"{count:,}"))
                    self.missing_table.setItem(i, 2, QTableWidgetItem(f"{percentage:.2f}%"))
            else:
                self.missing_table.setRowCount(1)
                self.missing_table.setColumnCount(1)
                self.missing_table.setHorizontalHeaderLabels(['Status'])
                self.missing_table.setItem(0, 0, QTableWidgetItem("✅ No missing values found"))
            
            self.missing_table.resizeColumnsToContents()
            
            # Update data types table
            self.types_table.clear()
            self.types_table.setRowCount(len(self.dataset.columns))
            self.types_table.setColumnCount(3)
            self.types_table.setHorizontalHeaderLabels(['Column', 'Data Type', 'Example Value'])
            
            for i, col in enumerate(self.dataset.columns):
                dtype = self.dataset[col].dtype
                example = str(self.dataset[col].iloc[0]) if len(self.dataset) > 0 and not pd.isna(self.dataset[col].iloc[0]) else "NaN"
                if len(example) > 30:
                    example = example[:27] + "..."
                
                self.types_table.setItem(i, 0, QTableWidgetItem(col))
                self.types_table.setItem(i, 1, QTableWidgetItem(str(dtype)))
                self.types_table.setItem(i, 2, QTableWidgetItem(example))
            
            self.types_table.resizeColumnsToContents()
            
            # Enable preprocessing button
            self.preprocess_button.setEnabled(True)
            self.preprocess_status.setText("Status: Ready for preprocessing")
            self.preprocess_status.setStyleSheet("background-color: #d4edda; color: #155724; padding: 10px;")
            
        except Exception as e:
            QMessageBox.warning(self, "Analysis Error", f"Error analyzing dataset: {str(e)}")
    
    # ============================================
    # PREPROCESSING METHODS
    # ============================================
    
    def apply_preprocessing(self):
        """Apply preprocessing to dataset"""
        if self.dataset is None:
            return
        
        try:
            target = self.target_combo.currentText()
            if not target:
                QMessageBox.warning(self, "Warning", "Please select a target column first")
                return
            
            # Store original data for comparison
            self.original_data = self.dataset.copy()
            self.target_column = target
            
            # Separate features and target
            X = self.dataset.drop(columns=[target])
            y = self.dataset[target]
            
            # Store original labels for comparison
            self.original_labels = y.copy()
            
            # Encode target variable
            if self.encode_checkbox.isChecked():
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.class_names = self.label_encoder.classes_
            else:
                y_encoded = y.values
                self.class_names = np.unique(y_encoded)
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            # First, handle missing values
            if self.impute_checkbox.isChecked():
                # Impute numeric columns
                if len(numeric_cols) > 0:
                    numeric_imputer = SimpleImputer(strategy='mean')
                    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
                
                # Impute categorical columns
                if len(categorical_cols) > 0:
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
            
            # Then encode categorical variables
            if len(categorical_cols) > 0 and self.encode_checkbox.isChecked():
                try:
                    # One-hot encode categorical variables
                    self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    X_categorical = self.onehot_encoder.fit_transform(X[categorical_cols])
                    categorical_feature_names = self.onehot_encoder.get_feature_names_out(categorical_cols)
                    
                    # Combine with numerical features
                    if len(numeric_cols) > 0:
                        X_numeric = X[numeric_cols].values
                        X_processed = np.hstack([X_numeric, X_categorical])
                        self.feature_names = list(numeric_cols) + list(categorical_feature_names)
                    else:
                        X_processed = X_categorical
                        self.feature_names = list(categorical_feature_names)
                except Exception as e:
                    print(f"One-hot encoding warning: {e}")
                    # Fallback: label encode categorical variables
                    for col in categorical_cols:
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                    X_processed = X.values
                    self.feature_names = list(X.columns)
            else:
                # If not encoding categorical, convert to numeric if needed
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        X[col] = pd.factorize(X[col])[0]
                X_processed = X.values
                self.feature_names = list(X.columns)
            
            # Scale numerical features
            if self.scale_checkbox.isChecked():
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
            
            # Split into labeled and unlabeled based on removal percentage
            removal_percentage = self.unlabeled_slider.value() / 100.0
            
            # Ensure we have enough samples
            min_samples = max(100, int(len(X_processed) * 0.01))
            if len(X_processed) < min_samples * 2:
                QMessageBox.warning(self, "Warning", f"Dataset too small for {removal_percentage*100:.0f}% removal. Using smaller removal.")
                removal_percentage = 0.5
            
            # Split data - FIXED: Ensure consistent sample sizes
            X_labeled, X_unlabeled, y_labeled, y_true_unlabeled = train_test_split(
                X_processed, y_encoded, 
                test_size=removal_percentage, 
                random_state=42,
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            # Verify dimensions
            if len(X_labeled) != len(y_labeled):
                raise ValueError(f"X_labeled ({len(X_labeled)}) and y_labeled ({len(y_labeled)}) have different lengths")
            if len(X_unlabeled) != len(y_true_unlabeled):
                raise ValueError(f"X_unlabeled ({len(X_unlabeled)}) and y_true_unlabeled ({len(y_true_unlabeled)}) have different lengths")
            
            # Store processed data
            self.X_labeled = X_labeled
            self.y_labeled = y_labeled
            self.X_unlabeled = X_unlabeled
            self.y_true_unlabeled = y_true_unlabeled
            
            # Update status
            self.status_bar.showMessage(
                f"Preprocessing complete: {len(X_labeled):,} labeled, {len(X_unlabeled):,} unlabeled samples"
            )
            
            # Enable training buttons
            self.train_supervised_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.training_status_label.setText("✅ Ready for training")
            self.training_status_label.setStyleSheet("background-color: #d4edda; color: #155724; padding: 10px;")
            self.next_to_training_button.setEnabled(True)
            
            # Show success message
            QMessageBox.information(
                self, "Preprocessing Complete",
                f"Preprocessing completed successfully!\n\n"
                f"• Original samples: {len(X):,}\n"
                f"• Labeled samples: {len(X_labeled):,} ({100-removal_percentage*100:.1f}%)\n"
                f"• Unlabeled samples: {len(X_unlabeled):,} ({removal_percentage*100:.1f}%)\n"
                f"• Features: {X_processed.shape[1]}\n"
                f"• Classes: {len(self.class_names)}\n\n"
                f"<b>Dataset dimensions verified:</b>\n"
                f"• X_labeled: {X_labeled.shape}\n"
                f"• y_labeled: {y_labeled.shape}\n"
                f"• X_unlabeled: {X_unlabeled.shape}\n"
                f"• y_true_unlabeled: {y_true_unlabeled.shape}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", f"Error during preprocessing: {str(e)}\n\nFull error: {e}")
    
    # ============================================
    # TRAINING METHODS
    # ============================================
    
    def train_supervised_model(self):
        """Train supervised model as baseline"""
        if self.X_labeled is None:
            return
        
        try:
            model_name = self.supervised_model_combo.currentText()
            
            # Create model
            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == "Support Vector Machine":
                model = SVC(probability=True, random_state=42)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = RandomForestClassifier(random_state=42)
            
            # Train on labeled data
            model.fit(self.X_labeled, self.y_labeled)
            self.supervised_model = model
            
            # Evaluate on unlabeled data (if we have true labels)
            if self.y_true_unlabeled is not None and len(self.y_true_unlabeled) > 0:
                # Verify dimensions
                if len(self.X_unlabeled) != len(self.y_true_unlabeled):
                    raise ValueError(f"Dimension mismatch: X_unlabeled ({len(self.X_unlabeled)}) vs y_true_unlabeled ({len(self.y_true_unlabeled)})")
                
                predictions = model.predict(self.X_unlabeled)
                accuracy = accuracy_score(self.y_true_unlabeled, predictions)
                precision = precision_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                recall = recall_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                f1 = f1_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                
                results_text = f"""
                <b>{model_name} Results (Baseline):</b><br>
                • Accuracy: <b>{accuracy:.3%}</b><br>
                • Precision: <b>{precision:.3%}</b><br>
                • Recall: <b>{recall:.3%}</b><br>
                • F1-Score: <b>{f1:.3%}</b><br>
                <br>
                <i>Trained on {len(self.X_labeled):,} labeled samples only</i>
                """
                
                self.supervised_results_label.setText(results_text)
                self.supervised_results_label.setStyleSheet(
                    "background-color: #d4edda; color: #155724; padding: 10px; border-radius: 4px;"
                )
                
                # Store supervised predictions for comparison
                self.supervised_predictions = predictions
                
            self.status_bar.showMessage(f"Supervised {model_name} training complete")
            
        except Exception as e:
            QMessageBox.warning(self, "Training Error", f"Error training supervised model: {str(e)}")
    
    def start_self_training(self):
        """Start self-training algorithm"""
        if self.X_labeled is None or self.X_unlabeled is None:
            return
        
        try:
            # Clear previous results
            self.progress_text.clear()
            self.results_text.clear()
            
            # Get parameters
            model_name = self.classifier_combo.currentText()
            threshold = self.threshold_slider.value() / 100.0
            max_iter = self.iter_spinbox.value()
            
            # Create base classifier
            if model_name == "Random Forest":
                base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "Logistic Regression":
                base_model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == "Support Vector Machine":
                base_model = SVC(probability=True, random_state=42)
            elif model_name == "Gradient Boosting":
                base_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_name == "Decision Tree":
                base_model = DecisionTreeClassifier(random_state=42)
            else:
                base_model = RandomForestClassifier(random_state=42)
            
            # Verify data dimensions before starting
            if len(self.X_labeled) != len(self.y_labeled):
                raise ValueError(f"X_labeled ({len(self.X_labeled)}) and y_labeled ({len(self.y_labeled)}) have different lengths")
            
            if self.y_true_unlabeled is not None and len(self.X_unlabeled) != len(self.y_true_unlabeled):
                raise ValueError(f"X_unlabeled ({len(self.X_unlabeled)}) and y_true_unlabeled ({len(self.y_true_unlabeled)}) have different lengths")
            
            # Initialize self-training classifier
            self.self_training_model = SelfTrainingClassifier(
                base_classifier=base_model,
                threshold=threshold,
                max_iter=max_iter
            )
            
            # Show progress
            self.training_progress.setVisible(True)
            self.training_progress.setRange(0, max_iter)
            self.training_progress.setValue(0)
            
            self.progress_text.append("🚀 Starting Self-Training Algorithm")
            self.progress_text.append(f"• Base model: {model_name}")
            self.progress_text.append(f"• Confidence threshold: {threshold:.2f}")
            self.progress_text.append(f"• Max iterations: {max_iter}")
            self.progress_text.append(f"• Initial labeled samples: {len(self.X_labeled):,}")
            self.progress_text.append(f"• Initial unlabeled samples: {len(self.X_unlabeled):,}")
            self.progress_text.append("-" * 50)
            
            # Train model
            QApplication.processEvents()  # Update UI
            
            # Pass y_true_unlabeled only if it exists and has correct length
            y_true_for_training = self.y_true_unlabeled if self.y_true_unlabeled is not None and len(self.y_true_unlabeled) == len(self.X_unlabeled) else None
            
            self.self_training_model.fit(
                self.X_labeled, 
                self.y_labeled, 
                self.X_unlabeled,
                y_true_for_training
            )
            
            # Update progress bar
            for i in range(max_iter):
                self.training_progress.setValue(i + 1)
                QApplication.processEvents()
            
            # Display training history
            self.progress_text.append("\n📊 Training History:")
            self.progress_text.append("Iter | Labeled | Unlabeled | New | Avg Conf | Accuracy")
            self.progress_text.append("-" * 60)
            
            for hist in self.self_training_model.history:
                acc_str = f"{hist['accuracy_on_unlabeled']:.3%}" if hist['accuracy_on_unlabeled'] is not None else "N/A"
                self.progress_text.append(
                    f"{hist['iteration']:4d} | "
                    f"{hist['labeled_samples']:7d} | "
                    f"{hist['unlabeled_samples']:9d} | "
                    f"{hist['new_labels_added']:3d} | "
                    f"{hist['avg_confidence']:.3f} | "
                    f"{acc_str}"
                )
            
            # Final evaluation
            if self.y_true_unlabeled is not None and len(self.y_true_unlabeled) > 0:
                # Verify dimensions
                if len(self.X_unlabeled) != len(self.y_true_unlabeled):
                    self.progress_text.append(f"\n⚠️ Warning: Dimension mismatch - X_unlabeled ({len(self.X_unlabeled)}), y_true_unlabeled ({len(self.y_true_unlabeled)})")
                    # Try to align them
                    min_len = min(len(self.X_unlabeled), len(self.y_true_unlabeled))
                    if min_len > 0:
                        predictions = self.self_training_model.predict(self.X_unlabeled[:min_len])
                        accuracy = accuracy_score(self.y_true_unlabeled[:min_len], predictions)
                        self.predicted_labels = predictions
                    else:
                        accuracy = 0
                        self.predicted_labels = None
                else:
                    predictions = self.self_training_model.predict(self.X_unlabeled)
                    self.predicted_labels = predictions
                    
                    accuracy = accuracy_score(self.y_true_unlabeled, predictions)
                    precision = precision_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                    recall = recall_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(self.y_true_unlabeled, predictions, average='weighted', zero_division=0)
                
                self.progress_text.append("\n" + "=" * 60)
                self.progress_text.append("✅ Self-Training Complete!")
                self.progress_text.append(f"Final accuracy on unlabeled data: {accuracy:.3%}")
                
                # Update results tab
                self.update_results_tab(accuracy, precision, recall, f1)
                self.update_visualizations()
                self.update_comparison_tab()
                
                # Enable navigation to results
                self.next_to_results_button.setEnabled(True)
                self.export_results_button.setEnabled(True)
                self.export_report_button.setEnabled(True)
                self.next_to_comparison_button.setEnabled(True)
            else:
                self.progress_text.append("\n" + "=" * 60)
                self.progress_text.append("✅ Self-Training Complete!")
                self.progress_text.append("Note: No true labels available for evaluation")
            
            self.training_progress.setVisible(False)
            self.status_bar.showMessage("Self-training complete!")
            
        except Exception as e:
            self.training_progress.setVisible(False)
            QMessageBox.critical(self, "Training Error", f"Error during self-training: {str(e)}")
    
    def update_results_tab(self, accuracy, precision, recall, f1):
        """Update results tab with metrics"""
        results_html = f"""
        <h2>Self-Training Results</h2>
        
        <h3>📈 Performance Metrics:</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #4a6fa5; color: white;">
                <th style="padding: 10px;">Metric</th>
                <th style="padding: 10px;">Value</th>
                <th style="padding: 10px;">Interpretation</th>
            </tr>
            <tr>
                <td style="padding: 10px;"><b>Accuracy</b></td>
                <td style="padding: 10px; font-weight: bold; color: #27ae60;">{accuracy:.3%}</td>
                <td style="padding: 10px;">Overall correct predictions</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><b>Precision</b></td>
                <td style="padding: 10px; font-weight: bold; color: #3498db;">{precision:.3%}</td>
                <td style="padding: 10px;">Correct positive predictions among all positive predictions</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><b>Recall</b></td>
                <td style="padding: 10px; font-weight: bold; color: #e67e22;">{recall:.3%}</td>
                <td style="padding: 10px;">Correct positive predictions among all actual positives</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><b>F1-Score</b></td>
                <td style="padding: 10px; font-weight: bold; color: #9b59b6;">{f1:.3%}</td>
                <td style="padding: 10px;">Harmonic mean of precision and recall</td>
            </tr>
        </table>
        
        <h3>📊 Training Summary:</h3>
        <ul>
            <li><b>Base Classifier:</b> {self.classifier_combo.currentText()}</li>
            <li><b>Confidence Threshold:</b> {self.threshold_slider.value()/100:.2f}</li>
            <li><b>Total Iterations:</b> {len(self.self_training_model.history)}</li>
            <li><b>Final Labeled Samples:</b> {self.self_training_model.history[-1]['labeled_samples']:,}</li>
            <li><b>Final Unlabeled Samples:</b> {self.self_training_model.history[-1]['unlabeled_samples']:,}</li>
            <li><b>Total Labels Recovered:</b> {self.self_training_model.history[-1]['labeled_samples'] - len(self.X_labeled):,}</li>
        </ul>
        
        <h3>🎯 Interpretation:</h3>
        """
        
        if accuracy > 0.9:
            results_html += "<p style='color: #27ae60; font-weight: bold;'>✅ Excellent recovery! The self-training algorithm successfully recovered most labels.</p>"
        elif accuracy > 0.7:
            results_html += "<p style='color: #f39c12; font-weight: bold;'>⚠️ Good recovery. The algorithm recovered a reasonable portion of labels.</p>"
        else:
            results_html += "<p style='color: #e74c3c; font-weight: bold;'>❌ Moderate recovery. Consider adjusting parameters or using a different base classifier.</p>"
        
        self.results_text.setHtml(results_html)
    
    def update_visualizations(self):
        """Update all visualization plots"""
        # Clear all figures
        self.training_figure.clear()
        self.cm_figure.clear()
        self.progress_figure.clear()
        self.conf_figure.clear()
        
        # 1. Training Progress Plot
        ax1 = self.training_figure.add_subplot(111)
        if self.self_training_model.history:
            iterations = [h['iteration'] for h in self.self_training_model.history]
            labeled_samples = [h['labeled_samples'] for h in self.self_training_model.history]
            new_labels = [h['new_labels_added'] for h in self.self_training_model.history]
            
            ax1.plot(iterations, labeled_samples, 'b-', linewidth=2, marker='o', label='Labeled Samples')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Labeled Samples', fontsize=12, color='b')
            ax1.set_title('Self-Training Progress', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.bar(iterations, new_labels, alpha=0.5, color='orange', label='New Labels Added')
            ax2.set_ylabel('New Labels Added', fontsize=12, color='orange')
            ax2.legend(loc='upper right')
        
        self.training_canvas.draw()
        
        # 2. Confusion Matrix
        ax3 = self.cm_figure.add_subplot(111)
        if self.y_true_unlabeled is not None and self.predicted_labels is not None:
            try:
                # Ensure lengths match
                min_len = min(len(self.y_true_unlabeled), len(self.predicted_labels))
                if min_len > 0:
                    cm = confusion_matrix(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                               xticklabels=self.class_names[:cm.shape[0]],
                               yticklabels=self.class_names[:cm.shape[0]])
                    ax3.set_xlabel('Predicted Label', fontsize=12)
                    ax3.set_ylabel('True Label', fontsize=12)
                    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            except Exception as e:
                ax3.text(0.5, 0.5, f"Could not create confusion matrix:\n{str(e)}", 
                        ha='center', va='center', transform=ax3.transAxes)
        
        self.cm_canvas.draw()
        
        # 3. Training History Plot
        ax4 = self.progress_figure.add_subplot(111)
        if self.self_training_model.history:
            iterations = [h['iteration'] for h in self.self_training_model.history]
            accuracies = [h['accuracy_on_unlabeled'] for h in self.self_training_model.history 
                         if h['accuracy_on_unlabeled'] is not None]
            
            if accuracies:
                ax4.plot(range(len(accuracies)), accuracies, 'g-', linewidth=2, marker='s')
                ax4.set_xlabel('Iteration', fontsize=12)
                ax4.set_ylabel('Accuracy', fontsize=12)
                ax4.set_title('Accuracy During Training', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim([0, 1.1])
        
        self.progress_canvas.draw()
        
        # 4. Confidence Distribution
        ax5 = self.conf_figure.add_subplot(111)
        if hasattr(self.self_training_model.final_classifier, 'predict_proba'):
            try:
                proba = self.self_training_model.final_classifier.predict_proba(self.X_unlabeled)
                if proba is not None:
                    max_proba = np.max(proba, axis=1)
                    ax5.hist(max_proba, bins=30, alpha=0.7, color='purple', edgecolor='black')
                    ax5.axvline(x=self.threshold_slider.value()/100, color='red', 
                               linestyle='--', linewidth=2, label=f'Threshold ({self.threshold_slider.value()/100:.2f})')
                    ax5.set_xlabel('Confidence Score', fontsize=12)
                    ax5.set_ylabel('Frequency', fontsize=12)
                    ax5.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
            except Exception as e:
                ax5.text(0.5, 0.5, f"Could not create confidence plot:\n{str(e)}", 
                        ha='center', va='center', transform=ax5.transAxes)
        
        self.conf_canvas.draw()
    
    def update_comparison_tab(self):
        """Update comparison tab with true vs predicted labels"""
        if self.y_true_unlabeled is None or self.predicted_labels is None:
            return
        
        # Calculate metrics with aligned lengths
        min_len = min(len(self.y_true_unlabeled), len(self.predicted_labels))
        if min_len == 0:
            return
        
        accuracy = accuracy_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len])
        precision = precision_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
        recall = recall_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
        f1 = f1_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
        
        # Update metric labels
        self.accuracy_label.setText(f"Accuracy: {accuracy:.3%}")
        self.precision_label.setText(f"Precision: {precision:.3%}")
        self.recall_label.setText(f"Recall: {recall:.3%}")
        self.f1_label.setText(f"F1-Score: {f1:.3%}")
        
        # Update comparison table
        n_samples = min(100, min_len)
        self.comparison_table.clear()
        self.comparison_table.setRowCount(n_samples)
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels(['Sample', 'True Label', 'Predicted Label', 'Match'])
        
        for i in range(n_samples):
            true_label = self.y_true_unlabeled[i]
            pred_label = self.predicted_labels[i]
            match = true_label == pred_label
            
            # Decode labels if encoder was used
            if hasattr(self.label_encoder, 'classes_'):
                try:
                    true_label_str = self.label_encoder.inverse_transform([true_label])[0]
                    pred_label_str = self.label_encoder.inverse_transform([pred_label])[0]
                except:
                    true_label_str = str(true_label)
                    pred_label_str = str(pred_label)
            else:
                true_label_str = str(true_label)
                pred_label_str = str(pred_label)
            
            self.comparison_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(true_label_str))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(pred_label_str))
            
            match_item = QTableWidgetItem("✅" if match else "❌")
            match_item.setTextAlignment(Qt.AlignCenter)
            if match:
                match_item.setBackground(QColor(220, 237, 200))
            else:
                match_item.setBackground(QColor(255, 220, 220))
            self.comparison_table.setItem(i, 3, match_item)
        
        self.comparison_table.resizeColumnsToContents()
        
        # Update visualization
        self.comparison_figure.clear()
        ax = self.comparison_figure.add_subplot(111)
        
        # Create bar plot of matches vs mismatches
        match_count = np.sum(self.y_true_unlabeled[:n_samples] == self.predicted_labels[:n_samples])
        mismatch_count = n_samples - match_count
        
        bars = ax.bar(['Correct', 'Incorrect'], [match_count, mismatch_count], 
                     color=['#27ae60', '#e74c3c'], edgecolor='black')
        
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(f'Label Recovery: {match_count}/{n_samples} Correct ({accuracy:.1%})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom')
        
        self.comparison_canvas.draw()
    
    # ============================================
    # EXPORT METHODS
    # ============================================
    
    def export_results(self):
        """Export results to CSV"""
        if self.predicted_labels is None:
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "self_training_results.csv", 
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Align lengths
                min_len = min(len(self.y_true_unlabeled), len(self.predicted_labels))
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'true_label': self.y_true_unlabeled[:min_len],
                    'predicted_label': self.predicted_labels[:min_len],
                    'is_correct': self.y_true_unlabeled[:min_len] == self.predicted_labels[:min_len]
                })
                
                # Add decoded labels if encoder was used
                if hasattr(self.label_encoder, 'classes_'):
                    try:
                        results_df['true_label_decoded'] = self.label_encoder.inverse_transform(self.y_true_unlabeled[:min_len])
                        results_df['predicted_label_decoded'] = self.label_encoder.inverse_transform(self.predicted_labels[:min_len])
                    except:
                        pass
                
                # Save to CSV
                results_df.to_csv(file_path, index=False)
                
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Results exported to:\n{file_path}\n\n"
                    f"Exported {len(results_df):,} samples."
                )
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting results: {str(e)}")
    
    def export_full_report(self):
        """Export full report with all details"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Full Report", "self_training_report.txt", 
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("SELF-TRAINING SEMI-SUPERVISED LEARNING REPORT\n")
                    f.write("=" * 70 + "\n\n")
                    
                    # Dataset information
                    f.write("DATASET INFORMATION\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Dataset: {os.path.basename(self.file_path_edit.text())}\n")
                    f.write(f"Total samples: {len(self.original_data):,}\n")
                    f.write(f"Features: {len(self.feature_names)}\n")
                    f.write(f"Target column: {self.target_column}\n")
                    f.write(f"Classes: {len(self.class_names)}\n\n")
                    
                    # Preprocessing information
                    f.write("PREPROCESSING SETTINGS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Label removal: {self.unlabeled_slider.value()}%\n")
                    f.write(f"Scale features: {self.scale_checkbox.isChecked()}\n")
                    f.write(f"Encode categorical: {self.encode_checkbox.isChecked()}\n")
                    f.write(f"Impute missing: {self.impute_checkbox.isChecked()}\n")
                    f.write(f"Labeled samples: {len(self.X_labeled):,}\n")
                    f.write(f"Unlabeled samples: {len(self.X_unlabeled):,}\n\n")
                    
                    # Training parameters
                    f.write("TRAINING PARAMETERS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Base classifier: {self.classifier_combo.currentText()}\n")
                    f.write(f"Confidence threshold: {self.threshold_slider.value()/100:.2f}\n")
                    f.write(f"Max iterations: {self.iter_spinbox.value()}\n")
                    f.write(f"Actual iterations: {len(self.self_training_model.history)}\n\n")
                    
                    # Results
                    f.write("RESULTS\n")
                    f.write("-" * 30 + "\n")
                    if self.y_true_unlabeled is not None and self.predicted_labels is not None:
                        min_len = min(len(self.y_true_unlabeled), len(self.predicted_labels))
                        if min_len > 0:
                            accuracy = accuracy_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len])
                            precision = precision_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
                            recall = recall_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
                            f1 = f1_score(self.y_true_unlabeled[:min_len], self.predicted_labels[:min_len], average='weighted', zero_division=0)
                            
                            f.write(f"Accuracy: {accuracy:.3%}\n")
                            f.write(f"Precision: {precision:.3%}\n")
                            f.write(f"Recall: {recall:.3%}\n")
                            f.write(f"F1-Score: {f1:.3%}\n\n")
                    
                    # Training history
                    f.write("TRAINING HISTORY\n")
                    f.write("-" * 30 + "\n")
                    f.write("Iter | Labeled | Unlabeled | New | Avg Conf | Accuracy\n")
                    for hist in self.self_training_model.history:
                        acc_str = f"{hist['accuracy_on_unlabeled']:.3%}" if hist['accuracy_on_unlabeled'] is not None else "N/A"
                        f.write(f"{hist['iteration']:4d} | "
                               f"{hist['labeled_samples']:7d} | "
                               f"{hist['unlabeled_samples']:9d} | "
                               f"{hist['new_labels_added']:3d} | "
                               f"{hist['avg_confidence']:8.3f} | "
                               f"{acc_str}\n")
                
                QMessageBox.information(self, "Report Exported", f"Full report exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting report: {str(e)}")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Self-Training Demo")
    app.setStyle('Fusion')
    
    window = SelfTrainingApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()