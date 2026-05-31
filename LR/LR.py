import sys
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QLineEdit, 
                             QPushButton, QGroupBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'archive', 'CarPrice_Assignment.csv')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

METRICS = {
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'RMSE': rmse
}

PENALTIES = {
    'None': None,
    'Ridge (L2)': 'l2', 
    'Lasso (L1)': 'l1',
    'ElasticNet': 'elasticnet'
}

class TrainingWorker(QThread):
    epoch_completed = pyqtSignal(int, float, np.ndarray, np.ndarray, float, float)
    training_finished = pyqtSignal()
    
    def __init__(self, X, y, penalty, alpha, learning_rate, epochs, metric):
        super().__init__()
        self.X = X
        self.y = y
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.metric = metric
        self._is_running = True
        self.model = None
        self.scaler = None
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.2, random_state=42)
            
            # Initialize model
            self.model = SGDRegressor(
                penalty=self.penalty,
                alpha=self.alpha,
                learning_rate='constant',
                eta0=self.learning_rate,
                max_iter=1,
                warm_start=True,
                random_state=42
            )
            
            # Prepare plot data
            x_min, x_max = self.X.min(), self.X.max()
            x_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            x_plot_scaled = self.scaler.transform(x_plot)
            
            # Train epoch by epoch
            for epoch in range(self.epochs):
                if not self._is_running:
                    break
                
                # Train for one epoch
                if epoch == 0:
                    self.model.partial_fit(X_train, y_train)
                else:
                    self.model.partial_fit(X_train, y_train)
                
                # Calculate predictions and error
                y_plot = self.model.predict(x_plot_scaled)
                y_pred = self.model.predict(X_test)
                error = METRICS[self.metric](y_test, y_pred)
                
                # Get coefficients (a) and intercept (b) for y = ax + b
                a = self.model.coef_[0]  # Slope
                b = self.model.intercept_[0]  # Intercept
                
                # Emit progress with coefficients
                self.epoch_completed.emit(epoch + 1, error, x_plot.flatten(), y_plot, a, b)
                
                # Slow down training for better visualization
                self.msleep(100)
                
            self.training_finished.emit()
            
        except Exception as e:
            print(f"Training error: {e}")

class RegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.feature_cols = []
        self.training_worker = None
        self.original_data = None
        self.regression_line = None
        self.regularization_points = None
        self.current_epoch = 0
        self.current_a = 0.0
        self.current_b = 0.0
        self.init_ui()
        self.load_data()
        
    def init_ui(self):
        self.setWindowTitle('Advanced Price Regression Simulator - y = ax + b')
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        control_panel = QWidget()
        control_panel.setFixedWidth(400)
        control_layout = QVBoxLayout(control_panel)
        
        # Feature selection
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout(feature_group)
        
        feature_layout.addWidget(QLabel("Select Feature (X):"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        feature_layout.addWidget(self.feature_combo)
        
        control_layout.addWidget(feature_group)
        
        # Regularization selection
        reg_group = QGroupBox("Regularization Settings")
        reg_layout = QVBoxLayout(reg_group)
        
        reg_layout.addWidget(QLabel("Regularization Type:"))
        self.penalty_combo = QComboBox()
        self.penalty_combo.addItems(list(PENALTIES.keys()))
        self.penalty_combo.currentTextChanged.connect(self.on_penalty_change)
        reg_layout.addWidget(self.penalty_combo)
        
        reg_layout.addWidget(QLabel("Alpha (Regularization Strength):"))
        self.alpha_edit = QLineEdit("0.01")
        reg_layout.addWidget(self.alpha_edit)
        
        control_layout.addWidget(reg_group)
        
        # Training parameters
        train_group = QGroupBox("Training Parameters")
        train_layout = QVBoxLayout(train_group)
        
        train_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_edit = QLineEdit("0.01")
        train_layout.addWidget(self.lr_edit)
        
        train_layout.addWidget(QLabel("Number of Epochs:"))
        self.epochs_edit = QLineEdit("50")
        train_layout.addWidget(self.epochs_edit)
        
        train_layout.addWidget(QLabel("Error Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(METRICS.keys()))
        train_layout.addWidget(self.metric_combo)
        
        control_layout.addWidget(train_group)
        
        # Regression Coefficients Display
        coeff_group = QGroupBox("Regression Coefficients - y = ax + b")
        coeff_layout = QVBoxLayout(coeff_group)
        
        self.coeff_label = QLabel("a (slope): --\nb (intercept): --\nEquation: --")
        self.coeff_label.setStyleSheet("QLabel { background-color: #e8f4fd; padding: 10px; border: 1px solid #b8daff; border-radius: 5px; font-weight: bold; }")
        self.coeff_label.setWordWrap(True)
        coeff_layout.addWidget(self.coeff_label)
        
        control_layout.addWidget(coeff_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # Control buttons
        btn_group = QGroupBox("Controls")
        btn_layout = QVBoxLayout(btn_group)
        
        self.start_btn = QPushButton("START TRAINING")
        self.start_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.clicked.connect(self.pause_training)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.restart_btn = QPushButton("RESTART")
        self.restart_btn.clicked.connect(self.restart_training)
        btn_layout.addWidget(self.restart_btn)
        
        control_layout.addWidget(btn_group)
        
        # Info display
        info_group = QGroupBox("Training Information")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("Select feature and press START")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        info_layout.addWidget(self.info_label)
        control_layout.addWidget(info_group)
        
        control_layout.addStretch()
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Right panel - plot
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        
        main_layout.addWidget(self.plot_widget)
        
        # Initialize
        self.on_penalty_change()
        self.update_coefficients_display()
        
    def load_data(self):
        try:
            if not os.path.exists(DATA_PATH):
                QMessageBox.critical(self, "Error", f"File not found: {DATA_PATH}")
                return
                
            self.df = pd.read_csv(DATA_PATH)
            # Remove non-numeric and irrelevant columns
            self.df = self.df.select_dtypes(include=[np.number])
            self.df = self.df.drop(['car_ID'], axis=1, errors='ignore')
            self.df = self.df.dropna()
            
            # Get feature columns (all except price)
            self.feature_cols = [col for col in self.df.columns if col != 'price']
            
            # Update feature combo
            self.feature_combo.clear()
            self.feature_combo.addItems(self.feature_cols)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def on_feature_changed(self):
        self.initialize_plot()
        self.update_coefficients_display()
            
    def on_penalty_change(self):
        penalty = self.penalty_combo.currentText()
        if penalty == 'None':
            self.alpha_edit.setEnabled(False)
            self.alpha_edit.setText("0.0")
        else:
            self.alpha_edit.setEnabled(True)
            self.alpha_edit.setText("0.01")
            
    def update_coefficients_display(self, a=None, b=None):
        """Update the coefficients display"""
        if a is None or b is None:
            self.coeff_label.setText(
                "a (slope): --\n"
                "b (intercept): --\n"
                "Equation: y = --"
            )
        else:
            equation = f"y = {a:.4f}x + {b:.4f}"
            if b < 0:
                equation = f"y = {a:.4f}x - {abs(b):.4f}"
            
            self.coeff_label.setText(
                f"a (slope): {a:.4f}\n"
                f"b (intercept): {b:.4f}\n"
                f"Equation: {equation}"
            )
            
    def initialize_plot(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        if not self.feature_combo.currentText():
            self.ax.text(0.5, 0.5, 'Select a feature to start', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
            self.canvas.draw()
            return
            
        feature = self.feature_combo.currentText()
        X = self.df[feature].values
        y = self.df['price'].values
        
        # Store original data
        self.original_data = (X, y)
        
        # Plot original data points in RED
        self.ax.scatter(X, y, color='red', alpha=0.7, s=40, label='Original Data')
        self.ax.set_xlabel(feature, fontsize=12)
        self.ax.set_ylabel('Price', fontsize=12)
        self.ax.set_title(f'Linear Regression: {feature} vs Price\ny = ax + b', fontsize=14, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Initialize regression line (always blue)
        self.regression_line, = self.ax.plot([], [], 'blue', linewidth=3, label='Regression Line')
        
        # Initialize regularization points
        self.regularization_points = None
        
        self.canvas.draw()
        
    def start_training(self):
        if self.training_worker and self.training_worker.isRunning():
            return
            
        if not self.feature_combo.currentText():
            QMessageBox.warning(self, "Warning", "Please select a feature first!")
            return
            
        try:
            feature = self.feature_combo.currentText()
            penalty_type = self.penalty_combo.currentText()
            penalty = PENALTIES[penalty_type]
            alpha = float(self.alpha_edit.text())
            learning_rate = float(self.lr_edit.text())
            epochs = int(self.epochs_edit.text())
            metric = self.metric_combo.currentText()
            
            # Prepare data
            X = self.df[[feature]].values.astype(float)
            y = self.df['price'].values.astype(float)
            
            # Initialize plot
            self.initialize_plot()
            
            # Color mapping for regularization points
            color_map = {
                'None': 'blue',
                'Ridge (L2)': 'yellow',
                'Lasso (L1)': 'green', 
                'ElasticNet': 'pink'
            }
            
            # Add regularization points if not None
            if penalty_type != 'None':
                reg_color = color_map[penalty_type]
                X_reg = self.df[[feature]].values.astype(float)
                y_reg = self.df['price'].values.astype(float)
                self.regularization_points = self.ax.scatter(
                    X_reg, y_reg, color=reg_color, alpha=0.6, s=30, 
                    label=f'{penalty_type} Points'
                )
            
            # Update legend
            self.ax.legend()
            
            # Setup progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(epochs)
            self.progress_bar.setValue(0)
            
            # Reset coefficients
            self.current_a = 0.0
            self.current_b = 0.0
            self.update_coefficients_display(0.0, 0.0)
            
            # Start training worker
            self.training_worker = TrainingWorker(X, y, penalty, alpha, learning_rate, epochs, metric)
            self.training_worker.epoch_completed.connect(self.on_epoch_completed)
            self.training_worker.training_finished.connect(self.on_training_finished)
            
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.current_epoch = 0
            
            self.training_worker.start()
            
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid parameter value: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")
            
    def on_epoch_completed(self, epoch, error, x_plot, y_plot, a, b):
        """Update plot for each completed epoch"""
        self.current_epoch = epoch
        self.current_a = a
        self.current_b = b
        
        # Update regression line (always blue)
        self.regression_line.set_data(x_plot, y_plot)
        
        # Update coefficients display
        self.update_coefficients_display(a, b)
        
        # Update title and info
        penalty_type = self.penalty_combo.currentText()
        alpha_display = float(self.alpha_edit.text()) if penalty_type != 'None' else 0.0
        metric_name = self.metric_combo.currentText()
        total_epochs = int(self.epochs_edit.text())
        
        # Create equation string for title
        equation = f"y = {a:.2f}x + {b:.2f}"
        if b < 0:
            equation = f"y = {a:.2f}x - {abs(b):.2f}"
        
        self.ax.set_title(
            f'{penalty_type} Regression: {self.feature_combo.currentText()} vs Price\n'
            f'{equation}\n'
            f'Epoch: {epoch}/{total_epochs} | {metric_name}: {error:.2f}'
        )
        
        # Adjust plot limits to ensure everything is visible
        if self.original_data:
            X_orig, y_orig = self.original_data
            self.ax.set_xlim(X_orig.min(), X_orig.max())
            self.ax.set_ylim(y_orig.min(), y_orig.max())
        
        # Update info label
        self.info_label.setText(
            f' Epoch: {epoch}/{total_epochs}\n'
            f' {metric_name}: {error:.2f}\n'
            f' Regularization: {penalty_type}\n'
            f'α: {alpha_display} | η: {float(self.lr_edit.text())}'
        )
        
        # Update progress bar
        self.progress_bar.setValue(epoch)
        
        # Redraw canvas
        self.canvas.draw()
            
    def pause_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.pause_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.info_label.setText("Training paused")
            
    def restart_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait(1000)
            
        self.initialize_plot()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.info_label.setText("Ready to start training")
        self.current_epoch = 0
        self.update_coefficients_display()
        
    def on_training_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
        # Show final coefficients
        equation = f"y = {self.current_a:.4f}x + {self.current_b:.4f}"
        if self.current_b < 0:
            equation = f"y = {self.current_a:.4f}x - {abs(self.current_b):.4f}"
            
        self.info_label.setText(f" Training completed!\nFinal Equation: {equation}")
        
    def closeEvent(self, event):
        """Ensure training stops when window is closed"""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait(1000)
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = RegressionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()