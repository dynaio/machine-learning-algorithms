import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class NaiveBayesLightGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naive Bayes Pro — Light Theme")
        self.setGeometry(70, 40, 1480, 940)
        self.setStyleSheet("""
            QMainWindow { background: #f8f9fa; }
            QLabel { color: #2c3e50; font-size: 11pt; }
            QGroupBox { font-weight: bold; color: #2980b9; border: 2px solid #3498db; border-radius: 8px; margin: 5px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #3498db; color: white; font-weight: bold; padding: 10px; border-radius: 6px; }
            QPushButton:hover { background: #2980b9; }
            QPushButton:disabled { background: #95a5a6; }
            QTextEdit { background: white; border: 1px solid #bdc3c7; border-radius: 6px; padding: 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox { padding: 6px; border: 1px solid #95a5a6; border-radius: 6px; }
        """)
        self.X = self.y = None
        self.model = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left Panel
        left = QVBoxLayout()
        self.build_controls(left)
        layout.addLayout(left, 1)

        # Right: Visualization
        right = QVBoxLayout()

        self.canvas = FigureCanvas(plt.figure(figsize=(12, 8)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_facecolor('white')
        self.canvas.figure.patch.set_facecolor('white')
        plt.style.use('default')  # Force light matplotlib style
        right.addWidget(self.canvas)

        # Stats Box
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setStyleSheet("background: #f8f9fa; border: 1px solid #3498db; color: #2c3e50; font-size: 11pt;")
        self.stats_box.setMaximumHeight(220)
        right.addWidget(QLabel("<b style='color:#2980b9;'>Model Performance & Prediction Log</b>"))
        right.addWidget(self.stats_box)

        layout.addLayout(right, 3)

    def build_controls(self, layout):
        title = QLabel("<h1 style='color:#2980b9;'>Naive Bayes Pro</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Load Data
        gb1 = QGroupBox("1. Load Dataset")
        v = QVBoxLayout()
        self.btn_load = QPushButton("Load CSV (must have 'Target' column)")
        self.btn_load.setStyleSheet("background:#27ae60; color:white; font-size:12pt;")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setStyleSheet("color:#7f8c8d;")
        v.addWidget(self.btn_load)
        v.addWidget(self.lbl_file)
        gb1.setLayout(v)
        layout.addWidget(gb1)

        # Variant
        gb2 = QGroupBox("2. Select Variant")
        v = QVBoxLayout()
        self.combo_variant = QComboBox()
        self.combo_variant.addItems(['GaussianNB', 'BernoulliNB', 'MultinomialNB'])
        self.combo_variant.currentTextChanged.connect(self.update_hint)
        v.addWidget(self.combo_variant)
        gb2.setLayout(v)
        layout.addWidget(gb2)

        # Preprocessing
        gb3 = QGroupBox("3. Preprocessing Options")
        form = QFormLayout()
        self.chk_scale = QCheckBox("Standard Scale (recommended for Gaussian)")
        self.chk_scale.setChecked(True)
        form.addRow(self.chk_scale)

        self.spin_binarize = QDoubleSpinBox()
        self.spin_binarize.setRange(0.0, 1.0)
        self.spin_binarize.setValue(0.0)
        self.spin_binarize.setSingleStep(0.1)
        form.addRow("Binarize threshold (Bernoulli):", self.spin_binarize)

        self.chk_abs = QCheckBox("Force non-negative (for Multinomial)")
        self.chk_abs.setChecked(True)
        form.addRow(self.chk_abs)
        gb3.setLayout(form)
        layout.addWidget(gb3)

        # RUN BUTTON
        gb4 = QGroupBox("4. Train Model")
        v = QVBoxLayout()
        self.btn_run = QPushButton("RUN NAIVE BAYES")
        self.btn_run.setStyleSheet("""
            background: #e74c3c; color: white; font-weight: bold;
            font-size: 20pt; padding: 18px; border-radius: 10px;
        """)
        self.btn_run.clicked.connect(self.run_model)
        self.btn_run.setEnabled(False)
        v.addWidget(self.btn_run)
        gb4.setLayout(v)
        layout.addWidget(gb4)

        # Status Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(QLabel("Status Log:"))
        layout.addWidget(self.log)
        layout.addStretch()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            if 'Target' not in df.columns:
                QMessageBox.critical(self, "Error", "CSV must contain a column named 'Target'")
                return

            self.y = df['Target'].values
            X_cols = df.drop('Target', axis=1).select_dtypes(include='number').columns
            self.X_raw = df[X_cols].values.astype(float)

            if self.X_raw.shape[1] > 2:
                self.X = PCA(n_components=2).fit_transform(self.X_raw)
                self.log.append("High-dim data → PCA (2D) for visualization\n")
            else:
                self.X = self.X_raw.copy()

            self.lbl_file.setText(f"{path.split('/')[-1]}\n{self.X.shape[0]} samples × {self.X.shape[1]} features")
            self.btn_run.setEnabled(True)
            self.log.append("Data loaded! Click the big red RUN button.\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_hint(self):
        hints = {
            'GaussianNB': "Best for continuous features (price, mileage, etc.)",
            'BernoulliNB': "Best for binary 0/1 features",
            'MultinomialNB': "Best for count data (e.g. word frequencies)"
        }
        self.log.append(f"\n→ {self.combo_variant.currentText()}: {hints[self.combo_variant.currentText()]}\n")

    def run_model(self):
        if self.X is None:
            return

        self.log.append("\nTraining... Please wait\n")
        X = self.X.copy()
        variant = self.combo_variant.currentText()

        # Preprocessing
        if variant == 'GaussianNB' and self.chk_scale.isChecked():
            X = StandardScaler().fit_transform(X)
        elif variant == 'MultinomialNB':
            if self.chk_abs.isChecked():
                X = np.abs(X)
            X = MinMaxScaler().fit_transform(X)
        elif variant == 'BernoulliNB':
            X = Binarizer(threshold=self.spin_binarize.value()).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.3, random_state=42, stratify=self.y)

        # Train model
        if variant == 'GaussianNB':
            self.model = GaussianNB()
        elif variant == 'BernoulliNB':
            self.model = BernoulliNB()
        else:
            self.model = MultinomialNB()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Plot
        self.ax.clear()
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Light, clean colors
        cmap_light = ListedColormap(['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#ffccff'])
        cmap_bold = ListedColormap(['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6'])

        self.ax.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=0.8, shading='auto')
        scatter = self.ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
                                  edgecolor='black', linewidth=1.2, s=90)

        # Confidence contours
        if hasattr(self.model, "predict_proba"):
            Z_prob = self.model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            confidence = np.max(Z_prob, axis=1).reshape(xx.shape)
            cs = self.ax.contour(xx, yy, confidence, levels=[0.6, 0.8, 0.95], colors='#2c3e50', alpha=0.7, linewidths=1.5)
            self.ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")

        self.ax.set_title(f"{variant} — Test Accuracy: {acc:.2%}", fontsize=18, color='#2980b9', pad=20)
        self.ax.grid(True, alpha=0.3, color='#95a5a6')
        self.canvas.draw()

        # Stats
        cm = confusion_matrix(y_test, y_pred)
        self.stats_box.clear()
        self.stats_box.append(f"<h3 style='color:#2980b9;'>Model: {variant}</h3>")
        self.stats_box.append(f"<b style='color:#27ae60;'>Accuracy: {acc:.2%}</b>")
        self.stats_box.append(f"<b>Classes:</b> {len(np.unique(self.y))}")
        self.stats_box.append(f"<pre>Confusion Matrix:\n{np.array2string(cm)}</pre>")
        self.stats_box.append("<span style='color:#e74c3c;'>Click anywhere on the plot to predict!</span>")

        self.log.append(f"Done! Accuracy: {acc:.2%}\n")

        # Click to predict
        def onclick(event):
            if event.inaxes != self.ax: return
            point = np.array([[event.xdata, event.ydata]])
            pred = self.model.predict(point)[0]
            prob = self.model.predict_proba(point)[0].max()
            self.stats_box.append(f"<span style='color:#e74c3c;'>→ Point → Class: <b>{pred}</b> | Confidence: <b>{prob:.1%}</b></span>")
        self.canvas.mpl_connect('button_press_event', onclick)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = NaiveBayesLightGUI()
    win.show()
    sys.exit(app.exec_())