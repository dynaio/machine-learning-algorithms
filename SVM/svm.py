# universal_svm.py — THE FINAL UNIVERSAL SVM — WORKS WITH ANY DATASET
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class SVMWorker(QThread):
    finished = pyqtSignal(object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str)
    progress = pyqtSignal(int)

    def __init__(self, X_train, X_test, y_train, y_test, X_raw_train, X_raw_test, kernel, C, gamma):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_raw_train = X_raw_train
        self.X_raw_test = X_raw_test
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def run(self):
        try:
            self.progress.emit(20)
            model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma,
                        probability=True, random_state=42)
            self.progress.emit(50)
            model.fit(self.X_train, self.y_train)
            self.progress.emit(80)

            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)

            # Smart 2D projection
            pca = PCA(n_components=2, random_state=42)
            X_all = np.vstack([self.X_raw_train, self.X_raw_test])
            X_2d = pca.fit_transform(X_all)
            X_test_2d = X_2d[len(self.X_raw_train):]

            # Sample for speed
            n_show = min(2000, len(X_test_2d))
            idx = np.random.choice(len(X_test_2d), n_show, replace=False)
            X_plot = X_test_2d[idx]
            y_plot = self.y_test[idx]

            # Grid in 2D
            h = 0.05
            pad = 2
            x_min, x_max = X_plot[:, 0].min() - pad, X_plot[:, 0].max() + pad
            y_min, y_max = X_plot[:, 1].min() - pad, X_plot[:, 1].max() + pad
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_high = pca.inverse_transform(grid_2d)

            Z = model.predict(grid_high).reshape(xx.shape)
            self.progress.emit(100)

            msg = f"SVM ({self.kernel}) — Acc: {acc:.2%} — {len(model.support_vectors_)} SVs"
            self.finished.emit(model, xx, yy, Z, X_plot, y_plot, acc, msg)

        except Exception as e:
            self.finished.emit(None, None, None, None, None, None, 0, f"Error: {e}")


class UniversalSVM(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal SVM — Works with ANY Dataset")
        self.setGeometry(50, 30, 1500, 900)
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_raw_train = self.X_raw_test = None
        self.model = None
        self.worker = None
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left Panel
        left = QVBoxLayout()
        title = QLabel("<h1>Universal SVM</h1><p>Any file • Any features • All kernels</p>")
        title.setAlignment(Qt.AlignCenter)
        left.addWidget(title)

        btn = QPushButton("Load Dataset (.csv, )")
        btn.clicked.connect(self.load_any_file)
        btn.setStyleSheet("font-size:16pt; padding:15px;")
        left.addWidget(btn)

        self.info = QLabel("No file loaded")
        left.addWidget(self.info)

        # Parameters
        gb = QGroupBox("SVM Parameters")
        form = QFormLayout()

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
        form.addRow("Kernel:", self.kernel_combo)

        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.1, 1000)
        self.c_spin.setValue(1.0)
        form.addRow("C:", self.c_spin)

        self.gamma_combo = QComboBox()
        self.gamma_combo.addItems(['scale', 'auto', '0.01', '0.1', '1.0'])
        form.addRow("Gamma:", self.gamma_combo)

        gb.setLayout(form)
        left.addWidget(gb)

        self.scale_chk = QCheckBox("Standard Scale features")
        self.scale_chk.setChecked(True)
        left.addWidget(self.scale_chk)

        self.run_btn = QPushButton("RUN SVM")
        self.run_btn.clicked.connect(self.run_svm)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("background:#e74c3c; color:white; font-size:18pt; padding:15px;")
        left.addWidget(self.run_btn)

        layout.addLayout(left, 1)

        # Right Panel
        right = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure(figsize=(12, 8)))
        self.ax = self.canvas.figure.subplots()
        right.addWidget(self.canvas)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        right.addWidget(QLabel("Results — Click plot to predict"))
        right.addWidget(self.result)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right.addWidget(self.progress)

        layout.addLayout(right, 3)

    def load_any_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "All Files (*.csv *.xls *.xlsx)")
        if not path: return

        try:
            if path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(path, engine='openpyxl' if path.endswith('.xlsx') else 'xlrd')
            else:
                df = pd.read_csv(path)

            # Auto-detect target column
            target_cols = [c for c in df.columns if 'target' in c.lower() or 'class' in c.lower() or 'loan' in c.lower()]
            if not target_cols:
                target_cols = [df.columns[-1]]  # fallback

            target = target_cols[0]
            X = df.drop(columns=[target])
            y = df[target]

            # Keep only numeric
            X = X.select_dtypes(include='number')
            if X.empty:
                self.result.setText("No numeric features found!")
                return

            X_raw = X.values
            y = y.values

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )

            self.X_raw_train, self.X_raw_test = X_train_raw, X_test_raw
            self.y_train, self.y_test = y_train, y_test

            self.info.setText(f"{path.split('/')[-1]}\n{len(df)} samples • Target: {target}")
            self.run_btn.setEnabled(True)
            self.result.setText("Data loaded! Click RUN SVM")

        except Exception as e:
            self.result.setText(f"Error loading file:\n{e}")

    def run_svm(self):
        if self.X_raw_train is None: return

        X_train = self.X_raw_train.copy()
        X_test = self.X_raw_test.copy()

        if self.scale_chk.isChecked():
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = None

        gamma = self.gamma_combo.currentText()
        if gamma not in ['scale', 'auto']:
            gamma = float(gamma)

        params = {
            'C': self.c_spin.value(),
            'kernel': self.kernel_combo.currentText(),
            'gamma': gamma
        }

        if self.thread and self.thread.isRunning():
            self.thread.terminate()

        self.thread = QThread()
        self.worker = SVMWorker(X_train, X_test, self.y_train, self.y_test,
                                self.X_raw_train, self.X_raw_test,
                                params['kernel'], params['C'], params['gamma'])
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_done)
        self.worker.progress.connect(self.progress.setValue)

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.thread.start()

    def on_done(self, model, xx, yy, Z, X_plot, y_plot, acc, msg):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        if model is None: return

        self.model = model
        self.ax.clear()

        colors_light = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF']
        colors_bold = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
        cmap_light = ListedColormap(colors_light[:len(np.unique(y_plot))])
        cmap_bold = ListedColormap(colors_bold[:len(np.unique(y_plot))])

        self.ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
        self.ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_bold,
                        edgecolor='black', s=80, linewidth=1)

        self.ax.set_title(f"Universal SVM — {msg}", fontsize=18)
        self.canvas.draw()

        self.result.setText(f"<h3>{msg}</h3>\nClick anywhere to predict!")

        def click(event):
            if event.inaxes != self.ax or not model: return
            point_2d = np.array([[event.xdata, event.ydata]])
            pca = PCA(n_components=2).fit(np.vstack([self.X_raw_train, self.X_raw_test]))
            point_high = pca.inverse_transform(point_2d)
            if self.scale_chk.isChecked():
                scaler = StandardScaler().fit(self.X_raw_train)
                point_high = scaler.transform(point_high)
            pred = model.predict(point_high)[0]
            prob = model.predict_proba(point_high)[0]
            self.result.append(f"Prediction: <b>{pred}</b> | Confidence: <b>{max(prob):.1%}</b>")

        self.canvas.mpl_connect('button_press_event', click)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = UniversalSVM()
    win.show()
    sys.exit(app.exec_())