import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QFileDialog,
                             QTextEdit, QGroupBox, QFormLayout, QMessageBox, QProgressBar,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class OPTICSWorker(QObject):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, str)  # labels, reachability, core_dist, X_pca, msg

    def __init__(self, X, min_samples, max_eps, metric, cluster_method, xi):
        super().__init__()
        self.X = X
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method
        self.xi = xi

    def run(self):
        try:
            optics = OPTICS(
                min_samples=self.min_samples,
                max_eps=self.max_eps if self.max_eps > 0 else np.inf,
                metric=self.metric,
                cluster_method=self.cluster_method,
                xi=self.xi if self.cluster_method == 'xi' else 0.05
            )
            labels = optics.fit_predict(self.X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X)

            msg = (f"OPTICS Complete!\n"
                   f"Clusters found: {n_clusters}\n"
                   f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)\n"
                   f"minPts={self.min_samples} | method={self.cluster_method}")

            self.finished.emit(labels, optics.reachability_, optics.core_distances_, X_pca, msg)
        except Exception as e:
            self.finished.emit(None, None, None, None, f"Error: {str(e)}")


class OPTICS_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BMW Sales — OPTICS Clustering (Advanced Density)")
        self.setGeometry(80, 50, 1400, 920)
        self.X = None
        self.worker = None
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left Panel
        left = QVBoxLayout()
        self.build_controls(left)
        main.addLayout(left, 1)

        # Right Panel: Two plots
        right = QVBoxLayout()

        # Top: Reachability Plot
        self.canvas_reach = FigureCanvas(plt.figure(figsize=(10, 4)))
        self.ax_reach = self.canvas_reach.figure.subplots()
        right.addWidget(QLabel("<b>Reachability Plot (Cluster Ordering)</b>"))
        right.addWidget(self.canvas_reach)

        # Bottom: 2D Clusters
        self.canvas_2d = FigureCanvas(plt.figure(figsize=(10, 6)))
        self.ax_2d = self.canvas_2d.figure.subplots()
        right.addWidget(QLabel("<b>2D Cluster Visualization (PCA)</b>"))
        right.addWidget(self.canvas_2d)

        self.txt = QTextEdit()
        self.txt.setMaximumHeight(120)
        self.txt.setReadOnly(True)
        right.addWidget(self.txt)

        main.addLayout(right, 3)

    def build_controls(self, layout):
        # Load
        gb1 = QGroupBox("1. Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV File")
        btn.clicked.connect(self.load_csv)
        self.lbl = QLabel("No dataset loaded")
        v.addWidget(btn); v.addWidget(self.lbl)
        gb1.setLayout(v)
        layout.addWidget(gb1)

        # Parameters
        gb2 = QGroupBox("2. OPTICS Parameters")
        f = QFormLayout()

        self.sp_minpts = QSpinBox(); self.sp_minpts.setRange(2, 200); self.sp_minpts.setValue(10)
        f.addRow("MinPts:", self.sp_minpts)

        self.sp_maxeps = QDoubleSpinBox(); self.sp_maxeps.setRange(0.1, 1000); self.sp_maxeps.setValue(999)
        f.addRow("Max eps (∞ = 999):", self.sp_maxeps)

        self.cb_metric = QComboBox()
        self.cb_metric.addItems(['euclidean', 'manhattan', 'cosine', 'chebyshev'])
        f.addRow("Metric:", self.cb_metric)

        self.cb_method = QComboBox()
        self.cb_method.addItems(['xi', 'dbscan'])
        self.cb_method.setCurrentText('xi')
        f.addRow("Extraction:", self.cb_method)

        self.sp_xi = QDoubleSpinBox(); self.sp_xi.setRange(0.001, 0.2); self.sp_xi.setValue(0.05); self.sp_xi.setSingleStep(0.01)
        self.sp_xi.setEnabled(True)
        self.cb_method.currentTextChanged.connect(lambda t: self.sp_xi.setEnabled(t == 'xi'))
        f.addRow("xi value:", self.sp_xi)

        gb2.setLayout(f)
        layout.addWidget(gb2)

        # Run
        gb3 = QGroupBox("3. Run OPTICS")
        v = QVBoxLayout()
        self.btn_run = QPushButton("Run OPTICS Clustering")
        self.btn_run.setStyleSheet("background:#9C27B0; color:white; font-weight:bold; font-size:16px; padding:12px")
        self.btn_run.clicked.connect(self.start_optics)
        v.addWidget(self.btn_run)

        self.prog = QProgressBar(); self.prog.setVisible(False)
        v.addWidget(self.prog)
        gb3.setLayout(v)
        layout.addWidget(gb3)
        layout.addStretch()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            cols = df.select_dtypes('number').columns
            if 'Target' in df.columns:
                cols = cols.drop('Target', errors='ignore')
            self.X = df[cols].values.astype(float)
            self.lbl.setText(f"{path.split('/')[-1]} → {self.X.shape}")
            self.txt.append("Data loaded — Ready for OPTICS!\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_optics(self):
        if self.X is None:
            return QMessageBox.warning(self, "No data", "Load a CSV first!")

        if self.thread and self.thread.isRunning():
            self.thread.quit(); self.thread.wait()

        self.thread = QThread()
        self.worker = OPTICSWorker(
            self.X,
            min_samples=self.sp_minpts.value(),
            max_eps=self.sp_maxeps.value(),
            metric=self.cb_metric.currentText(),
            cluster_method=self.cb_method.currentText(),
            xi=self.sp_xi.value()
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)

        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.thread.start()

    def on_finished(self, labels, reachability, core_distances, X_pca, msg):
        self.prog.setVisible(False)
        if labels is None:
            return QMessageBox.critical(self, "Error", msg)

        # Reachability Plot
        self.ax_reach.clear()
        ordering = np.arange(len(reachability))
        reachability[reachability == np.inf] = reachability[reachability != np.inf].max() * 1.2
        self.ax_reach.bar(ordering, reachability, width=1.0, color='lightblue', edgecolor='navy', alpha=0.8)
        self.ax_reach.set_title("OPTICS Reachability Plot")
        self.ax_reach.set_ylabel("Reachability Distance")
        self.ax_reach.set_xlabel("Processing Order")
        self.canvas_reach.draw()

        # 2D Clusters
        self.ax_2d.clear()
        unique = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))

        for lab, col in zip(unique, colors):
            mask = labels == lab
            if lab == -1:
                self.ax_2d.scatter(X_pca[mask, 0], X_pca[mask, 1], c='gray', s=30, alpha=0.6, label='Noise')
            else:
                self.ax_2d.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=60, edgecolors='k', linewidth=0.5, label=f'Cluster {lab}')

        self.ax_2d.set_title("OPTICS Clusters (PCA 2D)")
        self.ax_2d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.canvas_2d.draw()

        self.txt.clear()
        self.txt.append(f"<pre>{msg}</pre>")
        self.txt.append("OPTICS automatically finds the best number of clusters!\nTry changing MinPts for different granularity.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = OPTICS_GUI()
    win.show()
    sys.exit(app.exec_())