import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QFileDialog,
                             QTextEdit, QGroupBox, QFormLayout, QMessageBox, QProgressBar,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import hdbscan


class HDBSCANWorker(QObject):
    finished = pyqtSignal(object, np.ndarray, np.ndarray, np.ndarray, str)

    def __init__(self, X, min_cluster_size, min_samples, cluster_selection_epsilon, metric):
        super().__init__()
        self.X = X
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or None
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

    def run(self):
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
                cluster_selection_method='eom',  # Best default
                prediction_data=True
            )
            labels = clusterer.fit_predict(self.X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            probabilities = clusterer.probabilities_
            outlier_scores = clusterer.outlier_scores_

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X)

            msg = (f"HDBSCAN Complete!\n"
                   f"Clusters found: {n_clusters}\n"
                   f"Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)\n"
                   f"min_cluster_size={self.min_cluster_size} | min_samples={'auto' if not self.min_samples else self.min_samples}\n"
                   f"Robust, hierarchical, and automatic — the king has spoken.")

            self.finished.emit(clusterer, labels, X_pca, probabilities, msg)
        except Exception as e:
            self.finished.emit(None, None, None, None, f"Error: {str(e)}")


class HDBSCAN_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDBSCAN — The Ultimate Clustering Algorithm")
        self.setGeometry(60, 40, 1450, 950)
        self.setStyleSheet("background:#1e1e1e; color:#ffffff;")
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

        # Right Panel
        right = QVBoxLayout()

        # Top Plot: Clusters
        self.canvas_main = FigureCanvas(plt.figure(figsize=(11, 7)))
        self.ax_main = self.canvas_main.figure.subplots()
        self.ax_main.set_facecolor('#2d2d2d')
        self.canvas_main.figure.patch.set_facecolor('#1e1e1e')
        right.addWidget(self.canvas_main)

        # Bottom: Info
        self.txt = QTextEdit()
        self.txt.setStyleSheet("background:#2d2d2d; color:#00ff00; font-family: Consolas; font-size:11pt;")
        self.txt.setMaximumHeight(160)
        self.txt.setReadOnly(True)
        right.addWidget(self.txt)

        main.addLayout(right, 3)

    def build_controls(self, layout):
        # Title
        title = QLabel("<h2>HDBSCAN</h2><h4>Hierarchical Density-Based Clustering</h4>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Load
        gb1 = QGroupBox("1. Load Dataset")
        gb1.setStyleSheet("QGroupBox { font-weight: bold; }")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV (BMW, Iris, etc.)")
        btn.setStyleSheet("background:#4CAF50; color:white; padding:10px; font-weight:bold;")
        btn.clicked.connect(self.load_csv)
        self.lbl = QLabel("No file loaded")
        self.lbl.setStyleSheet("color:#aaa;")
        v.addWidget(btn); v.addWidget(self.lbl)
        gb1.setLayout(v)
        layout.addWidget(gb1)

        # Parameters
        gb2 = QGroupBox("2. HDBSCAN Parameters")
        gb2.setStyleSheet("QGroupBox { font-weight: bold; }")
        f = QFormLayout()

        self.sp_size = QSpinBox(); self.sp_size.setRange(2, 500); self.sp_size.setValue(15)
        f.addRow("Min Cluster Size:", self.sp_size)

        self.sp_samples = QSpinBox(); self.sp_samples.setRange(1, 100); self.sp_samples.setValue(5)
        self.sp_samples.setSpecialValueText("Auto (recommended)")
        f.addRow("Min Samples (robustness):", self.sp_samples)

        self.sp_eps = QDoubleSpinBox(); self.sp_eps.setRange(0.0, 10.0); self.sp_eps.setValue(0.0); self.sp_eps.setSingleStep(0.1)
        f.addRow("Cluster ε (0 = auto):", self.sp_eps)

        self.cb_metric = QComboBox()
        self.cb_metric.addItems(['euclidean', 'manhattan', 'cosine', 'chebyshev'])
        f.addRow("Distance Metric:", self.cb_metric)

        gb2.setLayout(f)
        layout.addWidget(gb2)

        # Run
        gb3 = QGroupBox("3. Run")
        v = QVBoxLayout()
        self.btn_run = QPushButton("RUN HDBSCAN")
        self.btn_run.setStyleSheet("""
            QPushButton { background:#FF1744; color:white; font-weight:bold; font-size:18px; padding:15px; }
            QPushButton:hover { background:#FF4560; }
        """)
        self.btn_run.clicked.connect(self.start_hdbscan)
        v.addWidget(self.btn_run)

        self.prog = QProgressBar(); self.prog.setVisible(False)
        v.addWidget(self.prog)
        gb3.setLayout(v)
        layout.addWidget(gb3)

        # Footer
        footer = QLabel("HDBSCAN > DBSCAN > OPTICS > K-Means (in most real-world cases)")
        footer.setStyleSheet("color:#888; font-style:italic;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        layout.addStretch()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            cols = df.select_dtypes(include='number').columns
            if 'Target' in df.columns:
                cols = cols.drop('Target', errors='ignore')
            self.X = df[cols].values.astype(float)
            self.lbl.setText(f"{path.split('/')[-1]}\n{self.X.shape[0]} samples × {self.X.shape[1]} features")
            self.txt.append("Data loaded. Ready to cluster with the best algorithm alive.\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_hdbscan(self):
        if self.X is None:
            return QMessageBox.warning(self, "No data", "Load a CSV first!")

        if self.thread and self.thread.isRunning():
            self.thread.quit(); self.thread.wait()

        self.thread = QThread()
        self.worker = HDBSCANWorker(
            self.X,
            min_cluster_size=self.sp_size.value(),
            min_samples=self.sp_samples.value() if self.sp_samples.value() > 1 else None,
            cluster_selection_epsilon=self.sp_eps.value(),
            metric=self.cb_metric.currentText()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)

        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.thread.start()

    def on_finished(self, clusterer, labels, X_pca, probs, msg):
        self.prog.setVisible(False)
        if clusterer is None:
            return QMessageBox.critical(self, "Error", msg)

        self.ax_main.clear()
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, lab in enumerate(unique_labels):
            if lab == -1:
                self.ax_main.scatter(X_pca[labels == lab, 0], X_pca[labels == lab, 1],
                                   c='lightgray', s=40, alpha=0.8, label='Noise', edgecolors='black', linewidth=0.5)
            else:
                color = colors[i % len(colors)]
                self.ax_main.scatter(X_pca[labels == lab, 0], X_pca[labels == lab, 1],
                                   c=[color], s=70, label=f'Cluster {lab}', edgecolors='black', linewidth=0.7)

        self.ax_main.set_title("HDBSCAN Clustering Result (PCA 2D)", fontsize=16, pad=20)
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")
        self.ax_main.grid(True, alpha=0.3)
        self.canvas_main.draw()

        self.txt.clear()
        self.txt.append(f"<pre style='color:#00ff41;'>{msg}</pre>")
        self.txt.append("<span style='color:#ff1744;'>HDBSCAN just owned your data.</span>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = HDBSCAN_GUI()
    win.show()
    sys.exit(app.exec_())