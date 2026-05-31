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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class DBSCANWorker(QObject):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, str)  # labels, X_pca, core_mask, msg
    error = pyqtSignal(str)

    def __init__(self, X, eps, min_samples, metric):
        super().__init__()
        self.X = X
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def run(self):
        try:
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, n_jobs=-1)
            labels = db.fit_predict(self.X)
            core_mask = np.zeros_like(labels, dtype=bool)
            core_mask[db.core_sample_indices_] = True

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X)

            msg = (f"DBSCAN Finished\n"
                   f"Clusters: {n_clusters} | Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)\n"
                   f"eps={self.eps:.3f} | minPts={self.min_samples} | metric={self.metric}")

            if 1 < n_clusters < len(labels):
                sil = silhouette_score(self.X, labels)
                msg += f"\nSilhouette Score: {sil:.3f}"

            self.finished.emit(labels, X_pca, core_mask, msg)
        except Exception as e:
            self.error.emit(str(e))


class DBSCANGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BMW Sales — DBSCAN Pro (Stable)")
        self.setGeometry(100, 50, 1350, 900)
        self.X = None
        self.worker = None
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left
        left = QVBoxLayout()
        self.build_controls(left)
        main.addLayout(left, 1)

        # Right
        right = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        self.ax = self.canvas.figure.subplots()
        right.addWidget(self.canvas)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        right.addWidget(self.txt, 1)
        main.addLayout(right, 3)

    def build_controls(self, layout):
        # Load
        gb1 = QGroupBox("1. Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV")
        btn.clicked.connect(self.load_csv)
        self.lbl = QLabel("No file")
        v.addWidget(btn); v.addWidget(self.lbl)
        gb1.setLayout(v)
        layout.addWidget(gb1)

        # Params
        gb2 = QGroupBox("2. DBSCAN Parameters")
        f = QFormLayout()

        self.sp_eps = QDoubleSpinBox(); self.sp_eps.setRange(0.01, 100); self.sp_eps.setValue(0.5); self.sp_eps.setSingleStep(0.05)
        f.addRow("eps:", self.sp_eps)

        self.sp_min = QSpinBox(); self.sp_min.setRange(2, 200); self.sp_min.setValue(5)
        f.addRow("MinPts:", self.sp_min)

        self.cb_metric = QComboBox()
        self.cb_metric.addItems(['euclidean', 'manhattan', 'cosine', 'chebyshev'])
        f.addRow("Metric:", self.cb_metric)

        self.chk_auto = QCheckBox("Auto-suggest eps")
        self.chk_auto.setChecked(True)
        f.addRow(self.chk_auto)

        gb2.setLayout(f)
        layout.addWidget(gb2)

        # Run
        gb3 = QGroupBox("3. Actions")
        v = QVBoxLayout()
        btn_kdist = QPushButton("k-distance Graph")
        btn_kdist.clicked.connect(self.plot_kdist)
        v.addWidget(btn_kdist)

        self.btn_run = QPushButton("Run DBSCAN")
        self.btn_run.setStyleSheet("background:#e91e63; color:white; font-weight:bold; padding:10px")
        self.btn_run.clicked.connect(self.start_dbscan)
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
            cols = df.select_dtypes(include='number').columns
            if 'Target' in df.columns:
                cols = cols.drop('Target', errors='ignore')
            self.X = df[cols].values.astype(float)
            self.lbl.setText(f"{path.split('/')[-1]} → {self.X.shape}")
            self.txt.append("Data loaded!\n")
            if self.chk_auto.isChecked():
                self.plot_kdist()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_kdist(self):
        if self.X is None: return
        k = self.sp_min.value()
        nn = NearestNeighbors(n_neighbors=k).fit(self.X)
        dists, _ = nn.kneighbors(self.X)
        kdist = np.sort(dists[:, k-1])

        self.ax.clear()
        self.ax.plot(kdist, 'b.-', markersize=3)
        self.ax.set_title(f"k-distance (k={k})")
        self.ax.set_ylabel(f"{k}th neighbor distance")
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(self.sp_eps.value(), color='r', linestyle='--', label=f"eps={self.sp_eps.value()}")
        self.ax.legend()
        self.canvas.draw()

        # Auto eps
        try:
            from kneed import KneeLocator
            kl = KneeLocator(range(len(kdist)), kdist, curve='convex', direction='increasing')
            if kl.knee:
                self.sp_eps.setValue(round(kdist[kl.knee], 3))
                self.txt.append(f"Auto eps → {kdist[kl.knee]:.3f}\n")
        except:
            pass

    def start_dbscan(self):
        if self.X is None:
            QMessageBox.warning(self, "No data", "Load CSV first")
            return

        # Clean previous thread
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        self.thread = QThread()
        self.worker = DBSCANWorker(self.X, self.sp_eps.value(), self.sp_min.value(), self.cb_metric.currentText())
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.prog.setVisible(True)
        self.prog.setRange(0, 0)
        self.thread.start()

    def on_finished(self, labels, X_pca, core_mask, msg):
        self.prog.setVisible(False)
        self.ax.clear()

        unique = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))

        for lab, col in zip(unique, colors):
            if lab == -1:
                self.ax.scatter(X_pca[labels == lab, 0], X_pca[labels == lab, 1],
                                c='gray', s=20, alpha=0.6, label='Noise')
            else:
                self.ax.scatter(X_pca[labels == lab, 0], X_pca[labels == lab, 1],
                                c=[col], s=50, edgecolors='k', linewidth=0.5)

        # Core points
        self.ax.scatter(X_pca[core_mask, 0], X_pca[core_mask, 1],
                        c='yellow', s=80, edgecolors='red', linewidth=2, label='Core Points')

        self.ax.set_title("DBSCAN Result")
        self.ax.legend()
        self.canvas.draw()

        self.txt.clear()
        self.txt.append(f"<pre>{msg}</pre>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DBSCANGUI()
    win.show()
    sys.exit(app.exec_())