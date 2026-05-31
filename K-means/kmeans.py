import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
                             QTextEdit, QGroupBox, QFormLayout, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sklearn.metrics.pairwise as pw
import warnings
warnings.filterwarnings('ignore')


class AdvancedKMeans:
    def __init__(self, n_clusters=3, init='kmeans++', max_iter=300, tol=1e-4,
                 random_state=42, distance_metric='euclidean', p=2):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.p = p
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _dist(self, X, Y):
        m = self.distance_metric
        if m == 'minkowski':
            return pw.pairwise_distances(X, Y, metric='minkowski', p=self.p)
        return pw.pairwise_distances(X, Y, metric=m)

    def _init_centroids(self, X):
        np.random.seed(self.random_state)
        n = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        if self.init == 'random_sample':
            idx = np.random.choice(n, self.n_clusters, replace=False)
            centroids = X[idx]
        elif self.init == 'random_space':
            idx = np.random.randint(0, n, self.n_clusters)
            centroids = X[idx]
        else:  # kmeans++
            centroids[0] = X[np.random.randint(n)]
            for i in range(1, self.n_clusters):
                d = np.min(self._dist(X, centroids[:i]), axis=1)
                probs = d ** 2 / (d ** 2).sum()
                cumprobs = probs.cumsum()
                r = np.random.rand()
                for j, p in enumerate(cumprobs):
                    if r < p:
                        centroids[i] = X[j]
                        break
        return centroids

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            distances = self._dist(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points) > 0:
                    cen = points.mean(axis=0)
                    if self.distance_metric == 'cosine':
                        norm = np.linalg.norm(cen)
                        if norm > 0:
                            cen /= norm
                    new_centroids.append(cen)
                else:
                    new_centroids.append(X[np.random.randint(len(X))])
            new_centroids = np.array(new_centroids)

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        distances = self._dist(X, self.centroids)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)
        return self


# -------------------------- Worker Thread --------------------------
class Worker(QThread):
    finished = pyqtSignal(object, object, object, str)  # kmeans, X_pca, centroids_pca, message

    def __init__(self, X, params):
        super().__init__()
        self.X = X
        self.params = params

    def run(self):
        try:
            km = AdvancedKMeans(
                n_clusters=self.params['k'],
                init=self.params['init'],
                max_iter=self.params['max_iter'],
                tol=self.params['tol'],
                distance_metric=self.params['metric'],
                p=self.params['p'],
                random_state=42
            )
            km.fit(self.X)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X)
            centroids_pca = pca.transform(km.centroids)

            sil = silhouette_score(self.X, km.labels_)
            msg = f"Done! → k={km.n_clusters} | Inertia={km.inertia_:.1f} | Silhouette={sil:.3f}"
            self.finished.emit(km, X_pca, centroids_pca, msg)
        except Exception as e:
            self.finished.emit(None, None, None, f"Error: {str(e)}")


# -------------------------- Main GUI --------------------------
class KMeansGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BMW Sales – Advanced K-Means Clustering")
        self.setGeometry(100, 50, 1350, 900)
        self.X = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel
        left = QVBoxLayout()
        self.build_controls(left)
        main_layout.addLayout(left, 1)

        # Right panel
        right = QVBoxLayout()
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.ax = self.canvas.figure.add_subplot(111)
        right.addWidget(self.canvas)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        right.addWidget(self.txt, 1)

        main_layout.addLayout(right, 3)

    def build_controls(self, layout):
        # Load
        gb1 = QGroupBox("1. Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV File")
        btn.clicked.connect(self.load_csv)
        self.lbl_file = QLabel("No file loaded")
        v.addWidget(btn)
        v.addWidget(self.lbl_file)
        gb1.setLayout(v)
        layout.addWidget(gb1)

        # Parameters
        gb2 = QGroupBox("2. Parameters")
        f = QFormLayout()

        self.cb_metric = QComboBox()
        self.cb_metric.addItems(['euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'])
        f.addRow("Distance:", self.cb_metric)

        self.cb_init = QComboBox()
        self.cb_init.addItems(['kmeans++', 'random_sample', 'random_space'])
        f.addRow("Init:", self.cb_init)

        self.sp_k = QSpinBox(); self.sp_k.setRange(2, 20); self.sp_k.setValue(4)
        f.addRow("k:", self.sp_k)

        self.sp_iter = QSpinBox(); self.sp_iter.setRange(10, 1000); self.sp_iter.setValue(300)
        f.addRow("Max iter:", self.sp_iter)

        self.sp_tol = QDoubleSpinBox(); self.sp_tol.setRange(1e-8, 1e-1); self.sp_tol.setValue(1e-4)
        f.addRow("Tolerance:", self.sp_tol)

        self.sp_p = QDoubleSpinBox(); self.sp_p.setRange(1, 10); self.sp_p.setValue(2)
        self.sp_p.setEnabled(False)
        self.cb_metric.currentTextChanged.connect(lambda t: self.sp_p.setEnabled(t == 'minkowski'))
        f.addRow("Minkowski p:", self.sp_p)

        gb2.setLayout(f)
        layout.addWidget(gb2)

        # Run
        gb3 = QGroupBox("3. Run")
        v = QVBoxLayout()
        btn_elbow = QPushButton("Elbow + Silhouette (find best k)")
        btn_elbow.clicked.connect(self.run_elbow)
        v.addWidget(btn_elbow)

        self.btn_run = QPushButton("Run K-Means Clustering")
        self.btn_run.setStyleSheet("font-weight: bold; background:#2196F3; color:white; padding:10px;")
        self.btn_run.clicked.connect(self.run_clustering)
        v.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        gb3.setLayout(v)
        layout.addWidget(gb3)

        layout.addStretch()

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            num_cols = df.select_dtypes(include='number').columns
            if 'Target' in df.columns:
                num_cols = num_cols.drop('Target', errors='ignore')
            self.X = df[num_cols].values
            self.lbl_file.setText(f"{path.split('/')[-1]} → {self.X.shape[0]} rows, {self.X.shape[1]} cols")
            self.txt.append("Data loaded successfully!\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_elbow(self):
        if self.X is None:
            return QMessageBox.warning(self, "No data", "Load CSV first!")
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        inertias = []; silhouettes = []
        for k in range(2, 11):
            km = AdvancedKMeans(n_clusters=k, init='kmeans++')
            km.fit(self.X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self.X, km.labels_))

        self.ax.clear()
        self.ax.plot(range(2, 11), inertias, 'bo-', label='Inertia')
        self.ax.set_xlabel('k'); self.ax.set_ylabel('Inertia', color='b')
        self.ax2 = self.ax.twinx()
        self.ax2.plot(range(2, 11), silhouettes, 'ro-', label='Silhouette')
        self.ax2.set_ylabel('Silhouette', color='r')
        self.ax.set_title("Elbow + Silhouette")
        self.ax.grid(True)
        self.canvas.draw()
        self.progress.setVisible(False)

    def run_clustering(self):
        if self.X is None:
            return QMessageBox.warning(self, "No data", "Load CSV first!")

        params = {
            'k': self.sp_k.value(),
            'init': self.cb_init.currentText(),
            'max_iter': self.sp_iter.value(),
            'tol': self.sp_tol.value(),
            'metric': self.cb_metric.currentText(),
            'p': self.sp_p.value()
        }

        self.worker = Worker(self.X, params)
        self.worker.finished.connect(self.clustering_finished)
        self.worker.start()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

    def clustering_finished(self, km, X_pca, centroids_pca, msg):
        self.progress.setVisible(False)
        if km is None:
            return QMessageBox.critical(self, "Error", msg)

        self.ax.clear()
        scatter = self.ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap='tab10', alpha=0.7, s=60)
        self.ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        c='red', marker='X', s=400, linewidths=4, label='Centroids')
        self.ax.set_title(f"K-Means (k={km.n_clusters}) – {km.init} – {km.distance_metric}")
        self.ax.legend()
        self.canvas.draw()

        self.txt.clear()
        self.txt.append(f"<h3>{msg}</h3>")
        self.txt.append("<pre>" + pd.DataFrame(km.centroids).round(4).to_string() + "</pre>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = KMeansGUI()
    win.show()
    sys.exit(app.exec_())