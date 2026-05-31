import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


class HierarchicalWorker(QObject):
    finished = pyqtSignal(object, list, str)  # linkage_matrix, frames, msg

    def __init__(self, X_2d, method, metric, divisive=False):
        super().__init__()
        self.X_2d = X_2d
        self.method = method
        self.metric = metric
        self.divisive = divisive

    def run(self):
        frames = []
        try:
            if not self.divisive:
                # Agglomerative: bottom-up
                Z = linkage(self.X_2d, method=self.method, metric=self.metric)
                max_d = Z[-1, 2]
                thresholds = np.linspace(0.01, max_d * 1.1, 80)
                for t in thresholds:
                    labels = fcluster(Z, t=t, criterion='distance')
                    frames.append({'type': 'agglomerative', 'Z': Z, 'threshold': t, 'labels': labels})
                msg = f"Agglomerative ({self.method} + {self.metric}) — {len(set(labels))-1} clusters max"
            else:
                # Divisive: top-down (bisecting k-means style)
                labels_final, history = self._divisive_clustering(self.X_2d)
                for step, labels in enumerate(history):
                    frames.append({'type': 'divisive', 'step': step, 'labels': labels})
                msg = f"Divisive (Bisecting K-Means) — {len(set(labels_final))} clusters"

            self.finished.emit(Z if not self.divisive else None, frames, msg)

        except Exception as e:
            self.finished.emit(None, [], f"Error: {e}")

    def _divisive_clustering(self, X, max_clusters=12):
        from sklearn.cluster import KMeans
        labels = np.zeros(len(X), int)
        history = [labels.copy()]
        to_split = [0]
        cluster_id = 1

        while len(to_split) > 0 and cluster_id < max_clusters:
            current = to_split.pop(0)
            mask = labels == current
            if mask.sum() < 8:
                continue
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X[mask])
            sub = km.labels_
            labels[mask] = np.where(sub == 0, current, cluster_id)
            to_split.extend([current, cluster_id])
            cluster_id += 1
            history.append(labels.copy())

        return labels, history


class HierarchicalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hierarchical Clustering Animator — Agglomerative & Divisive")
        self.setGeometry(80, 40, 1500, 960)
        self.X_2d = None
        self.frames = []
        self.Z = None
        self.anim = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left: Controls
        left = QVBoxLayout()
        self.build_controls(left)
        main.addLayout(left, 1)

        # Right: Two plots
        right = QVBoxLayout()

        # Top: Main clustering animation
        self.canvas_main = FigureCanvas(plt.figure(figsize=(10, 7)))
        self.ax_main = self.canvas_main.figure.subplots()
        right.addWidget(self.canvas_main)

        # Bottom: Dendrogram
        self.canvas_dendro = FigureCanvas(plt.figure(figsize=(10, 3)))
        self.ax_dendro = self.canvas_dendro.figure.subplots()
        right.addWidget(self.canvas_dendro)

        # Animation controls
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("Play Animation")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.set_frame)
        ctrl.addWidget(self.slider)

        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 30)
        self.speed.setValue(15)
        ctrl.addWidget(QLabel("Speed:"))
        ctrl.addWidget(self.speed)

        self.lbl_info = QLabel("Ready")
        ctrl.addWidget(self.lbl_info)

        right.addLayout(ctrl)

        self.log = QTextEdit(readOnly=True, maximumHeight=100)
        right.addWidget(self.log)

        main.addLayout(right, 3)

    def build_controls(self, layout):
        # Load
        gb = QGroupBox("1. Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV File")
        btn.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No file loaded")
        v.addWidget(btn); v.addWidget(self.lbl_file)
        gb.setLayout(v)
        layout.addWidget(gb)

        # Type
        gb = QGroupBox("2. Type")
        v = QVBoxLayout()
        self.radio_agg = QRadioButton("Agglomerative (Bottom-Up)")
        self.radio_div = QRadioButton("Divisive (Top-Down)")
        self.radio_agg.setChecked(True)
        v.addWidget(self.radio_agg); v.addWidget(self.radio_div)
        gb.setLayout(v)
        layout.addWidget(gb)

        # Linkage & Metric
        gb = QGroupBox("3. Parameters")
        form = QFormLayout()

        self.combo_linkage = QComboBox()
        self.combo_linkage.addItems(['ward', 'complete', 'average', 'single'])
        self.combo_linkage.setCurrentText('ward')
        form.addRow("Linkage:", self.combo_linkage)

        self.combo_metric = QComboBox()
        self.combo_metric.addItems(['euclidean', 'cityblock', 'cosine', 'correlation'])
        self.combo_metric.setCurrentText('euclidean')
        self.combo_metric.setEnabled(True)
        self.combo_linkage.currentTextChanged.connect(
            lambda x: self.combo_metric.setEnabled(x != 'ward')
        )
        form.addRow("Distance Metric:", self.combo_metric)

        gb.setLayout(form)
        layout.addWidget(gb)

        # Run
        gb = QGroupBox("4. Run")
        v = QVBoxLayout()
        self.btn_run = QPushButton("Start Hierarchical Animation")
        self.btn_run.setStyleSheet("background:#9C27B0; color:white; font-weight:bold; padding:12px; font-size:14pt")
        self.btn_run.clicked.connect(self.start_animation)
        v.addWidget(self.btn_run)
        self.progress = QProgressBar(visible=False)
        v.addWidget(self.progress)
        gb.setLayout(v)
        layout.addWidget(gb)
        layout.addStretch()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path: return
        df = pd.read_csv(path)
        cols = df.select_dtypes(include='number').columns
        if 'Target' in df.columns:
            cols = cols.drop('Target', errors='ignore')
        X = df[cols].values.astype(float)
        self.X_2d = PCA(n_components=2).fit_transform(X)
        self.lbl_file.setText(f"{path.split('/')[-1]} → {len(X)} points")
        self.log.append("Data loaded + PCA → 2D\n")

    def start_animation(self):
        if self.X_2d is None:
            return QMessageBox.warning(self, "No data", "Load CSV first!")

        method = self.combo_linkage.currentText()
        metric = self.combo_metric.currentText() if method != 'ward' else 'euclidean'
        divisive = self.radio_div.isChecked()

        worker = HierarchicalWorker(self.X_2d, method, metric, divisive)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda Z, frames, msg: self.on_done(thread, worker, Z, frames, msg))
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        thread.start()

    def on_done(self, thread, worker, Z, frames, msg):
        self.progress.setVisible(False)
        thread.quit(); thread.wait(); worker.deleteLater()
        if not frames:
            return QMessageBox.critical(self, "Error", msg)

        self.Z = Z
        self.frames = frames
        self.slider.setRange(0, len(frames)-1)
        self.log.append(f"{msg}\nAnimation ready! Click Play.\n")
        self.render_frame(0)

    def render_frame(self, idx):
        if not self.frames: return
        frame = self.frames[idx]
        self.ax_main.clear()
        self.ax_dendro.clear()

        X = self.X_2d
        labels = frame['labels']

        # Color points
        unique = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique))
        for i, lab in enumerate(unique):
            mask = labels == lab
            self.ax_main.scatter(X[mask, 0], X[mask, 1], c=[colors(i)], s=70, edgecolor='k', linewidth=0.6, label=f'Cluster {lab}')

        # Dendrogram (only for agglomerative)
        if frame['type'] == 'agglomerative' and self.Z is not None:
            dendrogram(self.Z, ax=self.ax_dendro, color_threshold=frame['threshold'], above_threshold_color='gray')
            self.ax_dendro.axhline(frame['threshold'], color='red', linestyle='--', linewidth=2, label=f'Cut = {frame["threshold"]:.2f}')
            self.ax_dendro.set_title("Dendrogram + Cut Line (Red)")
            self.ax_dendro.legend()
            self.canvas_dendro.draw()

        self.ax_main.set_title(f"Hierarchical Clustering — Frame {idx+1}/{len(self.frames)}")
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax_main.grid(True, alpha=0.3)
        self.canvas_main.draw()
        self.lbl_info.setText(f"Frame {idx+1}/{len(self.frames)} | Clusters: {len(unique)}")

    def set_frame(self, val):
        self.render_frame(val)

    def toggle_play(self):
        if not self.frames: return
        if self.anim is None:
            interval = 1000 // max(1, self.speed.value())
            self.anim = FuncAnimation(self.canvas_main.figure, lambda i: self.slider.setValue(i),
                                      frames=len(self.frames), interval=interval, repeat=True)
            self.btn_play.setText("Pause")
        else:
            self.anim.event_source.stop()
            self.anim = None
            self.btn_play.setText("Play Animation")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = HierarchicalGUI()
    win.show()
    sys.exit(app.exec_())