import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')


class DENCLUEWorker(QThread):
    finished = pyqtSignal(np.ndarray, list, str)  # labels, frames, message

    def __init__(self, X_2d, h, xi, tolerance=0.01):
        super().__init__()
        self.X_2d = X_2d
        self.h = h
        self.xi = xi
        self.tolerance = tolerance

    def gaussian_kernel(self, dist):
        return np.exp(-0.5 * (dist / self.h) ** 2)

    def run(self):
        X = self.X_2d
        n = len(X)
        tree = KDTree(X)
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n, bool)
        frames = []

        def hill_climb(x_start):
            x = x_start.copy()
            for _ in range(50):
                dists, idxs = tree.query(x, k=min(50, n))
                if dists[0] == 0:
                    weights = self.gaussian_kernel(dists[1:])
                    neighbors = X[idxs[1:]]
                else:
                    weights = self.gaussian_kernel(dists)
                    neighbors = X[idxs]
                if len(neighbors) == 0:
                    break
                gradient = np.sum(weights[:, None] * (neighbors - x), axis=0) / (np.sum(weights) + 1e-8)
                step = gradient * (self.h ** 2)
                if np.linalg.norm(step) < self.tolerance:
                    break
                x += step
            return x

        # Animation frames
        attractors = []
        for i in range(n):
            if visited[i]:
                continue
            attractor = hill_climb(X[i])
            dists, idxs = tree.query(attractor, k=n)
            density = np.sum(self.gaussian_kernel(dists))
            if density >= self.xi:
                attractors.append(attractor)
                mask = ~visited[idxs]
                labels[idxs[mask]] = cluster_id
                visited[idxs[mask]] = True
                cluster_id += 1

            # Frame for animation
            frame = {
                'current': i,
                'attractor': attractor,
                'attractors': attractors.copy(),
                'labels': labels.copy(),
                'visited': visited.copy()
            }
            frames.append(frame)

        # Final labeling for noise
        final_labels = labels.copy()
        final_labels[labels == -1] = -1
        msg = f"DENCLUE Complete! Found {cluster_id} clusters + noise"
        self.finished.emit(final_labels, frames, msg)


class DENCLUEGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DENCLUE — Density-Based Clustering with Hill Climbing")
        self.setGeometry(60, 40, 1500, 900)
        self.setStyleSheet("""
            QMainWindow { background: #f8f9fa; }
            QLabel { color: #2c3e50; font-size: 11pt; }
            QGroupBox { font-weight: bold; color: #2980b9; border: 2px solid #3498db; border-radius: 8px; padding-top: 10px; margin: 5px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #e74c3c; color: white; font-weight: bold; padding: 12px; border-radius: 8px; font-size: 14pt; }
            QPushButton:hover { background: #c0392b; }
            QTextEdit { background: white; border: 1px solid #bdc3c7; border-radius: 6px; }
        """)
        self.X_2d = None
        self.frames = []
        self.anim = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left: Controls
        left = QVBoxLayout()
        self.build_controls(left)
        main.addLayout(left, 1)

        # Right: Canvas + Controls
        right = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure(figsize=(12, 8)))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_facecolor('white')
        right.addWidget(self.canvas)

        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("Play Animation")
        self.btn_play.clicked.connect(self.toggle_animation)
        ctrl.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.set_frame)
        ctrl.addWidget(self.slider)

        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 30)
        self.speed.setValue(12)
        ctrl.addWidget(QLabel("Speed:"))
        ctrl.addWidget(self.speed)

        self.lbl_info = QLabel("Ready")
        ctrl.addWidget(self.lbl_info)
        right.addLayout(ctrl)

        self.log = QTextEdit(readOnly=True, maximumHeight=140)
        right.addWidget(self.log)

        main.addLayout(right, 3)

    def build_controls(self, layout):
        title = QLabel("<h1 style='color:#2980b9;'>DENCLUE Animator</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Load
        gb = QGroupBox("1. Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV File")
        btn.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No file loaded")
        v.addWidget(btn); v.addWidget(self.lbl_file)
        gb.setLayout(v)
        layout.addWidget(gb)

        # Parameters
        gb = QGroupBox("2. DENCLUE Parameters")
        form = QFormLayout()
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(0.1, 10.0)
        self.h_spin.setValue(1.0)
        self.h_spin.setSingleStep(0.1)
        form.addRow("Bandwidth (h):", self.h_spin)

        self.xi_spin = QDoubleSpinBox()
        self.xi_spin.setRange(0.001, 1.0)
        self.xi_spin.setValue(0.01)
        self.xi_spin.setSingleStep(0.005)
        form.addRow("Density threshold (ξ):", self.xi_spin)
        gb.setLayout(form)
        layout.addWidget(gb)

        # Run
        gb = QGroupBox("3. Run DENCLUE")
        v = QVBoxLayout()
        self.btn_run = QPushButton("RUN DENCLUE")
        self.btn_run.clicked.connect(self.start_denclue)
        self.btn_run.setEnabled(False)
        v.addWidget(self.btn_run)
        self.progress = QProgressBar(visible=False)
        v.addWidget(self.progress)
        gb.setLayout(v)
        layout.addWidget(gb)

        info = QLabel("<b>DENCLUE</b>: Points climb to local density attractors.<br>"
                     "High h → smoother, fewer clusters<br>"
                     "Low ξ → more clusters", alignment=Qt.AlignCenter)
        info.setStyleSheet("color:#7f8c8d; font-size:10pt;")
        layout.addWidget(info)
        layout.addStretch()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            cols = df.select_dtypes(include='number').columns
            X = df[cols].values.astype(float)
            if X.shape[1] > 2:
                self.X_2d = PCA(n_components=2).fit_transform(X)
                self.log.append("High-dim → PCA (2D)\n")
            else:
                self.X_2d = X
            self.lbl_file.setText(f"{path.split('/')[-1]} → {len(X)} points")
            self.btn_run.setEnabled(True)
            self.log.append("Data loaded! Click RUN.\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_denclue(self):
        if self.X_2d is None: return
        self.worker = DENCLUEWorker(self.X_2d, self.h_spin.value(), self.xi_spin.value())
        self.worker.finished.connect(self.on_finished)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.worker.start()

    def on_finished(self, labels, frames, msg):
        self.progress.setVisible(False)
        self.frames = frames
        self.slider.setRange(0, len(frames)-1)
        self.log.append(f"{msg}\nAnimation ready!\n")
        self.render_frame(0)

    def render_frame(self, idx):
        if not self.frames: return
        frame = self.frames[idx]
        self.ax.clear()

        X = self.X_2d
        labels = frame['labels']
        visited = frame['visited']

        # Unvisited
        self.ax.scatter(X[~visited, 0], X[~visited, 1], c='lightgray', s=40, alpha=0.8)

        # Clustered
        unique = np.unique(labels[labels >= 0])
        colors = plt.cm.get_cmap('tab10', len(unique) if len(unique) > 0 else 1)
        for i, cid in enumerate(unique):
            mask = labels == cid
            self.ax.scatter(X[mask, 0], X[mask, 1], c=[colors(i)], s=80, edgecolor='black', linewidth=0.8)

        # Noise
        noise = (labels == -1) & visited
        self.ax.scatter(X[noise, 0], X[noise, 1], c='black', s=30, marker='x')

        # Current point
        if 'current' in frame:
            i = frame['current']
            self.ax.scatter(X[i, 0], X[i, 1], c='red', s=200, marker='*', edgecolor='yellow', linewidth=3)

        # Attractor
        if 'attractor' in frame:
            a = frame['attractor']
            self.ax.scatter(a[0], a[1], c='gold', s=300, marker='D', edgecolor='red', linewidth=3)
            # Draw path
            if 'current' in frame:
                self.ax.plot([X[i, 0], a[0]], [X[i, 1], a[1]], 'r--', alpha=0.6, lw=2)

        # All attractors
        for attr in frame.get('attractors', []):
            self.ax.scatter(attr[0], attr[1], c='orange', s=100, marker='D', edgecolor='darkred', linewidth=1.5)

        self.ax.set_title(f"DENCLUE — Frame {idx+1}/{len(self.frames)} | Clusters: {len(unique)}")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.lbl_info.setText(f"Frame {idx+1}/{len(self.frames)}")

    def set_frame(self, val):
        self.render_frame(val)

    def toggle_animation(self):
        if not self.frames: return
        if self.anim is None:
            interval = 1000 // max(1, self.speed.value())
            self.anim = FuncAnimation(self.canvas.figure, lambda i: self.slider.setValue(i),
                                      frames=len(self.frames), interval=interval, repeat=True)
            self.btn_play.setText("Pause")
        else:
            self.anim.event_source.stop()
            self.anim = None
            self.btn_play.setText("Play Animation")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DENCLUEGUI()
    win.show()
    sys.exit(app.exec_())