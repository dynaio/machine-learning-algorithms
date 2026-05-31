import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
import hdbscan
import warnings
warnings.filterwarnings('ignore')


class AnimationWorker(QObject):
    finished = pyqtSignal(object, list, str)  # clusterer, frames, message

    def __init__(self, X_2d, algo, params):
        super().__init__()
        self.X_2d = X_2d
        self.algo = algo
        self.params = params

    def run(self):
        frames = []
        try:
            if self.algo == 'dbscan':
                clusterer = DBSCAN(**self.params).fit(self.X_2d)
                labels = clusterer.labels_
                # Simulate point-by-point discovery
                order = np.random.permutation(len(self.X_2d))
                for i in range(len(order)):
                    visited = np.zeros(len(self.X_2d), bool)
                    visited[order[:i+1]] = True
                    circles = [Circle(self.X_2d[order[i]], self.params['eps'], color='red', fill=False, alpha=0.4, lw=2)]
                    frames.append({'visited': visited, 'labels': labels, 'circles': circles, 'current': order[i]})
                msg = f"DBSCAN Animation Ready | {len(set(labels))-1} clusters"

            elif self.algo == 'optics':
                clusterer = OPTICS(**self.params).fit(self.X_2d)
                labels = clusterer.labels_
                order = clusterer.ordering_
                reach = clusterer.reachability_
                for i, idx in enumerate(order):
                    visited = np.zeros(len(self.X_2d), bool)
                    visited[order[:i+1]] = True
                    r = reach[idx] if reach[idx] != np.inf else self.params.get('max_eps', 1.0)
                    r = min(r, 5.0)  # cap for visibility
                    circles = [Circle(self.X_2d[idx], r, color='purple', fill=False, alpha=0.5, lw=1.5)]
                    frames.append({'visited': visited, 'labels': labels, 'circles': circles, 'current': idx})
                msg = f"OPTICS Animation Ready | {len(set(labels))-1} clusters"

            else:  # hdbscan
                clusterer = hdbscan.HDBSCAN(**self.params, prediction_data=True).fit(self.X_2d)
                labels = clusterer.labels_
                probs = clusterer.probabilities_
                order = np.random.permutation(len(self.X_2d))  # HDBSCAN has no natural order
                for i in range(len(order)):
                    visited = np.zeros(len(self.X_2d), bool)
                    visited[order[:i+1]] = True
                    prob = probs[order[i]]
                    radius = prob * 2.0 + 0.1
                    circles = [Circle(self.X_2d[order[i]], radius, color='green', fill=False, alpha=0.6, lw=2)]
                    frames.append({'visited': visited, 'labels': labels, 'circles': circles, 'current': order[i]})
                msg = f"HDBSCAN Animation Ready | {len(set(labels))-1} clusters"

            self.finished.emit(clusterer, frames, msg)

        except Exception as e:
            self.finished.emit(None, [], f"Error: {e}")


class DensityAnimatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Density Clustering Animator — DBSCAN / OPTICS / HDBSCAN")
        self.setGeometry(100, 50, 1400, 900)
        self.X = None
        self.X_2d = None
        self.frames = []
        self.anim = None
        self.worker = None
        self.thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left Panel
        left = QVBoxLayout()
        self.build_controls(left)
        layout.addLayout(left, 1)

        # Right: Canvas + Controls
        right = QVBoxLayout()

        self.canvas = FigureCanvas(plt.figure(figsize=(12, 9)))
        self.ax = self.canvas.figure.subplots()
        right.addWidget(self.canvas)

        # Animation controls
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.set_frame)
        ctrl.addWidget(self.slider)

        self.lbl_frame = QLabel("0 / 0")
        ctrl.addWidget(self.lbl_frame)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed)
        ctrl.addWidget(QLabel("Speed:"))
        ctrl.addWidget(self.speed_slider)

        right.addLayout(ctrl)

        self.log = QTextEdit()
        self.log.setMaximumHeight(120)
        self.log.setReadOnly(True)
        right.addWidget(self.log)

        layout.addLayout(right, 3)

    def build_controls(self, layout):
        # Load
        gb = QGroupBox("Load Data")
        v = QVBoxLayout()
        btn = QPushButton("Load CSV File")
        btn.clicked.connect(self.load_data)
        self.lbl_file = QLabel("No file loaded")
        v.addWidget(btn); v.addWidget(self.lbl_file)
        gb.setLayout(v)
        layout.addWidget(gb)

        # Algorithm
        gb = QGroupBox("Algorithm")
        v = QVBoxLayout()
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(['dbscan', 'optics', 'hdbscan'])
        self.combo_algo.currentTextChanged.connect(self.update_params_ui)
        v.addWidget(self.combo_algo)
        gb.setLayout(v)
        layout.addWidget(gb)

        # Parameters
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)
        self.update_params_ui()

        # Run
        gb = QGroupBox("Run")
        v = QVBoxLayout()
        self.btn_run = QPushButton("Generate Animation")
        self.btn_run.setStyleSheet("background:#2196F3; color:white; font-weight:bold; padding:10px")
        self.btn_run.clicked.connect(self.start_animation)
        v.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        gb.setLayout(v)
        layout.addWidget(gb)
        layout.addStretch()

    def update_params_ui(self):
        algo = self.combo_algo.currentText()
        layout = self.params_layout
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

        if algo == 'dbscan':
            self.eps_spin = QDoubleSpinBox(); self.eps_spin.setRange(0.05, 5.0); self.eps_spin.setValue(0.5); self.eps_spin.setSingleStep(0.05)
            self.minpts_spin = QSpinBox(); self.minpts_spin.setRange(2, 50); self.minpts_spin.setValue(5)
            layout.addRow("eps:", self.eps_spin)
            layout.addRow("minPts:", self.minpts_spin)

        elif algo == 'optics':
            self.minpts_spin = QSpinBox(); self.minpts_spin.setRange(2, 50); self.minpts_spin.setValue(10)
            layout.addRow("min_samples:", self.minpts_spin)

        else:  # hdbscan
            self.minsize_spin = QSpinBox(); self.minsize_spin.setRange(3, 100); self.minsize_spin.setValue(15)
            self.minsample_spin = QSpinBox(); self.minsample_spin.setRange(1, 50); self.minsample_spin.setValue(5)
            layout.addRow("min_cluster_size:", self.minsize_spin)
            layout.addRow("min_samples:", self.minsample_spin)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            cols = df.select_dtypes(include='number').columns
            if 'Target' in df.columns:
                cols = cols.drop('Target', errors='ignore')
            self.X = df[cols].values.astype(float)
            pca = PCA(n_components=2)
            self.X_2d = pca.fit_transform(self.X)
            self.lbl_file.setText(f"{path.split('/')[-1]} → {self.X.shape[0]} points")
            self.log.append("Data loaded & projected to 2D with PCA.\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_animation(self):
        if self.X_2d is None:
            return QMessageBox.warning(self, "No data", "Load CSV first!")

        algo = self.combo_algo.currentText()
        params = {}
        if algo == 'dbscan':
            params = {'eps': self.eps_spin.value(), 'min_samples': self.minpts_spin.value()}
        elif algo == 'optics':
            params = {'min_samples': self.minpts_spin.value(), 'max_eps': np.inf, 'cluster_method': 'xi'}
        else:
            params = {'min_cluster_size': self.minsize_spin.value(),
                      'min_samples': self.minsample_spin.value() if self.minsample_spin.value() > 1 else None}

        if self.thread and self.thread.isRunning():
            self.thread.terminate(); self.thread.wait()

        self.thread = QThread()
        self.worker = AnimationWorker(self.X_2d, algo, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_animation_done)
        self.thread.finished.connect(self.thread.deleteLater)
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        self.thread.start()

    def on_animation_done(self, clusterer, frames, msg):
        self.progress.setVisible(False)
        if not clusterer:
            return QMessageBox.critical(self, "Error", msg)

        self.frames = frames
        self.slider.setRange(0, len(frames)-1)
        self.lbl_frame.setText(f"0 / {len(frames)-1}")
        self.log.append(msg + "\nAnimation ready! Click Play.\n")
        self.current_frame = 0
        self.render_frame(0)

    def render_frame(self, idx):
        if not self.frames: return
        frame = self.frames[idx]
        self.ax.clear()

        visited = frame['visited']
        labels = frame['labels']
        X = self.X_2d

        # Unvisited points
        self.ax.scatter(X[~visited, 0], X[~visited, 1], c='lightgray', s=30, alpha=0.7)

        # Visited points colored by cluster
        unique = np.unique(labels[visited])
        colors = plt.cm.get_cmap('tab10', len(unique))
        for i, lab in enumerate(unique):
            if lab == -1: continue
            mask = (labels == lab) & visited
            self.ax.scatter(X[mask, 0], X[mask, 1], c=[colors(i)], s=60, edgecolor='k', linewidth=0.5)

        # Noise
        noise = (labels == -1) & visited
        self.ax.scatter(X[noise, 0], X[noise, 1], c='black', s=25, marker='x')

        # Epsilon / Reachability circles
        for circle in frame.get('circles', []):
            self.ax.add_patch(circle)

        # Current point
        curr = frame['current']
        self.ax.scatter(X[curr, 0], X[curr, 1], c='red', s=200, marker='*', edgecolor='yellow', linewidth=2)

        self.ax.set_title(f"{self.combo_algo.currentText().upper()} — Frame {idx}/{len(self.frames)-1}")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def set_frame(self, val):
        self.current_frame = val
        self.render_frame(val)
        self.lbl_frame.setText(f"{val} / {len(self.frames)-1}")

    def toggle_play(self):
        if not self.frames:
            return
        if self.anim is None or not self.anim.event_source:
            interval = 1000 // self.speed_slider.value()
            self.anim = FuncAnimation(self.canvas.figure, self.next_frame, frames=len(self.frames),
                                      interval=interval, repeat=True)
            self.btn_play.setText("Pause")
        else:
            self.anim.event_source.stop()
            self.anim = None
            self.btn_play.setText("Play")

    def next_frame(self, _):
        next_idx = (self.current_frame + 1) % len(self.frames)
        self.slider.setValue(next_idx)

    def update_speed(self, val):
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None
            self.toggle_play()
            self.toggle_play()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DensityAnimatorGUI()
    win.show()
    sys.exit(app.exec_())