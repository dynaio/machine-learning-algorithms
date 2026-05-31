import sys
import random
import math
import numpy as np
from scipy import stats
from collections import deque
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.cm as cm

class MM1QueueSimulation:
    def __init__(self, lambda_rate=0.8, mu_rate=1.0, max_customers=1000):
        self.lambda_rate = lambda_rate  # Arrival rate
        self.mu_rate = mu_rate  # Service rate
        self.max_customers = max_customers
        self.reset()
    
    def reset(self):
        self.queue = deque()
        self.server_busy = False
        self.current_time = 0
        self.next_arrival = self.generate_interarrival()
        self.next_departure = float('inf')
        self.queue_length_history = []
        self.time_history = []
        self.customers_served = 0
        self.total_wait_time = 0
        self.total_system_time = 0
        self.arrival_times = []
        self.departure_times = []
    
    def generate_interarrival(self):
        return random.expovariate(self.lambda_rate)
    
    def generate_service(self):
        return random.expovariate(self.mu_rate)
    
    def run_simulation(self):
        self.reset()
        
        while self.customers_served < self.max_customers:
            # Process next event
            if self.next_arrival < self.next_departure:
                # Arrival event
                self.current_time = self.next_arrival
                self.handle_arrival()
                self.next_arrival = self.current_time + self.generate_interarrival()
            else:
                # Departure event
                self.current_time = self.next_departure
                self.handle_departure()
            
            # Record queue length
            self.queue_length_history.append(len(self.queue))
            self.time_history.append(self.current_time)
        
        return self.calculate_metrics()
    
    def handle_arrival(self):
        self.arrival_times.append(self.current_time)
        if not self.server_busy:
            self.server_busy = True
            service_time = self.generate_service()
            self.next_departure = self.current_time + service_time
        else:
            self.queue.append(self.current_time)
    
    def handle_departure(self):
        self.customers_served += 1
        self.departure_times.append(self.current_time)
        
        if self.queue:
            arrival_time = self.queue.popleft()
            wait_time = self.current_time - arrival_time
            self.total_wait_time += wait_time
            service_time = self.generate_service()
            system_time = wait_time + service_time
            self.total_system_time += system_time
            self.next_departure = self.current_time + service_time
        else:
            self.server_busy = False
            self.next_departure = float('inf')
    
    def calculate_metrics(self):
        # Experimental metrics
        L_exp = np.mean(self.queue_length_history) if self.queue_length_history else 0
        W_exp = self.total_system_time / self.customers_served if self.customers_served > 0 else 0
        Wq_exp = self.total_wait_time / self.customers_served if self.customers_served > 0 else 0
        rho = self.lambda_rate / self.mu_rate
        
        # Theoretical metrics (M/M/1 formulas)
        L_theory = rho / (1 - rho)
        Lq_theory = rho**2 / (1 - rho)
        W_theory = 1 / (self.mu_rate - self.lambda_rate)
        Wq_theory = rho / (self.mu_rate - self.lambda_rate)
        
        return {
            'experimental': {
                'L': L_exp,  # Avg customers in system
                'Lq': L_exp - rho,  # Avg customers in queue
                'W': W_exp,  # Avg time in system
                'Wq': Wq_exp,  # Avg waiting time in queue
                'rho': rho,  # Utilization
                'queue_history': (self.time_history, self.queue_length_history)
            },
            'theoretical': {
                'L': L_theory,
                'Lq': Lq_theory,
                'W': W_theory,
                'Wq': Wq_theory,
                'rho': rho
            }
        }

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f8f9fa')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def apply_style(self):
        self.axes.set_facecolor('#ffffff')
        self.fig.patch.set_facecolor('#f8f9fa')

class AnimatedGraph(MplCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_style()
        self.x_data = []
        self.y_data = []
        self.line, = self.axes.plot([], [], 'b-', linewidth=2, alpha=0.7)
        self.axes.grid(True, alpha=0.3)
        
    def update_data(self, x_data, y_data, title="", xlabel="", ylabel=""):
        self.line.set_data(x_data, y_data)
        self.axes.set_xlim(0, max(x_data) if x_data else 1)
        self.axes.set_ylim(0, max(y_data) * 1.1 if y_data else 1)
        self.axes.set_title(title, fontsize=12, fontweight='bold')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.draw()

class QueueSimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulation = MM1QueueSimulation()
        self.init_ui()
        self.screen_size = QApplication.primaryScreen().size()
        
    def init_ui(self):
        self.setWindowTitle("M/M/1 Queue Simulation - Markov Process")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        central_widget = QWidget()
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        
        # Create tabs
        self.setup_tabs()
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Apply styles
        self.apply_styles()
        
        # Run initial simulation
        self.run_simulation()
    
    def create_control_panel(self):
        panel = QGroupBox("Simulation Parameters")
        layout = QGridLayout()
        
        # Lambda input
        layout.addWidget(QLabel("Arrival Rate (λ):"), 0, 0)
        self.lambda_input = QDoubleSpinBox()
        self.lambda_input.setRange(0.1, 0.95)
        self.lambda_input.setValue(0.8)
        self.lambda_input.setSingleStep(0.05)
        self.lambda_input.setDecimals(2)
        layout.addWidget(self.lambda_input, 0, 1)
        
        # Mu input
        layout.addWidget(QLabel("Service Rate (μ):"), 0, 2)
        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(0.2, 5.0)
        self.mu_input.setValue(1.0)
        self.mu_input.setSingleStep(0.1)
        layout.addWidget(self.mu_input, 0, 3)
        
        # Customers input
        layout.addWidget(QLabel("Number of Customers:"), 1, 0)
        self.customers_input = QSpinBox()
        self.customers_input.setRange(100, 999999999)
        self.customers_input.setValue(1000)
        self.customers_input.setSingleStep(100)
        layout.addWidget(self.customers_input, 1, 1)
        
        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(self.run_button, 1, 2, 1, 2)
        
        # Real-time displays
        layout.addWidget(QLabel("Current ρ (λ/μ):"), 2, 0)
        self.rho_label = QLabel("0.80")
        self.rho_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.rho_label, 2, 1)
        
        layout.addWidget(QLabel("Stability:"), 2, 2)
        self.stability_label = QLabel("Stable")
        self.stability_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addWidget(self.stability_label, 2, 3)
        
        panel.setLayout(layout)
        return panel
    
    def setup_tabs(self):
        # Tab 1: Queue Evolution
        self.queue_tab = QWidget()
        self.setup_queue_tab()
        self.tab_widget.addTab(self.queue_tab, "📈 Queue Evolution")
        
        # Tab 2: Theoretical Values
        self.theory_tab = QWidget()
        self.setup_theory_tab()
        self.tab_widget.addTab(self.theory_tab, "📚 Theoretical Results")
        
        # Tab 3: Experimental Results
        self.experimental_tab = QWidget()
        self.setup_experimental_tab()
        self.tab_widget.addTab(self.experimental_tab, "🔬 Experimental Results")
        
        # Tab 4: Comparison
        self.comparison_tab = QWidget()
        self.setup_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "⚖️ Comparison")
        
        # Tab 5: Convergence Analysis
        self.convergence_tab = QWidget()
        self.setup_convergence_tab()
        self.tab_widget.addTab(self.convergence_tab, "🔄 Convergence")
    
    def setup_queue_tab(self):
        layout = QVBoxLayout()
        
        # Queue evolution graph
        self.queue_graph = AnimatedGraph(self, width=10, height=5)
        layout.addWidget(self.queue_graph)
        
        # Statistics panel
        stats_panel = QGroupBox("Real-time Statistics")
        stats_layout = QGridLayout()
        
        self.current_queue_label = QLabel("0")
        self.current_queue_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2196F3;")
        stats_layout.addWidget(QLabel("Current Queue Length:"), 0, 0)
        stats_layout.addWidget(self.current_queue_label, 0, 1)
        
        self.avg_queue_label = QLabel("0")
        self.avg_queue_label.setStyleSheet("font-size: 18px; color: #4CAF50;")
        stats_layout.addWidget(QLabel("Average Queue Length:"), 1, 0)
        stats_layout.addWidget(self.avg_queue_label, 1, 1)
        
        stats_panel.setLayout(stats_layout)
        layout.addWidget(stats_panel)
        
        self.queue_tab.setLayout(layout)
    
    def setup_theory_tab(self):
        layout = QVBoxLayout()
        
        # Create a scroll area for the theory tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        theory_content = QWidget()
        scroll.setWidget(theory_content)
        theory_layout = QVBoxLayout(theory_content)
        
        # Formulas and explanations
        formula_group = QGroupBox("M/M/1 Queue Formulas")
        formula_layout = QVBoxLayout()
        
        formulas = [
            "Utilization Factor: ρ = λ / μ",
            "Avg Customers in System: L = ρ / (1 - ρ)",
            "Avg Customers in Queue: Lq = ρ² / (1 - ρ)",
            "Avg Time in System: W = 1 / (μ - λ)",
            "Avg Waiting Time: Wq = ρ / (μ - λ)"
        ]
        
        for formula in formulas:
            label = QLabel(formula)
            label.setStyleSheet("font-family: 'Courier New'; font-size: 14px;")
            formula_layout.addWidget(label)
        
        formula_group.setLayout(formula_layout)
        theory_layout.addWidget(formula_group)
        
        # Theoretical values display
        values_group = QGroupBox("Theoretical Values")
        values_layout = QGridLayout()
        
        self.theory_L = QLabel("0.00")
        self.theory_Lq = QLabel("0.00")
        self.theory_W = QLabel("0.00")
        self.theory_Wq = QLabel("0.00")
        
        metrics = [
            ("L (Avg in System):", self.theory_L),
            ("Lq (Avg in Queue):", self.theory_Lq),
            ("W (Avg Time in System):", self.theory_W),
            ("Wq (Avg Waiting Time):", self.theory_Wq)
        ]
        
        for i, (label_text, value_label) in enumerate(metrics):
            values_layout.addWidget(QLabel(label_text), i, 0)
            value_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px;")
            values_layout.addWidget(value_label, i, 1)
        
        values_group.setLayout(values_layout)
        theory_layout.addWidget(values_group)
        
        # Theoretical graph
        self.theory_graph = MplCanvas(self, width=10, height=4)
        theory_layout.addWidget(self.theory_graph)
        
        layout.addWidget(scroll)
        self.theory_tab.setLayout(layout)
    
    def setup_experimental_tab(self):
        layout = QVBoxLayout()
        
        # Experimental values
        exp_group = QGroupBox("Experimental Results")
        exp_layout = QGridLayout()
        
        self.exp_L = QLabel("0.00")
        self.exp_Lq = QLabel("0.00")
        self.exp_W = QLabel("0.00")
        self.exp_Wq = QLabel("0.00")
        self.exp_error = QLabel("0.00%")
        
        metrics = [
            ("L (Experimental):", self.exp_L),
            ("Lq (Experimental):", self.exp_Lq),
            ("W (Experimental):", self.exp_W),
            ("Wq (Experimental):", self.exp_Wq),
            ("Error vs Theory:", self.exp_error)
        ]
        
        for i, (label_text, value_label) in enumerate(metrics):
            exp_layout.addWidget(QLabel(label_text), i, 0)
            value_label.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 14px;")
            exp_layout.addWidget(value_label, i, 1)
        
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)
        
        # Histogram of queue lengths
        self.histogram = MplCanvas(self, width=10, height=5)
        layout.addWidget(self.histogram)
        
        self.experimental_tab.setLayout(layout)
    
    def setup_comparison_tab(self):
        layout = QVBoxLayout()
        
        # Bar chart comparison
        self.comparison_chart = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.comparison_chart)
        
        # Error table
        error_group = QGroupBox("Error Analysis")
        error_layout = QGridLayout()
        
        self.error_L = QLabel("0.00%")
        self.error_Lq = QLabel("0.00%")
        self.error_W = QLabel("0.00%")
        self.error_Wq = QLabel("0.00%")
        
        errors = [
            ("L Error:", self.error_L),
            ("Lq Error:", self.error_Lq),
            ("W Error:", self.error_W),
            ("Wq Error:", self.error_Wq)
        ]
        
        for i, (label_text, error_label) in enumerate(errors):
            error_layout.addWidget(QLabel(label_text), i, 0)
            error_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            error_layout.addWidget(error_label, i, 1)
        
        error_group.setLayout(error_layout)
        layout.addWidget(error_group)
        
        self.comparison_tab.setLayout(layout)
    
    def setup_convergence_tab(self):
        layout = QVBoxLayout()
        
        # Convergence graph
        self.convergence_graph = MplCanvas(self, width=10, height=5)
        layout.addWidget(self.convergence_graph)
        
        # Convergence controls
        controls = QHBoxLayout()
        self.convergence_button = QPushButton("Run Convergence Analysis")
        self.convergence_button.clicked.connect(self.run_convergence_analysis)
        self.convergence_button.setStyleSheet("background-color: #9C27B0; color: white;")
        controls.addWidget(self.convergence_button)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        self.convergence_tab.setLayout(layout)
    
    def run_simulation(self):
        # Get parameters
        lambda_rate = self.lambda_input.value()
        mu_rate = self.mu_input.value()
        max_customers = self.customers_input.value()
        
        # Update simulation
        self.simulation.lambda_rate = lambda_rate
        self.simulation.mu_rate = mu_rate
        self.simulation.max_customers = max_customers
        
        # Update rho display
        rho = lambda_rate / mu_rate
        self.rho_label.setText(f"{rho:.2f}")
        
        # Check stability
        if rho >= 1:
            self.stability_label.setText("UNSTABLE!")
            self.stability_label.setStyleSheet("font-weight: bold; color: #f44336;")
            QMessageBox.warning(self, "Stability Warning", 
                              f"System is unstable (ρ = {rho:.2f} ≥ 1).\nPlease ensure λ < μ.")
            return
        else:
            self.stability_label.setText("Stable")
            self.stability_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        
        # Run simulation
        metrics = self.simulation.run_simulation()
        
        # Update displays
        self.update_displays(metrics)
        self.statusBar().showMessage(f"Simulation completed with {max_customers} customers")
    
    def update_displays(self, metrics):
        exp = metrics['experimental']
        theory = metrics['theoretical']
        
        # Tab 1: Queue Evolution
        time_history, queue_history = exp['queue_history']
        self.queue_graph.update_data(
            time_history[:1000],  # Limit for performance
            queue_history[:1000],
            "Évolution de la longueur de la file d'attente (M/M/1)",
            "Temps",
            "Nombre de clients dans la file"
        )
        self.current_queue_label.setText(str(queue_history[-1] if queue_history else 0))
        self.avg_queue_label.setText(f"{exp['L']:.2f}")
        
        # Tab 2: Theoretical values
        self.theory_L.setText(f"{theory['L']:.3f}")
        self.theory_Lq.setText(f"{theory['Lq']:.3f}")
        self.theory_W.setText(f"{theory['W']:.3f}")
        self.theory_Wq.setText(f"{theory['Wq']:.3f}")
        
        # Plot theoretical distribution
        self.plot_theoretical_distribution(theory['rho'])
        
        # Tab 3: Experimental values
        self.exp_L.setText(f"{exp['L']:.3f}")
        self.exp_Lq.setText(f"{exp['Lq']:.3f}")
        self.exp_W.setText(f"{exp['W']:.3f}")
        self.exp_Wq.setText(f"{exp['Wq']:.3f}")
        
        # Calculate error
        error_L = abs(exp['L'] - theory['L']) / theory['L'] * 100 if theory['L'] > 0 else 0
        self.exp_error.setText(f"{error_L:.2f}%")
        
        # Plot histogram
        self.plot_queue_histogram(exp['queue_history'][1])
        
        # Tab 4: Comparison
        self.plot_comparison(exp, theory)
        
        # Update error labels
        self.error_L.setText(f"{error_L:.2f}%")
        error_Lq = abs(exp['Lq'] - theory['Lq']) / theory['Lq'] * 100 if theory['Lq'] > 0 else 0
        self.error_Lq.setText(f"{error_Lq:.2f}%")
        error_W = abs(exp['W'] - theory['W']) / theory['W'] * 100 if theory['W'] > 0 else 0
        self.error_W.setText(f"{error_W:.2f}%")
        error_Wq = abs(exp['Wq'] - theory['Wq']) / theory['Wq'] * 100 if theory['Wq'] > 0 else 0
        self.error_Wq.setText(f"{error_Wq:.2f}%")
    
    def plot_theoretical_distribution(self, rho):
        self.theory_graph.axes.clear()
        
        # Theoretical probability distribution
        k = np.arange(0, 20)
        prob = [(1 - rho) * (rho**x) for x in k]
        
        bars = self.theory_graph.axes.bar(k, prob, alpha=0.7, color='skyblue', edgecolor='navy')
        self.theory_graph.axes.plot(k, prob, 'ro-', alpha=0.6)
        
        self.theory_graph.axes.set_title('Theoretical Queue Length Distribution', fontweight='bold')
        self.theory_graph.axes.set_xlabel('Number of Customers in System')
        self.theory_graph.axes.set_ylabel('Probability')
        self.theory_graph.axes.grid(True, alpha=0.3)
        
        # Annotate the most probable state
        most_probable = np.argmax(prob)
        self.theory_graph.axes.annotate(f'P(k={most_probable}) = {prob[most_probable]:.3f}',
                                      xy=(most_probable, prob[most_probable]),
                                      xytext=(most_probable + 2, prob[most_probable] + 0.05),
                                      arrowprops=dict(arrowstyle='->', color='red'),
                                      fontweight='bold')
        
        self.theory_graph.axes.set_facecolor('#ffffff')
        self.theory_graph.fig.tight_layout()
        self.theory_graph.draw()
    
    def plot_queue_histogram(self, queue_history):
        self.histogram.axes.clear()
        
        # Create histogram
        n, bins, patches = self.histogram.axes.hist(queue_history, bins=30, alpha=0.7, 
                                                   color='lightgreen', edgecolor='darkgreen', 
                                                   density=True)
        
        # Add theoretical curve
        rho = self.simulation.lambda_rate / self.simulation.mu_rate
        x = np.linspace(0, max(queue_history) if queue_history else 10, 100)
        theoretical_density = [(1 - rho) * (rho**int(xi)) for xi in x]
        self.histogram.axes.plot(x, theoretical_density, 'r-', linewidth=2, 
                                label='Theoretical Distribution')
        
        self.histogram.axes.set_title('Experimental Queue Length Distribution', fontweight='bold')
        self.histogram.axes.set_xlabel('Queue Length')
        self.histogram.axes.set_ylabel('Probability Density')
        self.histogram.axes.legend()
        self.histogram.axes.grid(True, alpha=0.3)
        
        self.histogram.axes.set_facecolor('#ffffff')
        self.histogram.fig.tight_layout()
        self.histogram.draw()
    
    def plot_comparison(self, exp, theory):
        self.comparison_chart.axes.clear()
        
        metrics = ['L (System)', 'Lq (Queue)', 'W (Time)', 'Wq (Wait)']
        x = np.arange(len(metrics))
        width = 0.35
        
        theoretical_values = [theory['L'], theory['Lq'], theory['W'], theory['Wq']]
        experimental_values = [exp['L'], exp['Lq'], exp['W'], exp['Wq']]
        
        bars1 = self.comparison_chart.axes.bar(x - width/2, theoretical_values, width, 
                                              label='Theoretical', color='skyblue', alpha=0.8)
        bars2 = self.comparison_chart.axes.bar(x + width/2, experimental_values, width, 
                                              label='Experimental', color='lightgreen', alpha=0.8)
        
        self.comparison_chart.axes.set_title('Theoretical vs Experimental Results', 
                                           fontweight='bold', fontsize=14)
        self.comparison_chart.axes.set_xlabel('Metrics')
        self.comparison_chart.axes.set_ylabel('Values')
        self.comparison_chart.axes.set_xticks(x)
        self.comparison_chart.axes.set_xticklabels(metrics)
        self.comparison_chart.axes.legend()
        self.comparison_chart.axes.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                self.comparison_chart.axes.annotate(f'{height:.2f}',
                                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                                  xytext=(0, 3),
                                                  textcoords="offset points",
                                                  ha='center', va='bottom', fontsize=9)
        
        self.comparison_chart.axes.set_facecolor('#ffffff')
        self.comparison_chart.fig.tight_layout()
        self.comparison_chart.draw()
    
    def run_convergence_analysis(self):
        self.convergence_graph.axes.clear()
        
        lambda_rate = self.lambda_input.value()
        mu_rate = self.mu_input.value()
        rho = lambda_rate / mu_rate
        
        # Theoretical value
        L_theory = rho / (1 - rho)
        
        # Run simulations with increasing number of customers
        customer_counts = np.logspace(2, 5, 20, dtype=int)
        experimental_L = []
        
        self.convergence_button.setEnabled(False)
        QApplication.processEvents()
        
        for i, n in enumerate(customer_counts):
            sim = MM1QueueSimulation(lambda_rate, mu_rate, n)
            metrics = sim.run_simulation()
            experimental_L.append(metrics['experimental']['L'])
            
            # Update progress in status bar
            progress = (i + 1) / len(customer_counts) * 100
            self.statusBar().showMessage(f"Running convergence analysis... {progress:.1f}%")
            QApplication.processEvents()
        
        # Plot convergence
        self.convergence_graph.axes.semilogx(customer_counts, experimental_L, 'b-o', 
                                           linewidth=2, markersize=6, label='Experimental L')
        self.convergence_graph.axes.axhline(y=L_theory, color='r', linestyle='--', 
                                          linewidth=2, label='Theoretical L')
        
        self.convergence_graph.axes.fill_between(customer_counts, 
                                               np.array(experimental_L) * 0.95,
                                               np.array(experimental_L) * 1.05,
                                               alpha=0.2, color='blue')
        
        self.convergence_graph.axes.set_title('Convergence of Experimental to Theoretical Values', 
                                            fontweight='bold')
        self.convergence_graph.axes.set_xlabel('Number of Customers (log scale)')
        self.convergence_graph.axes.set_ylabel('Average Customers in System (L)')
        self.convergence_graph.axes.legend()
        self.convergence_graph.axes.grid(True, alpha=0.3, which='both')
        
        # Add convergence annotation
        final_error = abs(experimental_L[-1] - L_theory) / L_theory * 100
        self.convergence_graph.axes.annotate(f'Final error: {final_error:.2f}%',
                                           xy=(customer_counts[-1], experimental_L[-1]),
                                           xytext=(customer_counts[-1]/2, experimental_L[-1]),
                                           arrowprops=dict(arrowstyle='->', color='green'),
                                           fontweight='bold')
        
        self.convergence_graph.axes.set_facecolor('#ffffff')
        self.convergence_graph.fig.tight_layout()
        self.convergence_graph.draw()
        
        self.convergence_button.setEnabled(True)
        self.statusBar().showMessage(f"Convergence analysis completed. Final error: {final_error:.2f}%")
    
    def apply_styles(self):
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QLabel {
                padding: 2px;
            }
        """)

def main():
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Set application palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = QueueSimulationApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()   