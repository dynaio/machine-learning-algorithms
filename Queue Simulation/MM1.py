import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import stats

class MM1QueueSimulation:
    """
    M/M/1 Queue Simulation Core Logic
    """
    def __init__(self, arrival_rate=0.5, service_rate=1.0, num_customers=100, seed=42):
        self.lambda_rate = arrival_rate
        self.mu_rate = service_rate
        self.rho = arrival_rate / service_rate
        self.num_customers = num_customers
        self.seed = seed
        
        # Initialize arrays
        self.arrival_times = None
        self.service_times = None
        self.departure_times = None
        self.time_in_system = None
        self.time_in_queue = None
        self.time_points = None
        self.queue_lengths = None
        
    def calculate_theoretical_values(self):
        """Calculate theoretical M/M/1 queue metrics"""
        rho = self.rho
        
        L = rho / (1 - rho)
        Lq = rho * rho / (1 - rho)
        W = 1 / (self.mu_rate - self.lambda_rate)
        Wq = rho / (self.mu_rate - self.lambda_rate)
        Ws = 1 / self.mu_rate
        
        return {
            'L': L,
            'Lq': Lq,
            'W': W,
            'Wq': Wq,
            'Ws': Ws,
            'rho': rho
        }
    
    def generate_interarrival_times(self):
        """Generate exponential interarrival times"""
        np.random.seed(self.seed)
        interarrival_times = np.random.exponential(scale=1/self.lambda_rate, 
                                                  size=self.num_customers)
        arrival_times = np.cumsum(interarrival_times)
        return arrival_times
    
    def generate_service_times(self):
        """Generate exponential service times"""
        np.random.seed(self.seed + 1)
        service_times = np.random.exponential(scale=1/self.mu_rate, 
                                             size=self.num_customers)
        return service_times
    
    def simulate_queue(self):
        """Run the M/M/1 queue simulation"""
        # Generate times
        arrival_times = self.generate_interarrival_times()
        service_times = self.generate_service_times()
        
        # Initialize arrays
        departure_times = np.zeros(self.num_customers)
        start_service_times = np.zeros(self.num_customers)
        
        # Process first customer
        departure_times[0] = arrival_times[0] + service_times[0]
        start_service_times[0] = arrival_times[0]
        
        # Process remaining customers
        for i in range(1, self.num_customers):
            start_service_times[i] = max(arrival_times[i], departure_times[i-1])
            departure_times[i] = start_service_times[i] + service_times[i]
        
        # Calculate times
        time_in_system = departure_times - arrival_times
        time_in_queue = start_service_times - arrival_times
        
        # Store results
        self.arrival_times = arrival_times
        self.service_times = service_times
        self.departure_times = departure_times
        self.start_service_times = start_service_times
        self.time_in_system = time_in_system
        self.time_in_queue = time_in_queue
        
        # Generate queue length time series
        self.generate_queue_length_timeseries()
        
        return True
    
    def generate_queue_length_timeseries(self, max_time=None):
        """Create time series of queue length"""
        if max_time is None:
            max_time = self.departure_times[-1] + 1
        
        # Combine events
        events = []
        for i in range(self.num_customers):
            events.append((self.arrival_times[i], 'arrival'))
            events.append((self.departure_times[i], 'departure'))
        
        # Sort events
        events.sort(key=lambda x: x[0])
        
        # Generate time series
        time_points = [0]
        queue_lengths = [0]
        current_length = 0
        
        for time, event_type in events:
            if time > max_time:
                break
            
            if event_type == 'arrival':
                current_length += 1
            else:
                current_length -= 1
                if current_length < 0:
                    current_length = 0
            
            time_points.append(time)
            queue_lengths.append(current_length)
        
        self.time_points = np.array(time_points)
        self.queue_lengths = np.array(queue_lengths)
        
        return self.time_points, self.queue_lengths
    
    def calculate_simulation_values(self):
        """Calculate metrics from simulation data"""
        # Time averages
        W_sim = np.mean(self.time_in_system)
        Wq_sim = np.mean(self.time_in_queue)
        Ws_sim = np.mean(self.service_times)
        
        # Time-weighted averages for queue lengths
        total_time = self.time_points[-1]
        
        # Average number in system
        weighted_sum = 0
        for i in range(len(self.time_points) - 1):
            dt = self.time_points[i+1] - self.time_points[i]
            weighted_sum += self.queue_lengths[i] * dt
        L_sim = weighted_sum / total_time
        
        # Average number in queue (excluding customer being served)
        queue_only_lengths = np.maximum(self.queue_lengths - 1, 0)
        weighted_sum_q = 0
        for i in range(len(self.time_points) - 1):
            dt = self.time_points[i+1] - self.time_points[i]
            weighted_sum_q += queue_only_lengths[i] * dt
        Lq_sim = weighted_sum_q / total_time
        
        # Little's Law verification
        L_from_little = self.lambda_rate * W_sim
        Lq_from_little = self.lambda_rate * Wq_sim
        
        return {
            'L_sim': L_sim,
            'Lq_sim': Lq_sim,
            'W_sim': W_sim,
            'Wq_sim': Wq_sim,
            'Ws_sim': Ws_sim,
            'L_from_little': L_from_little,
            'Lq_from_little': Lq_from_little
        }

class PlotCanvas(FigureCanvas):
    """Custom Matplotlib canvas for embedding in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def clear_plot(self):
        self.fig.clear()

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.simulation = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('M/M/1 Queue Simulation')
        self.setGeometry(100, 100, 1600, 1200)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Create tab widget for plots
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create result display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        main_layout.addWidget(self.results_text)
        
        # Initialize simulation
        self.run_simulation()
        
    def create_control_panel(self):
        """Create control panel with parameters"""
        panel = QGroupBox("Simulation Parameters")
        layout = QGridLayout()
        
        # Lambda parameter
        layout.addWidget(QLabel("Arrival Rate (λ):"), 0, 0)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.1, 10.0)
        self.lambda_spin.setValue(0.5)
        self.lambda_spin.setSingleStep(0.1)
        layout.addWidget(self.lambda_spin, 0, 1)
        
        # Mu parameter
        layout.addWidget(QLabel("Service Rate (μ):"), 1, 0)
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0.1, 10.0)
        self.mu_spin.setValue(1.0)
        self.mu_spin.setSingleStep(0.1)
        layout.addWidget(self.mu_spin, 1, 1)
        
        # Number of customers
        layout.addWidget(QLabel("Number of Customers:"), 2, 0)
        self.num_customers_spin = QSpinBox()
        self.num_customers_spin.setRange(10, 1000)
        self.num_customers_spin.setValue(100)
        layout.addWidget(self.num_customers_spin, 2, 1)
        
        # Seed
        layout.addWidget(QLabel("Random Seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 1000)
        self.seed_spin.setValue(42)
        layout.addWidget(self.seed_spin, 3, 1)
        
        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button, 4, 0, 1, 2)
        
        panel.setLayout(layout)
        return panel
    
    def run_simulation(self):
        """Run simulation with current parameters"""
        try:
            # Get parameters
            lambda_val = self.lambda_spin.value()
            mu_val = self.mu_spin.value()
            num_customers = self.num_customers_spin.value()
            seed = self.seed_spin.value()
            
            # Check stability condition
            if lambda_val >= mu_val:
                QMessageBox.warning(self, "Warning", 
                                  "System is unstable: λ must be < μ\n" +
                                  f"Current: λ={lambda_val}, μ={mu_val}")
                return
            
            # Create and run simulation
            self.simulation = MM1QueueSimulation(
                arrival_rate=lambda_val,
                service_rate=mu_val,
                num_customers=num_customers,
                seed=seed
            )
            self.simulation.simulate_queue()
            
            # Calculate values
            theoretical = self.simulation.calculate_theoretical_values()
            simulated = self.simulation.calculate_simulation_values()
            
            # Update plots
            self.update_plots(theoretical, simulated)
            
            # Update results text
            self.update_results_text(theoretical, simulated)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed: {str(e)}")
    
    def update_plots(self, theoretical, simulated):
        """Update all plots"""
        # Clear existing tabs
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)
        
        # Create plots
        self.create_queue_length_plot()
        self.create_time_distribution_plots()
        self.create_comparison_plot(theoretical, simulated)
        self.create_littles_law_plot(simulated)
        
    def create_queue_length_plot(self):
        """Create queue length evolution plot"""
        canvas = PlotCanvas(self, width=8, height=5)
        ax = canvas.fig.add_subplot(111)
        
        ax.step(self.simulation.time_points, self.simulation.queue_lengths, 
                where='post', linewidth=1.5, color='blue')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Number of Customers in Queue', fontsize=12)
        ax.set_title(f'Queue Length Evolution (λ={self.simulation.lambda_rate}, ' +
                    f'μ={self.simulation.mu_rate}, ρ={self.simulation.rho:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        canvas.fig.tight_layout()
        self.tab_widget.addTab(canvas, "Queue Length")
    
    def create_time_distribution_plots(self):
        """Create plots for time distributions"""
        canvas = PlotCanvas(self, width=10, height=8)
        
        # Create subplots
        ax1 = canvas.fig.add_subplot(221)
        ax2 = canvas.fig.add_subplot(222)
        ax3 = canvas.fig.add_subplot(223)
        ax4 = canvas.fig.add_subplot(224)
        
        # Plot 1: Interarrival times
        interarrival_times = np.diff(self.simulation.arrival_times)
        ax1.hist(interarrival_times, bins=30, density=True, alpha=0.7, 
                color='blue', label='Simulated')
        x = np.linspace(0, np.max(interarrival_times), 100)
        pdf = stats.expon.pdf(x, scale=1/self.simulation.lambda_rate)
        ax1.plot(x, pdf, 'r-', linewidth=2, label='Theoretical')
        ax1.set_xlabel('Interarrival Time')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Interarrival Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Service times
        ax2.hist(self.simulation.service_times, bins=30, density=True, 
                alpha=0.7, color='blue', label='Simulated')
        x = np.linspace(0, np.max(self.simulation.service_times), 100)
        pdf = stats.expon.pdf(x, scale=1/self.simulation.mu_rate)
        ax2.plot(x, pdf, 'r-', linewidth=2, label='Theoretical')
        ax2.set_xlabel('Service Time')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Service Time Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time in system
        ax3.hist(self.simulation.time_in_system, bins=30, density=True, 
                alpha=0.7, color='blue')
        ax3.set_xlabel('Time in System')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Time in System Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time in queue
        ax4.hist(self.simulation.time_in_queue, bins=30, density=True, 
                alpha=0.7, color='blue')
        ax4.set_xlabel('Time in Queue')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Time in Queue Distribution')
        ax4.grid(True, alpha=0.3)
        
        canvas.fig.suptitle('Time Distributions in M/M/1 Queue', 
                          fontsize=16, fontweight='bold')
        canvas.fig.tight_layout()
        self.tab_widget.addTab(canvas, "Time Distributions")
    
    def create_comparison_plot(self, theoretical, simulated):
        """Create comparison plot between theoretical and simulated values"""
        canvas = PlotCanvas(self, width=8, height=6)
        ax = canvas.fig.add_subplot(111)
        
        categories = ['L (System)', 'Lq (Queue)', 'W (System)', 'Wq (Queue)', 'Ws (Service)']
        
        theoretical_vals = [
            theoretical['L'],
            theoretical['Lq'],
            theoretical['W'],
            theoretical['Wq'],
            theoretical['Ws']
        ]
        
        simulated_vals = [
            simulated['L_sim'],
            simulated['Lq_sim'],
            simulated['W_sim'],
            simulated['Wq_sim'],
            simulated['Ws_sim']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, theoretical_vals, width, 
                      label='Theoretical', color='red', alpha=0.8)
        bars2 = ax.bar(x + width/2, simulated_vals, width, 
                      label='Simulated', color='blue', alpha=0.8)
        
        ax.set_xlabel('Performance Measures', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Theoretical vs Simulated Performance Measures', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (th, sim) in enumerate(zip(theoretical_vals, simulated_vals)):
            ax.text(i - width/2, th, f'{th:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, sim, f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
        
        canvas.fig.tight_layout()
        self.tab_widget.addTab(canvas, "Comparison")
    
    def create_littles_law_plot(self, simulated):
        """Create plot showing Little's Law verification"""
        canvas = PlotCanvas(self, width=8, height=6)
        ax = canvas.fig.add_subplot(111)
        
        # Prepare data
        categories = ['L = λW', 'Lq = λWq']
        from_simulation = [simulated['L_sim'], simulated['Lq_sim']]
        from_little = [simulated['L_from_little'], simulated['Lq_from_little']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, from_simulation, width, 
                      label='Direct from Simulation', color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, from_little, width, 
                      label='From Little\'s Law', color='green', alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Verification of Little\'s Law', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (sim, little) in enumerate(zip(from_simulation, from_little)):
            ax.text(i - width/2, sim, f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, little, f'{little:.3f}', ha='center', va='bottom', fontsize=9)
        
        canvas.fig.tight_layout()
        self.tab_widget.addTab(canvas, "Little's Law")
    
    def update_results_text(self, theoretical, simulated):
        """Update the results text display"""
        text = "=" * 70 + "\n"
        text += "M/M/1 QUEUE SIMULATION RESULTS\n"
        text += "=" * 70 + "\n\n"
        
        # Parameters
        text += f"PARAMETERS:\n"
        text += f"Arrival Rate (λ) = {self.simulation.lambda_rate}\n"
        text += f"Service Rate (μ) = {self.simulation.mu_rate}\n"
        text += f"Utilization (ρ = λ/μ) = {self.simulation.rho:.4f}\n"
        text += f"Number of Customers = {self.simulation.num_customers}\n"
        text += f"Random Seed = {self.simulation.seed}\n\n"
        
        text += "-" * 70 + "\n"
        text += "THEORETICAL VALUES (M/M/1):\n"
        text += "-" * 70 + "\n"
        text += f"L  (Average number in system)  = {theoretical['L']:.4f}\n"
        text += f"Lq (Average number in queue)   = {theoretical['Lq']:.4f}\n"
        text += f"W  (Average time in system)    = {theoretical['W']:.4f}\n"
        text += f"Wq (Average time in queue)     = {theoretical['Wq']:.4f}\n"
        text += f"Ws (Average service time)      = {theoretical['Ws']:.4f}\n\n"
        
        text += "-" * 70 + "\n"
        text += "SIMULATION VALUES:\n"
        text += "-" * 70 + "\n"
        text += f"L  (Average number in system)  = {simulated['L_sim']:.4f}\n"
        text += f"Lq (Average number in queue)   = {simulated['Lq_sim']:.4f}\n"
        text += f"W  (Average time in system)    = {simulated['W_sim']:.4f}\n"
        text += f"Wq (Average time in queue)     = {simulated['Wq_sim']:.4f}\n"
        text += f"Ws (Average service time)      = {simulated['Ws_sim']:.4f}\n\n"
        
        text += "-" * 70 + "\n"
        text += "LITTLE'S LAW VERIFICATION:\n"
        text += "-" * 70 + "\n"
        text += f"L from simulation  = {simulated['L_sim']:.4f}\n"
        text += f"λW from Little's Law = {simulated['L_from_little']:.4f}\n"
        text += f"Difference = {abs(simulated['L_sim'] - simulated['L_from_little']):.4f}\n"
        text += f"Relative Error = {abs((simulated['L_sim'] - simulated['L_from_little'])/simulated['L_sim']*100):.2f}%\n\n"
        
        text += f"Lq from simulation = {simulated['Lq_sim']:.4f}\n"
        text += f"λWq from Little's Law = {simulated['Lq_from_little']:.4f}\n"
        text += f"Difference = {abs(simulated['Lq_sim'] - simulated['Lq_from_little']):.4f}\n"
        text += f"Relative Error = {abs((simulated['Lq_sim'] - simulated['Lq_from_little'])/simulated['Lq_sim']*100):.2f}%\n\n"
        
        text += "-" * 70 + "\n"
        text += "THEORETICAL vs SIMULATION COMPARISON:\n"
        text += "-" * 70 + "\n"
        
        # Calculate relative errors
        errors = {
            'L': abs((theoretical['L'] - simulated['L_sim']) / theoretical['L'] * 100),
            'Lq': abs((theoretical['Lq'] - simulated['Lq_sim']) / theoretical['Lq'] * 100),
            'W': abs((theoretical['W'] - simulated['W_sim']) / theoretical['W'] * 100),
            'Wq': abs((theoretical['Wq'] - simulated['Wq_sim']) / theoretical['Wq'] * 100),
            'Ws': abs((theoretical['Ws'] - simulated['Ws_sim']) / theoretical['Ws'] * 100)
        }
        
        text += f"L  (System length):  Theoretical={theoretical['L']:.4f}, " \
                f"Simulated={simulated['L_sim']:.4f}, " \
                f"Error={errors['L']:.2f}%\n"
        text += f"Lq (Queue length):   Theoretical={theoretical['Lq']:.4f}, " \
                f"Simulated={simulated['Lq_sim']:.4f}, " \
                f"Error={errors['Lq']:.2f}%\n"
        text += f"W  (System time):    Theoretical={theoretical['W']:.4f}, " \
                f"Simulated={simulated['W_sim']:.4f}, " \
                f"Error={errors['W']:.2f}%\n"
        text += f"Wq (Queue time):     Theoretical={theoretical['Wq']:.4f}, " \
                f"Simulated={simulated['Wq_sim']:.4f}, " \
                f"Error={errors['Wq']:.2f}%\n"
        text += f"Ws (Service time):   Theoretical={theoretical['Ws']:.4f}, " \
                f"Simulated={simulated['Ws_sim']:.4f}, " \
                f"Error={errors['Ws']:.2f}%\n\n"
        
        text += "=" * 70 + "\n"
        
        self.results_text.setText(text)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set application font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()