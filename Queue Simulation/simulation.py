import math
import random

def mm1_ultra_simple(lambda_rate, mu_rate, num_clients=6):
    """
    Ultra-simple M/M/1 simulation showing basic events
    """
    print("ULTRA SIMPLE M/M/1 SIMULATION")
    print(f"λ={lambda_rate}, μ={mu_rate}, ρ={lambda_rate/mu_rate:.2f}")
    print("=" * 50)
    
    # Initialize
    time = 0.0
    server_busy = False
    queue = []
    next_departure = float('inf')
    
    # Generate arrival times
    arrivals = [0.0]
    for i in range(1, num_clients):
        u = random.random()
        if u == 0: u = 0.0001  # Avoid log(0)
        inter_arrival = -1/lambda_rate * math.log(u)
        arrivals.append(arrivals[-1] + inter_arrival)
    
    print("Arrival times:", [f"{t:.2f}" for t in arrivals])
    print()
    
    client_id = 1
    for arrival_time in arrivals:
        print(f"Client {client_id} ARRIVES at time {arrival_time:.2f}")
        
        # Generate service time for this client
        u = random.random()
        if u == 0: u = 0.0001
        service_time = -1/mu_rate * math.log(u)
        print(f"  Needs service time: {service_time:.2f}")
        
        if not server_busy:
            # Start service immediately
            server_busy = True
            start_time = arrival_time
            departure = arrival_time + service_time
            wait = 0.0
            print(f"  Starts service immediately")
            print(f"  Will depart at time {departure:.2f}")
            next_departure = departure
        else:
            # Go to queue
            queue.append((client_id, arrival_time, service_time))
            print(f"  Joins queue (position {len(queue)})")
        
        # Check if any departures happen before next arrival
        if client_id < len(arrivals) and next_departure < arrivals[client_id]:
            print(f"\nDEPARTURE at time {next_departure:.2f}")
            if queue:
                next_client, arr_time, serv_time = queue.pop(0)
                wait_time = next_departure - arr_time
                print(f"  Client {next_client} starts service (waited {wait_time:.2f})")
                next_departure = next_departure + serv_time
                print(f"  Will depart at {next_departure:.2f}")
            else:
                server_busy = False
                next_departure = float('inf')
                print(f"  Server becomes idle")
        
        client_id += 1
        print()

# Run the ultra simple version
mm1_ultra_simple(lambda_rate=0.5, mu_rate=0.8, num_clients=6)
