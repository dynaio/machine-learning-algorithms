from math import exp

state = {
    'w1': 0.5,  
    'w2': 0.1,  
    'w3': 0.62, 
    'w4': 0.2,  
    'w5': -0.2, 
    'w6': 0.3,  
    
    'b1': 0.4,    
    'b2': -0.1,   
    'b3': 1.83,   
    
    'step': 0,
    'y_pred': 0.0,
    'error': 0.0,   
    'loss': 999.0,
}

def activation(z):
    return 1 / (1 + exp(-z))  # sigmoid activation function woks well for this problem 
    #return max(0, z)           # ReLU activation function caused   b1 = nan    b2 = nan    b3 = inf !!!
    #return z                  # Linear activation function causes   b1 = nan    b2 = nan    b3 = nan !!!

def forward(x1, x2):
    z1 = x1 * state['w1'] + x2 * state['w2'] + state['b1']
    z2 = x1 * state['w3'] + x2 * state['w4'] + state['b2']
    
    h1 = activation(z1)
    h2 = activation(z2)
    
    # Output layer
    z_out = h1 * state['w5'] + h2 * state['w6'] + state['b3']
    y_hat = activation(z_out)
    
    cache = {
        'x1': x1, 'x2': x2,
        'z1': z1, 'z2': z2,
        'h1': h1, 'h2': h2,
        'z_out': z_out,
        'y_hat': y_hat
    }
    return y_hat, cache

def mse_loss(target, pred):
    e = target - pred
    return e * e   

def compute_gradients(target, cache):
    y_hat = cache['y_hat']
    h1 = cache['h1']
    h2 = cache['h2']    
    dy = (y_hat - target) * y_hat * (1 - y_hat)          
    grads = {}
    grads['b3'] = dy
    grads['w5'] = dy * h1
    grads['w6'] = dy * h2

    dh1 = dy * state['w5']
    dh2 = dy * state['w6']
    
    dz1 = dh1 * h1 * (1 - h1)
    dz2 = dh2 * h2 * (1 - h2)
    
    grads['b1'] = dz1
    grads['b2'] = dz2
    
    grads['w1'] = dz1 * cache['x1']
    grads['w2'] = dz1 * cache['x2']
    grads['w3'] = dz2 * cache['x1']
    grads['w4'] = dz2 * cache['x2']
    
    return grads

def update_state(grads, lr=0.1, step=0, y_pred=0.0, target=0.0):
    for key in ['w1','w2','w3','w4','w5','w6','b1','b2','b3']:
        if key in grads:
            state[key] -= lr * grads[key]
    
    state['step']   = step
    state['y_pred'] = y_pred
    state['error']  = target - y_pred
    state['loss']   = mse_loss(target, y_pred)


def print_current_state():
    s = state
    print(f"Step {s['step']:3d} | loss = {s['loss']:.6f}")
    print(f"   y_pred = {s['y_pred']:.4f}    target = 0.03    error = {s['error']:+.4f}")
    print()
    print("Weights & biases:")
    print(f"  w1h1 = {s['w1']:.4f}   w2h1 = {s['w2']:.4f}")
    print(f"  w3h2 = {s['w3']:.4f}   w4h2 = {s['w4']:.4f}")
    print(f"  w5 h1out = {s['w5']:.4f}   w6 h2out = {s['w6']:.4f}")
    print(f"  b1 = {s['b1']:.4f}    b2 = {s['b2']:.4f}    b3 = {s['b3']:.4f}")
    print("-" * 65)

x1 = 0.1
x2 = 0.3
target = 0.03
lr = 0.1
n_steps = 50

print("Initial state:")
print_current_state()

for step in range(1, n_steps + 1):
    y_pred, cache = forward(x1, x2)
    grads = compute_gradients(target, cache)
    update_state(grads, lr, step, y_pred, target)
    
    if step == 1 or step % 10 == 0 or step == n_steps:
        print_current_state()

print_current_state()