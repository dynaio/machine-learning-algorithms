from math import exp


state = {
    'w11': 0.1, 'w12': 0.2, 'w13': 0.3,
    'w21': 0.4, 'w22': 0.5, 'w23': 0.6,
    'w31': 0.1, 'w32': 0.4,
    'w41': 0.2, 'w42': 0.5,
    'w51': 0.3, 'w52': 0.6,
    'bh1': 0.7, 'bh2': 0.8, 'bh3': 0.9,
    'bo1': 0.1, 'bo2': 0.2,
    
    'step': 0,
    'y1_pred': 0.0,
    'y2_pred': 0.0,
    'err1': 0.0,
    'err2': 0.0,
    'loss': 999.0,          
}

def activation(z):
    return 1 / (1 + exp(-z))  # Sigmoid activation function woks well for this problem
    #return max(0, z)           # ReLU activation function caused problems OverflowError: (34, 'Numerical result out of range')!!
    #return z                  # Linear activation function causes brblmes everything become nan after a few steps, because the gradients become too large and cause overflow in the exponential function. We need to use a non-linear activation function to keep the gradients in check and allow the network to learn effectively.

def forward(x1, x2):
    z1 = x1 * state['w11'] + x2 * state['w21'] + state['bh1']
    z2 = x1 * state['w12'] + x2 * state['w22'] + state['bh2']
    z3 = x1 * state['w13'] + x2 * state['w23'] + state['bh3']
    
    h1 = activation(z1)
    h2 = activation(z2)
    h3 = activation(z3)
    
    zo1 = h1 * state['w31'] + h2 * state['w41'] + h3 * state['w51'] + state['bo1']
    zo2 = h1 * state['w32'] + h2 * state['w42'] + h3 * state['w52'] + state['bo2']
    
    y1_hat = activation(zo1)
    y2_hat = activation(zo2)
    
    cache = {
        'x1': x1, 'x2': x2,
        'z1': z1, 'z2': z2, 'z3': z3,
        'h1': h1, 'h2': h2, 'h3': h3,
        'zo1': zo1, 'zo2': zo2,
        'y1_hat': y1_hat, 'y2_hat': y2_hat
    }
    return (y1_hat, y2_hat), cache

def mse_loss(y1_true, y2_true, y1_pred, y2_pred):
    e1 = y1_true - y1_pred
    e2 = y2_true - y2_pred
    return (e1**2 + e2**2) / 2

def compute_gradients(y1_true, y2_true, cache):
    y1_hat = cache['y1_hat']
    y2_hat = cache['y2_hat']
    h1, h2, h3 = cache['h1'], cache['h2'], cache['h3']
    
    dy1 = (y1_hat - y1_true) * y1_hat * (1 - y1_hat)
    dy2 = (y2_hat - y2_true) * y2_hat * (1 - y2_hat)
    
    grads = {}
    
    grads['bo1'] = dy1
    grads['bo2'] = dy2
    
    grads['w31'] = dy1 * h1
    grads['w32'] = dy2 * h1
    grads['w41'] = dy1 * h2
    grads['w42'] = dy2 * h2
    grads['w51'] = dy1 * h3
    grads['w52'] = dy2 * h3
    
    dh1 = dy1 * state['w31'] + dy2 * state['w32']
    dh2 = dy1 * state['w41'] + dy2 * state['w42']
    dh3 = dy1 * state['w51'] + dy2 * state['w52']
    
    dz1 = dh1 * h1 * (1 - h1)
    dz2 = dh2 * h2 * (1 - h2)
    dz3 = dh3 * h3 * (1 - h3)
    
    grads['bh1'] = dz1
    grads['bh2'] = dz2
    grads['bh3'] = dz3
    
    grads['w11'] = dz1 * cache['x1']
    grads['w21'] = dz1 * cache['x2']
    grads['w12'] = dz2 * cache['x1']
    grads['w22'] = dz2 * cache['x2']
    grads['w13'] = dz3 * cache['x1']
    grads['w23'] = dz3 * cache['x2']
    
    return grads


def update_state(grads, lr=0.1, step=0, y1_pred=0.0, y2_pred=0.0, y1_true=0.0, y2_true=0.0):
    for key in state:
        if key in grads:
            state[key] -= lr * grads[key]
    
    state['step']    = step
    state['y1_pred'] = y1_pred
    state['y2_pred'] = y2_pred
    state['err1']    = y1_true - y1_pred
    state['err2']    = y2_true - y2_pred
    state['loss']    = mse_loss(y1_true, y2_true, y1_pred, y2_pred)


def print_current_state():
    print(f"Step {state['step']:3d} | loss = {state['loss']:.6f}")
    print(f"   y1_pred = {state['y1_pred']:.4f}   (target 0.7)   err1 = {state['err1']:+.4f}")
    print(f"   y2_pred = {state['y2_pred']:.4f}   (target 0.4)   err2 = {state['err2']:+.4f}")
    print()
    
    print("Selected weights:")
    for k in ['w11','w12','w13','w21','w22','w23','bh1','bh2','bh3','w31','w32','bo1','bo2']:
        print(f"  {k:>5} = {state[k]:.4f}")
    print("-" * 60)

x1, x2       = 0.5, 0.8
y1_true      = 0.7
y2_true      = 0.4
lr           = 0.1
n_steps      = 30

print("Initial state:")
print_current_state()

for step in range(1, n_steps + 1):
    (y1p, y2p), cache = forward(x1, x2)
    grads = compute_gradients(y1_true, y2_true, cache)
    
    update_state(grads, lr, step, y1p, y2p, y1_true, y2_true)
    
    if step % 5 == 0 or step == 1 or step == n_steps:
        print_current_state()

print_current_state()