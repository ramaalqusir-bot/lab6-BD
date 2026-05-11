import numpy as np

def ring_allreduce(data):
    n = len(data)
    d = len(data[0])
    chunk = d // n
    out = [x.copy() for x in data]
    
    # Reduce-scatter
    for step in range(1, n):
        for i in range(n):
            send = (i + 1) % n
            start = (i * chunk) % d
            out[i][start:start+chunk] += out[send][start:start+chunk]
    
    # All-gather
    for step in range(1, n):
        for i in range(n):
            send = (i + 1) % n
            start = (send * chunk) % d
            out[send][start:start+chunk] = out[i][start:start+chunk]
    return out

data = [np.array([1,2,3,4]), np.array([5,6,7,8]), np.array([9,10,11,12]), np.array([13,14,15,16])]
print("Ring All-Reduce:", ring_allreduce(data)[0])
print("Expected sum:", sum(data))
print("\nNaive messages: 2n, Ring messages: 2(n-1)")
print("Ring is bandwidth-optimal: each worker sends/receives 2d*(n-1)/n bytes")
