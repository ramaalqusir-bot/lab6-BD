import numpy as np
import time
import sys

print("=" * 60)
print("Task 1.1: Memory Profiling")
print("=" * 60)

for rows in [10000, 50000, 100000, 500000, 1000000]:
    start = time.time()
    X = np.random.randn(rows, 50).astype(np.float32)
    t = time.time() - start
    print(f"{rows:8} rows | Memory: {sys.getsizeof(X):10} bytes | Time: {t:.4f}s")

print("\nAnswer: sys.getsizeof() includes Python overhead, X.nbytes is raw data")
print("5M rows x 50 x 8 bytes = 1907 MB for float64")
