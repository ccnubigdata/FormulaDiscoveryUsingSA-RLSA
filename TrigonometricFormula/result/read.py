import numpy as np

f = np.load('simplify_result.npy', allow_pickle=True)
f = sorted(f, key=lambda x: len(x))
for formula in f:
    print(formula)

