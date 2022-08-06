import numpy as np

f = np.load('angle.npy', allow_pickle=True)

a = [1716, 576, 515, 563, 1307, 522]
for idx in a:
    print(f[idx])
