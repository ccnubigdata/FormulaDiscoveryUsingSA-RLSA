import numpy as np
from sympy import *
A, B, C = symbols('A B C')
Area = symbols('Area')
f = np.load('vary2_side_angle.npy', allow_pickle=True)
print(len(f))