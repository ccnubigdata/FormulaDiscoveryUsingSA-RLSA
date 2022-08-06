import numpy as np
from copy import deepcopy
from sympy import *
import random
import math
f = np.load('result.npy', allow_pickle=True)
f = sorted(f, key=lambda x: len(x))
print(len(f))
nf = []
A, B = symbols('A B')
pi = math.acos(-1)
import re
for formula in f:
    idx = formula.find('=')
    lf = formula[:idx]
    rf = formula[idx+1:]
    if lf == rf:
        continue
    lf = expand(lf)
    rf = expand(rf)
    key1 = lf/rf
    s = str(key1)
    ch = key1
    key = re.findall(r"[(](.*?)[)]", s)
    key = list(set(key))

    if len(key) == 1:
        ch = ch.subs({simplify(key[0]): A})
    else:
        alist = []
        blist = []
        for k in key:
            if 'A' in k and 'B' not in k:
                alist.append(k)
            elif 'B' in k and 'A' not in k:
                blist.append(k)
        if len(alist) + len(blist) == len(key):
            if len(alist) == 1:
                ch = ch.subs({simplify(alist[0]): A})
            elif len(alist) == 0:
                ch = ch.subs({B: A})
            if len(blist) == 1:
                ch = ch.subs({simplify(blist[0]): B})
    key1 = ch
    key2 = 1/key1
    if key1 in nf or key2 in nf:
        continue
    nf.append(key1)
    # print(formula, key1)

f = deepcopy(nf)
print(len(f))

nf = set()
vis = list()
f = sorted(f, key=lambda x: len(str(x)))

for formula in f:
    s = str(formula)
    s = str(expand(s))
    idx = s.find('/')
    if idx != -1:
        if s[idx+1] == '(':
            ff = expand(s[:idx]) + -1*expand(s[idx+2:-1])
        else:
            ff = expand(s[:idx]) + -1*expand(s[idx+1:])
    else:
        ff = expand(s)-1
    if ff not in vis and -1*ff not in vis:
        nf.add(str(ff)+' = 0')
        vis.append(ff)

f = list(nf)
print(len(f))
np.save('simplify_result.npy', f)


