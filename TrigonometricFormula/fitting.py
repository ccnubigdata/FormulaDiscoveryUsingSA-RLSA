import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import *
import math

coe_value = np.load('coefficient_value.npy', allow_pickle=True)
coe_label = np.load('coefficient_label.npy', allow_pickle=True)
def find_2(aim, a=coe_value):
    l = 0
    r = len(a)
    while l < r:
        m = (l + r) // 2
        if a[m] > aim:
            r = m
        else:
            l = m + 1
    return (l+r)//2
def deal_result(x, y, dx, dy, leny):
    x_train = np.array(dx).T[:15]
    y_train = np.array(dy).T[:15]
    lineModel = LinearRegression()
    lineModel.fit(x_train, y_train)
    arr, b = lineModel.coef_[0], lineModel.intercept_[0]
    zero = 1e-8
    fr, fl = 0, 0
    fl = y[0]

    for i in range(0, len(arr)-leny):
        ai = arr[i]
        if abs(ai) < zero:
            continue
        idx = find_2(abs(ai), coe_value) - 1
        if idx + 1 < len(coe_value) and abs(coe_value[idx] - abs(ai)) > abs(coe_value[idx + 1] - abs(ai)):
            idx += 1
        if abs(coe_value[idx] - abs(ai)) > zero:
            # print(ai, coe_value[idx], coe_label[idx])
            return -1
        if ai > 0:
            ai = simplify(coe_label[idx])
        else:
            ai = -1*simplify(coe_label[idx])
        fr += ai*x[i]
    if abs(b) > zero:
        idx = find_2(abs(b), coe_value) - 1
        if idx + 1 < len(coe_value) and abs(coe_value[idx] - abs(b)) > abs(coe_value[idx + 1] - abs(b)):
            idx += 1
        if abs(coe_value[idx] - abs(b)) > zero:
            # print(ai, coe_value[idx], coe_label[idx])
            return -1
        if b > zero:
            b = simplify(coe_label[idx])
            fr += b
        else:
            b = simplify(coe_label[idx])
            fr -= b

    return str(fl) + '=' + str(fr)

def linear_fitting(datax, datay, epoch):
    x_num_len, data_len = np.shape(datax)
    y_num_len, _ = np.shape(datay)

    k = epoch
    x_train = np.array(datax).T[:20]
    x_test = np.array(datax).T[20:]
    y_train = np.array(datay).T[:20]
    y_test = np.array(datay).T[20:]
    # print(x_train)
    lineModel = LinearRegression()
    lineModel.fit(x_train, y_train)
    # print(lineModel.coef_)
    Y_predict = lineModel.predict(x_test)
    final_score = 0
    for k in range(len(Y_predict)):
        final_score += abs((Y_predict[k] - y_test[k])/y_test[k])
    final_score /= 10

    return final_score[0]


