import numpy as np
from sympy import *
from sklearn.linear_model import LinearRegression
coe_value = np.load('coefficient_value.npy', allow_pickle=True)
coe_label = np.load('coefficient_label.npy', allow_pickle=True)

def linear_fitting(x_data, y_data):
    _, data_len = np.shape(x_data)
    x_train = np.array(x_data).T
    y_train = np.array(y_data).T
    line_model = LinearRegression()
    line_model.fit(x_train, y_train)
    y_predict = line_model.predict(x_train)
    line_model.score(x_train, y_train)
    final_score = 0
    lenx = 0
    for k in range(len(y_predict)):
        final_score += abs((y_predict[k] - y_train[k])/y_train[k])
        lenx += 1
    final_score /= lenx
    arr, b = line_model.coef_[0], line_model.intercept_[0]
    return final_score, arr, b


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

def SimplifyFormula(formula):
    S = symbols('Area')
    equal_idx = formula.find('=')
    left_formula = simplify(formula[: equal_idx])
    right_formula = simplify(formula[equal_idx+1: ])

    area_formula = 1
    count = 0
    while str(left_formula).find('Area') == 0 and count < 4:
        area_formula *= S
        left_formula /= S
        count += 1

    right_formula = simplify(right_formula / left_formula)
    left_formula = area_formula

    # squre_idx = str(left_formula).find('**')
    # if squre_idx == -1:
    #     p = 1
    # else:
    #     p = int(str(left_formula)[squre_idx+2:])
    # right_formula = simplify(right_formula**(1/p))

    return str(left_formula)+'='+str(right_formula)

def deal_result(x, y, dx, dy, leny, aim_str='Area'):
    x_train = np.array(dx).T
    y_train = np.array(dy).T
    lineModel = LinearRegression()
    lineModel.fit(x_train, y_train)
    arr, b = lineModel.coef_[0], lineModel.intercept_[0]
    # print(arr, x, y)
    zero = 1e-8
    S = Symbol(aim_str)
    fr, fl = 0, 1
    chy = y[0]
    # print(leny, len(x))
    for i in range(len(x)-leny, len(x)):
        ai = arr[i]
        # print(i, ai, x[i])
        if abs(ai) < zero:
            continue
        idx = find_2(abs(ai), coe_value) - 1
        if idx + 1 < len(coe_value) and abs(coe_value[idx] - abs(ai)) > abs(coe_value[idx + 1] - abs(ai)):
            idx += 1
        if abs(coe_value[idx] - abs(ai)) > zero:
            # print(ai, coe_value[idx], coe_label[idx])
            return -1

        if ai > 0:
            ai = -1 * simplify(coe_label[idx])
        else:
            ai = simplify(coe_label[idx])
        # print(x[i])
        chy += ai * x[i]

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
        b = simplify(coe_label[idx])
        fr += b
    fr = simplify(fr)
    chy = factor(chy)


    chy = chy.factor()
    fr = fr.factor()

    return SimplifyFormula(str(chy) + '=' + str(fr))

