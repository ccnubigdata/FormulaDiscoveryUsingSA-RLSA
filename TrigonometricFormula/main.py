from TrigonometricFormula.SA import SA
import numpy as np
import datetime

def load_var(path='datasets/', filename='var'):
    ori_var = np.load(path + filename + '.npy', allow_pickle=True)
    vis = list()
    var = list()
    ori_var = sorted(ori_var, key=lambda x: len(str(x)))
    for it in ori_var:
        if it not in vis:
            ch = [it]
            var.append(ch)
            vis.extend(ch)
    return var, np.load(path + 'data' + '.npy', allow_pickle=True).item()


if __name__ == '__main__':
    varx, datax = load_var(filename='angle')
    adj = np.load('datasets/adj.npy', allow_pickle=True)
    t1 = datetime.datetime.now()
    result = list()
    ep = list()
    sa = SA(var_x=varx, data_x=datax, adj=adj, n_actions=10,
            max_T=5, min_T=1,
            rate_T=0.995, len_code_x=6)
    print('The number of terms is {}。'.format(len(varx)))
    for i in range(100):
        exp, times, code = sa.run(isshown=0)
        if exp != -1:
            if exp in result:
                print('Finding repeated formulas: {}'.format(exp))
                pass
            else:
                print('Discover a new formula!!!: {},\n code: {}\n'.format(exp, code))
            result.append(exp)
    np.save('result/result.npy', result)
    t2 = datetime.datetime.now()
    print('The total time spent is: {}'.format(t2-t1))
