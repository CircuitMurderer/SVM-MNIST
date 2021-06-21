"""
Least-square SVM
@Python 3.8.2
Author: Zhao Yuqi
"""

import numpy as np
import pandas as pd
import os


class SVM:
    def __init__(self, train_set, class_set, param_c, kernel):
        self.dataset = train_set
        self.labels = class_set
        self.C = param_c
        self.kn = kernel

        self.N = np.shape(train_set)[0]
        self.ruler = self.N // 100

        self.alpha_mat = np.mat(np.zeros((self.N, 1)))
        self.b = 0

        self.K = np.mat(np.zeros((self.N, self.N)))

        for i in range(self.N):
            self.K[:, i] = self.kernel_func(self.dataset[i, :])
            if i % self.ruler == 0:
                print('\rBuilding Model...', i // self.ruler + 1, '%', sep='', end='')
        print()

    def kernel_func(self, in_vec):
        x = np.mat(self.dataset)
        m, n = np.shape(x)

        a = np.mat(in_vec)
        k = np.mat(np.zeros((m, 1)))
        if self.kn[0] == 'lin':
            k = x * a.T
        elif self.kn[0] == 'rbf':
            for j in range(m):
                d_row = x[j, :] - a
                k[j] = d_row * d_row.T
            k = np.exp(k / (-1 * self.kn[1] ** 2))
        else:
            raise NameError('No such kernel name.')

        return k

    def least_squares(self):
        units = np.mat(np.ones((self.N, 1)))
        z_mat = np.mat(np.zeros((1, 1)))
        i_mat = np.eye(self.N)

        print('Creating UP and DOWN matrix...', end='')
        up_mat = np.hstack((z_mat, units.T))
        down_mat = np.hstack((units, self.K + i_mat / self.C))
        print('Done.')

        print('Creating calculate matrix...', end='')
        comp_mat = np.vstack((up_mat, down_mat))
        res_mat = np.hstack((z_mat, np.mat(self.labels)))
        res_mat = res_mat.T
        print('Done.')

        print('Calculating the result...', end='')
        b_alpha = np.linalg.inv(comp_mat) * res_mat
        print('Done.')

        print('Training...', end='')
        self.b = b_alpha[0, 0]
        for i in range(self.N):
            self.alpha_mat[i, 0] = b_alpha[i + 1, 0]
        print('Done.')

        return self.alpha_mat, self.b, self.K

    def train_model(self):
        alpha, beta, _ = self.least_squares()
        return alpha, beta

    def pre_value(self, vec):
        kv = self.kernel_func(vec)
        ret = kv.T * self.alpha_mat + self.b

        return np.sign(ret)

    def predict(self, test_set, test_label):
        test_n = len(test_set)
        test_ruler = test_n // 100

        error = 0.
        tp = fp = tn = fn = 0.
        for i in range(test_n):
            pre = self.pre_value(test_set[i])
            if i % test_ruler == 0:
                print('\rPredicting...', i // test_ruler + 1, '%', sep='', end='')
            if pre != test_label[i]:
                error += 1
                if pre == 1:
                    fp += 1
                else:
                    fn += 1
            else:
                if pre == 1:
                    tp += 1
                else:
                    tn += 1
        print()

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        acc = 1 - error / test_n

        return acc, prec, rec


def read_data(file_path):
    if os.path.isfile('arr_bin.npy'):
        return np.load('arr_bin.npy')
    else:
        data = pd.read_csv(file_path, encoding='utf8', header=None)
        arr0 = data.values[1:].astype('float32')

        for i in range(len(arr0[:, 0])):
            if arr0[i, 0] == 0:
                arr0[i, 0] = -1
            else:
                arr0[i, 0] = 1

        for item in arr0:
            for i in range(1, len(item)):
                if item[i] > 128:
                    item[i] = 1
                else:
                    item[i] = 0

        np.save('arr_bin.npy', arr0)
        return arr0


if __name__ == '__main__':
    arr = read_data('train.csv')

    train = 20000
    test = 5000

    train_sets, train_labels = arr[:train, 1:], arr[:train, 0]
    test_sets, test_labels = arr[train:train + test, 1:], arr[train:train + test, 0]

    c = 1
    k1 = 0.3
    mod = 'lin'

    kern = (mod, k1)

    svm = SVM(train_sets, train_labels, c, kern)
    svm.train_model()

    accuracy, precision, recall = svm.predict(test_sets, test_labels)

    print('\nSupport Vector Machine:')
    print('Train set:', str(train) + ',', 'Test set:', test,
          '- Accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}'.format(accuracy, precision, recall))
