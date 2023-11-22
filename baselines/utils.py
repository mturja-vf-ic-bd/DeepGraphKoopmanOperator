import numpy as np
from unittest import TestCase
from sklearn.metrics.pairwise import pairwise_kernels


def polynomial_kernel(X, d):
    """

    :param X: n x t matrix
    :param d: max degree of the polynomial
    :return:
    """
    psiX = []
    if d == 0:
        return X
    for i in range(0, d + 1):
        psiX.append(np.power(X, i))

    return np.stack(psiX, axis=1).reshape(-1, X.shape[1])


class KernelRidgeRegression:
    def __init__(self, l):
        self.l = l

    def fit(self, X, Y=None):
        if Y is None:
            Y = X[:, 1:]
            X = X[:, :-1]
        G = X @ X.T
        A = Y @ X.T
        G_inv = np.linalg.inv(G + self.l * np.eye(G.shape[0]))
        K = A @ G_inv
        self._operator = K

    def predict(self, X):
        return self._operator @ X


class testKernel(TestCase):
    def test_polynomial_kernel(self):
        X = np.arange(0, 50).reshape(5, 10)
        d = 3
        psiX = polynomial_kernel(X, d)
        x = np.random.random((5, 1))

        krr = KernelRidgeRegression(l=10)
        krr.fit(X[:, :-1], X[:, 1:])
        y = krr.predict(x)

        self.assertEqual(psiX.shape, (X.shape[0] * (d + 1), X.shape[1]))
        self.assertEqual(x.shape, y.shape)

