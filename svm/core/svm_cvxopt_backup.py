from __future__ import division

from abc import ABC, abstractmethod

import numpy as np
from cvxopt import matrix, solvers

from svm.core.kernels import get_kernel_function


class SVM(ABC):
    def __init__(
        self,
        kernel="linear",
        C=100,
        degree=3.0,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        max_iter=-1,
    ):
        self.kernel = kernel
        self.C = C

        self.alpha = None
        self.w = None
        self.bias = None

        self.X = None
        self.Y = None

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    @abstractmethod
    def fit(self):
        pass

    def cal_kernel(self, X, Z):
        X = np.array(X)
        Z = np.array(Z)
        gamma_value = self.cal_gamma_value(X, gamma=self.gamma)
        self.kernel_func = get_kernel_function(
            kernel=self.kernel,
            degree=self.degree,
            gamma=gamma_value,
            coef=self.coef0,
        )
        if X.ndim == 1 and Z.ndim == 1:
            return self.kernel_func(X, Z)
        elif X.ndim == 1:
            K = np.zeros(len(Z))
            for i in range(len(Z)):
                K[i] = self.kernel_func(X, Z[i])
            return K
        elif Z.ndim == 1:
            K = np.zeros(len(X))
            for i in range(len(X)):
                K[i] = self.kernel_func(X[i], Z)
            return K
        else:
            K = np.zeros((X.shape[0], Z.shape[0]))
            for i in range(X.shape[0]):
                for j in range(Z.shape[0]):
                    K[i][j] = self.kernel_func(X[i], Z[j])
            return K

    def cal_gamma_value(self, x, gamma="scale"):
        """Calculate the gamma value of input.
        Args
        ----------
            x: input arrays

        Return
        ----------
            gamma_value(x)
        """
        x = np.array(x)
        if x.ndim == 1:
            n_features = 1
        else:
            _, n_features = x.shape

        if gamma == "scale":
            return 1 / (n_features * np.var(x))
        elif gamma == "auto":
            return 1 / n_features
        else:
            return float(gamma)

    def objective(self, alpha, X, Y):
        """Returns the SVM objective function based in the input model defined by:"""
        return np.sum(alpha) - 0.5 * np.sum(
            np.outer(Y, Y) * self.cal_kernel(X, X) * np.outer(alpha, alpha)
        )

    def project(self, X_train, Y_train, X_test):
        return (
            np.dot(self.alpha * Y_train, self.cal_kernel(X_train, X_test))
            - self.bias
        )

    def predict(self, X_test):
        return np.sign(self.project(self.X, self.Y, X_test))

    @staticmethod
    def keep_support_vector(alpha, X, Y, support_threshold=1e-5):
        mask = alpha > support_threshold
        index = np.arange(len(alpha))[mask]
        alpha = alpha[mask]
        X = X[mask]
        Y = Y[mask]

        return alpha, X, Y, index, mask


class SVM_cvx_backup(SVM):
    def __init__(
        self,
        degree=3.0,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        max_iter=-1,
        kernel="linear",
        C=1000,
    ):
        super().__init__(kernel, C)

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.X = X
        self.Y = Y
        self.support_vectors_ = None

        num_samples = X.shape[0]
        cal_kernel = self.cal_kernel(X, X)

        # solve using CVXOPT optimizer
        P = matrix(np.outer(Y, Y) * cal_kernel)
        q = matrix(-np.ones(num_samples))
        A = matrix(Y, (1, num_samples), "d")
        b = matrix(0.0)

        if self.C is None:
            G = matrix(-np.eye(num_samples))
            h = matrix(np.zeros(num_samples))
        else:
            tmp1 = np.diag(np.ones(num_samples) * -1)
            tmp2 = np.identity(num_samples)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(num_samples)
            tmp2 = np.ones(num_samples) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(sol["x"])

        self.alpha, self.X, self.Y, index, mask = SVM.keep_support_vector(
            self.alpha, self.X, self.Y
        )

        self.support_vectors_ = self.X

        # Bias parameter
        self.bias = 0
        for i in range(len(self.alpha)):
            self.bias -= self.Y[i]
            self.bias += np.sum(
                self.alpha * self.Y * cal_kernel[index[i], mask]
            )
        self.bias /= len(self.alpha)

    def decision_function(
        self,
        X_test=None,
        X_train=None,
        Y_train=None,
    ):
        X_train = self.X
        Y_train = self.Y
        return (
            np.dot(self.alpha * Y_train, self.cal_kernel(X_train, X_test))
            - self.bias
        )
