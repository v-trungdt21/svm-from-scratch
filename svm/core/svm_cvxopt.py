from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)

from svm.core.kernels import cal_rbf, get_kernel_function


def get_kernel_function_new(kernel="rbf", degree=2.0, gamma=1.0, coef=0.0):
    """Calculate the sigmoid value btw two vectors.
    Args
    ----------
        x, z: input arrays
        degree: (int, default=3). Value in poly kernel
        gamma: ('scale', 'auto') or float. Value in poly, sigmoid, rbf kernels
        coef: (float, default=0.0). Value in poly, sigmoid kernels

    Return
    ----------
        cal_kernel_(x, z)
    """

    degree = float(degree)
    kernel = kernel.lower().strip()

    default_kernels = ["linear", "poly", "rbf", "sigmoid"]

    if kernel not in default_kernels:
        raise BaseException(
            "SVM currently support ['linear'1, 'poly', \
            'rbf','sigmoid'] kernels, please choose again!"
        )
    elif kernel == "linear":
        return partial(linear_kernel)
    elif kernel == "poly":
        return partial(
            polynomial_kernel, coef0=coef, gamma=gamma, degree=degree
        )
    elif kernel == "rbf":
        return partial(rbf_kernel, gamma=gamma)
    elif kernel == "sigmoid":
        return partial(sigmoid_kernel, gamma=gamma, coef0=coef)


class SVM_cvxopt:
    def __init__(
        self, C=100, kernel="linear", degree=3.0, gamma=1.0, coef=0.0, **params
    ):
        self.w = None
        self.b = None
        self.support_vectors_ = None
        if kernel == "rbf":
            self.kernel = self.rbf_kernel(gamma=gamma)
        else:
            self.kernel = get_kernel_function(
                kernel=kernel, degree=degree, gamma=gamma, coef=coef
            )
        self.C = C

    def rbf_cal(self, X, Z, gamma):
        X = X.T
        Z = Z.T

        kernel = get_kernel_function(
            gamma=gamma,
        )

        if X.ndim == 1 and Z.ndim == 1:
            return kernel(X, Z)
        elif X.ndim == 1:
            K = np.zeros(len(Z))
            for i in range(len(Z)):
                K[i] = kernel(X, Z[i])
            return K
        elif Z.ndim == 1:
            K = np.zeros(len(X))
            for i in range(len(X)):
                K[i] = kernel(X[i], Z)
            return K
        else:
            K = np.zeros((X.shape[0], Z.shape[0]))
            for i in range(X.shape[0]):
                for j in range(Z.shape[0]):
                    K[i][j] = kernel(X[i], Z[j])
            return K

    def rbf_kernel(self, gamma=1.0):
        return partial(cal_rbf, gamma=gamma)

    def fit(self, X, y):
        """
        Training an SVM classifier using input data

        Args:
            X: Training features. Shape (N,d) with d is number of dimension
            and N is the number of data points
            y: Training labels. Shape (N,) with N is the number of data
            points
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.double)
        if len(X.shape) != 2:
            raise ValueError(
                "The shape of the dimension of features (%d) should be Nxd."
            )

        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.double)

        if y.shape[0] != X.shape[0]:
            X = X.T
        if y.shape[0] != X.shape[0]:
            raise ValueError("Shape mismatches between X and Y.")

        N = len(X)
        X = X.T
        y = y.reshape((1, N))

        X_kernel = self.kernel(X, X)
        V = X * y

        print("Shape of things")
        print(np.outer(y, y).shape)
        print(X_kernel.shape)

        V_kernel = np.outer(y, y) * X_kernel
        # V_kernel = self.kernel(V, V)
        K = matrix(V_kernel)
        print("V", V.shape)
        p = matrix(-np.ones((N, 1)))  # all-one vector
        # Build A, b, G, h
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))  # for all lambda_n >= 0
        h = matrix(np.vstack((np.zeros((N, 1)), self.C * np.ones((N, 1)))))

        A = matrix(y)  # the equality constrain is actually y^T lambda = 0
        b = matrix(np.zeros((1, 1)))

        solvers.options["show_progress"] = False
        sol = solvers.qp(K, p, G, h, A, b)
        lamda_matrix = np.array(sol["x"])
        lamda_matrix_ravel = np.ravel(sol["x"])

        # Find the support vectors
        epsilon = 1e-5  # just a small number, greater than 1e-9

        S = np.where(lamda_matrix > epsilon)[0]
        S2 = np.where(lamda_matrix < 0.999 * self.C)[0]

        M = [val for val in S if val in S2]

        self.margin_features = X[:, M]
        self.margin_labels = y[:, M]

        VS = V[:, S]
        XS = X[:, S]
        yS = y[:, S]
        lS = lamda_matrix[S]

        self.w = VS.dot(lS).T
        self.b = 0

        sv = lamda_matrix_ravel > epsilon

        ind = np.arange(len(lamda_matrix))[sv]

        a = lamda_matrix_ravel[sv]
        sv_y = y[0, sv]

        print("Shape X_kernel", X_kernel.shape)
        for n in range(len(a)):
            self.b += sv_y[n]
            self.b -= np.sum(a * sv_y * X_kernel[ind[n], sv])
        self.b /= len(a)

        self.support_vectors_ = XS.T
        self.support_vectors_label = yS
        self.lamda_support_vectors = lamda_matrix[S]

    def predict(self, features):
        """
        Infer feature set after training.
        features: np.array: data you need to infer.
        output: {-1,1}: label.
        """
        if not hasattr(self, "w") and not hasattr(self, "b"):
            raise BaseException("Please fit the model before inference.")

        if not isinstance(features, (list, np.ndarray)):
            raise TypeError("The features should be list or np.ndarray")

        if isinstance(features, list):
            features = np.asarray(features)

        if features.shape[-1] != self.w.shape[-1]:
            print(
                "The shape of the dimension of features (%d) should be the same as the dimension W matrix (%d)"
                % (features.shape[-1], self.w.shape[-1])
            )
            raise ValueError(
                "The shape of the dimension of features (%d) should be the same as the dimension W matrix (%d)"
                % (features.shape[-1], self.w.shape[-1])
            )

        return np.squeeze(np.sign(self.decision_function(features)))

    def decision_function(self, features):
        print("Decision")
        print(self.support_vectors_.shape, features.shape)
        A = np.dot(
            self.lamda_support_vectors.T * self.support_vectors_label,
            self.kernel(self.support_vectors_.T, features.T),
        )
        return A + self.b
