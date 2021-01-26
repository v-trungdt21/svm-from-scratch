from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers

from svm.core.kernels import cal_rbf, get_kernel_function


class SVM_cvxopt:
    def __init__(
        self, C=100, kernel="linear", degree=3.0, gamma=1.0, coef=0.0, **params
    ):
        self.w = None
        self.b = None
        self.support_vectors_ = None
        self.kernel = get_kernel_function(
            kernel=kernel, degree=degree, gamma=gamma, coef=coef
        )
        self.C = C

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

        V_kernel = np.outer(y, y) * X_kernel
        # V_kernel = self.kernel(V, V)
        K = matrix(V_kernel)
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
        # print("Decision")
        # print(self.support_vectors_.shape, features.shape)
        A = np.dot(
            self.lamda_support_vectors.T * self.support_vectors_label,
            self.kernel(self.support_vectors_.T, features.T),
        )
        return A + self.b
