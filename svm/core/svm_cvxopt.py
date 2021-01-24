import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers

from svm.core.kernels import get_kernel_function


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
        pass

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
        V = X * y
        V_kernel = self.kernel(V.T, V)
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
        # Find the support vectors
        epsilon = 1e-6  # just a small number, greater than 1e-9
        S = np.where(lamda_matrix > epsilon)[0]

        VS = V[:, S]
        XS = X[:, S]
        yS = y[:, S]
        lS = lamda_matrix[S]

        # calculate w and b
        self.w = VS.dot(lS).T
        self.b = np.mean(yS.T - self.w.dot(XS))
        self.lamda_matrix = lamda_matrix
        self._get_support_vectors(X.T)

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

        return np.squeeze(np.sign(np.dot(self.w, features.T) + self.b))

    def decision_function(self, features):
        return np.dot(self.w, features.T) + self.b

    def _get_support_vectors(self, features):
        distance_to_margin = self.decision_function(features)
        support_vectors_index = np.where(
            np.abs(distance_to_margin[0]) <= 1 + 1e-1
        )
        self.support_vectors_ = features[support_vectors_index]
