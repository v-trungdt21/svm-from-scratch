import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, printing, solvers


class SVM_cvxopt:
    def __init__(self):
        self.w = None
        self.b = None
        pass

    def fit(self, X, y):
        """
        Training an SVM classifier using input data

        Args:
            X: Training features. Shape (d,N) with d is number of dimension
            and N is the number of data points
            y: Training labels. Shape (1,N) with N is the number of data
            points
        """
        N = len(X.T)
        V = X * y
        K = matrix(V.T.dot(V))
        p = matrix(-np.ones((N, 1)))  # all-one vector
        # Build A, b, G, h
        G = matrix(-np.eye(N))  # for all lambda_n >= 0
        h = matrix(np.zeros((N, 1)))
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
