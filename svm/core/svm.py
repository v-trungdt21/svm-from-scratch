import numpy as np

from svm.core.kernels import get_kernel_function
from svm.core.opt.smo import SequentialMinimalOptimizer as SMO
from svm.utils import BaseException


class SVM:
    def __init__(
        self,
        kernel="linear",
        C=None,
        degree=2.0,
        gamma=5.0,
        coef=1.0,
        tol=1e-3,
        max_iter=5,
    ):
        """Initialize SVM model

        Args:
            kernel: Specifies the kernel type to be used in the algorithm.
            C: Regularization parameter. If None, use linear SVM.
            degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            coef: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            tol: Tolerance for stopping criterion.
        """
        self.kernel = get_kernel_function(
            kernel, degree=degree, gamma=gamma, coef=coef
        )

        self.C = C
        if self.C is not None:
            self.C = float(self.C)

        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y, optimizer="smo"):
        """Fit the SVM model according to the given training data.

        Args:
            X: Training feature vectors.
            y: Target values (class labels in classification, real numbers in regression)
            optimizer: Training optimizer

        """
        if optimizer == "smo":
            opt = SMO(X, y, self.C, self.kernel, self.tol)
        else:
            raise BaseException("Optimizer not implemented.")

        self.dual_coef_, self.coef_, self.intercept_ = opt.solve(
            max_iter=self.max_iter
        )

        self.support_vectors_ = X[np.where(self.dual_coef_ > 0)[0], :]
        self.n_support_ = self.support_vectors_.shape

    # def decision_function(self, X):

    # def predict(self, X):
    #     """Perform model inference on feature vectors in X

    #     Args:
    #         X: np.array: data you need to infer.

    #     """
    #     if (
    #         not hasattr(self, "coef_")
    #         and not hasattr(self, "intercept_")
    #         and not hasattr(self, "support_vectors_")
    #     ):
    #         raise BaseException("Please fit the model before inference.")
    #     X = np.asarray(features)
    #     if hasattr(self, "coef_"):
    #         return np.sign(np.dot(self.w, X) + self.b)
    #     # TODO: calculate sum based on alpha, support vector and kernel
    #     else:
    #         # bs = X.shape[0]
    #         # y_hat = np.zeros(bs)
    #         # for i in range(bs):
    #         #     s = 0
    #        return None

    def cal_gamma_value(self, x, gamma="scale"):
        """Calculate the gamma value of input.
        Args
        ----------
            x: input arrays

        Return
        ----------
            gamma_value(x)
        """
        if gamma == "scale":
            return 1 / (x.shape[0] * np.var(x))
        elif gamma == "auto":
            return 1 / x.shape[0]
        else:
            return float(gamma)
