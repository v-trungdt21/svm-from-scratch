import numpy as np

from svm.utils import BaseException

from .kernels import get_kernel_function


class SVM:
    def __init__(
        self, kernel="linear", C=None, degree=2.0, gamma=5.0, coef=1.0
    ):
        """Initialize SVM model
        kernel: Specifies the kernel type to be used in the algorithm.
        C: Regularization parameter. If None, use linear SVM.
        degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        coef: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """
        self.kernel = get_kernel_function(
            kernel, degree=degree, gamma=gamma, coef=coef
        )
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, data):
        pass

    def predict(self, features):
        """Infer feature set after training.
        features: np.array: data you need to infer.
        output: {-1,1}: label.
        """
        if (
            not hasattr(self, "w")
            and not hasattr(self, "b")
            and not hasattr(self, "support_vectors")
        ):
            raise BaseException("Please fit the model before inference.")
        X = np.asarray(features)
        if hasattr(self, "w"):
            return np.sign(np.dot(self.w, X) + self.b)
        # TODO: calculate sum based on alpha, support vector and kernel
        else:
            # bs = X.shape[0]
            # y_hat = np.zeros(bs)
            # for i in range(bs):
            #     s = 0
            return None
