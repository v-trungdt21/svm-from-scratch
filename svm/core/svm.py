import numpy as np

from svm.utils import BaseException


class SVM:
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, features):
        """
        Infer feature set after training.
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
