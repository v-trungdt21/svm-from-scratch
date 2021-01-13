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
        if not hasattr(self, "w") and not hasattr(self, "b"):
            raise BaseException("Please fit the model before inference.")

        return np.sign(np.dot(self.w, features) + self.b)
