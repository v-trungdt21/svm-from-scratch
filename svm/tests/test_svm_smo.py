import logging

import numpy as np
import pytest
import sklearn
from sklearn.svm import SVC

from svm.core.svm import SVM
from svm.utils.data import generate_linear_separable_dataset


def test_linear_separable():
    eps = 1e-3

    X, y = generate_linear_separable_dataset()

    sklearn_svm = SVC(kernel="linear", C=1e5)
    sklearn_svm.fit(X, y)
    sklearn_w = sklearn_svm.coef_
    sklearn_b = sklearn_svm.intercept_

    my_svm = SVM(kernel="linear", C=1e5)
    my_svm.fit(X, y)
    my_svm_w = my_svm.coef_
    my_svm_b = my_svm.intercept_

    w_diff = abs(sklearn_w - my_svm_w)
    b_diff = abs(sklearn_b - my_svm_b)

    # Test training SVM
    assert np.all([w_diff < eps])
    assert np.all([b_diff < eps])

    # # Test inference
    # points = [
    #     [1.9, 2.1, 3.1],
    #     [2.1, 1.9, 2.8],
    #     [3.8, 2.1, 3.1],
    #     [4.2, 2.1, 2.9],
    # ]
    # prediction_diff = clf.predict(points) - svm_opt.predict(points)

    # assert np.all([prediction_diff == 0])
