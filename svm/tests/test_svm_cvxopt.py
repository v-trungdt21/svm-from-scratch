import numpy as np
import pytest
import sklearn
from sklearn.svm import SVC

from svm.core.svm_cvxopt import SVM_cvxopt


def test_result_to_sklearn():

    means = [[2, 2, 3], [4, 2, 3]]
    cov = [[0.3, 0.2, 0.1], [0.2, 0.3, 0.3], [0.3, 0.2, 0.1]]
    N = 10
    X0 = np.random.multivariate_normal(means[0], cov, N)  # class 1
    X1 = np.random.multivariate_normal(means[1], cov, N)  # class -1
    X = np.concatenate((X0.T, X1.T), axis=1)  # all data
    y = np.concatenate(
        (np.ones((1, N)), -1 * np.ones((1, N))), axis=1
    )  # labels
    epsilon = 1e-2

    svm_opt = SVM_cvxopt()
    clf = SVC(kernel="linear", C=1e5)

    svm_opt.fit(X, y)
    y1 = y.reshape((2 * N,))
    X1 = X.T  # each sample is one row
    clf.fit(X1, y1)

    w_sklearn = clf.coef_
    b_sklearn = clf.intercept_

    w_svm_cvxopt = svm_opt.w
    b_svm_cvxopt = svm_opt.b

    w_diff = w_sklearn - w_svm_cvxopt
    b_diff = b_sklearn - b_svm_cvxopt

    # Test training SVM
    assert np.all([w_diff < epsilon])
    assert np.all([b_diff < epsilon])

    # Test inference
    points = [
        [1.9, 2.1, 3.1],
        [2.1, 1.9, 2.8],
        [3.8, 2.1, 3.1],
        [4.2, 2.1, 2.9],
    ]
    prediction_diff = clf.predict(points) - svm_opt.predict(points)

    assert np.all([prediction_diff == 0])
