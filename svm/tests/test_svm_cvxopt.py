import numpy as np
import pytest
import sklearn
from sklearn.svm import SVC

from svm.core.svm_cvxopt import SVM_cvxopt
from svm.utils.data import (
    generate_linear_separable_dataset,
    generate_linear_separable_dataset_old,
    generate_nonlinear_separable_dataset_old,
)


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

    y1 = y.reshape((2 * N,))
    X1 = X.T  # each sample is one row
    # print(X, y1)
    print(X.shape, y1.shape)
    svm_opt.fit(X.T, y1)
    clf.fit(X1, y1)

    # print(y.shape)
    # print(y1.shape)

    # print(y, y1)

    w_sklearn = clf.coef_
    b_sklearn = clf.intercept_

    w_svm_cvxopt = svm_opt.w
    b_svm_cvxopt = svm_opt.b

    w_diff = w_sklearn - w_svm_cvxopt
    b_diff = b_sklearn - b_svm_cvxopt

    print(w_sklearn, w_svm_cvxopt)
    print(b_sklearn, b_svm_cvxopt)
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

    # print(svm_opt.decision_function(X.T))
    # print(svm_opt.lamda_matrix)
    # print(svm_opt.support_vectors_)
    # print(clf.support_vectors_)
    # a = a/0
    assert np.all([prediction_diff == 0])


def test_result_with_data_gen():
    test_kernel = "linear"
    epsilon = 2 * 1e-2
    # X, y = generate_nonlinear_separable_dataset_old()

    X, y = generate_linear_separable_dataset_old()
    X = X.astype("float")
    y = y.astype("float")

    sample_idx = np.random.random_integers(0, X.shape[0] - 1, 5)
    sample_points = X[sample_idx]

    svm_opt = SVM_cvxopt(kernel=test_kernel, C=100)
    clf = SVC(kernel=test_kernel, gamma=1, degree=2.0, C=100)

    # print(type(X))
    # print(X.shape, y.shape)

    idx = np.where(y == 0)
    y[idx] = -1

    svm_opt.fit(X, y)
    clf.fit(X, y)

    if test_kernel == "linear":
        w_sklearn = clf.coef_
        b_sklearn = clf.intercept_

        w_svm_cvxopt = svm_opt.w
        b_svm_cvxopt = svm_opt.b

        w_diff = w_sklearn - w_svm_cvxopt
        b_diff = b_sklearn - b_svm_cvxopt

        print(w_sklearn, w_svm_cvxopt)
        print(b_sklearn, b_svm_cvxopt)

        print(w_diff, b_diff)
        # Test training SVM
        assert np.all([abs(w_diff) < epsilon])
        assert np.all([abs(b_diff) < epsilon])

    dec_sklearn = clf.decision_function(sample_points)
    dec_svm_cvxopt = svm_opt.decision_function(sample_points)

    dec_diff = dec_sklearn - dec_svm_cvxopt
    print("DIFF")
    print(dec_sklearn, dec_svm_cvxopt)
    # print(dec_diff)

    assert np.all([abs(dec_diff) < epsilon])
    # a=a/0


if __name__ == "__main__":
    test_result_to_sklearn()
    # test_result_with_data_gen()
