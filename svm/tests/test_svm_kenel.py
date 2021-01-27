"""Based class for implement svm kernels.
"""
import logging
import timeit
from functools import partial

import numpy as np
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)

from svm.core import kernels
from svm.utils import BaseException

logging.basicConfig(level=logging.DEBUG)


def cal_gamma_value(x, gamma="scale"):
    """Calculate the gamma value of input.
    Args
    ----------
        x: input arrays

    Return
    ----------
        gamma_value(x)
    """
    if x.ndim == 1:
        n_features = 1
    else:
        n_features = x.shape[1]

    if gamma == "scale":
        return 1 / (n_features * np.var(x))
    elif gamma == "auto":
        return 1 / n_features
    else:
        return float(gamma)


def cal_kernel_value(x, z, kernel_func):
    """Calculate the kernel value."""
    if x.ndim == 1 and z.ndim == 1:
        return kernel_func(x, z)

    elif x.ndim == 1:
        kernel_values = np.zeros(len(z))
        for i in range(len(z)):
            kernel_values[i] = kernel_func(x, z[i])
        return kernel_values
    elif z.ndim == 1:
        kernel_values = np.zeros(len(x))
        for i in range(len(x)):
            kernel_values[i] = kernel_func(x[i], z)
        return kernel_values
    else:
        kernel_values = np.zeros((x.shape[0], z.shape[0]))
        for i in range(x.shape[0]):
            for j in range(z.shape[0]):
                kernel_values[i][j] = kernel_func(x[i], z[j])
        return kernel_values


def test_linear_kernel():
    """Compare the poly kernel of sklearn and our poly kernel."""
    x = np.array([[1, 2, 3, 4, 4], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    z = np.array([[3, 4, 5, 6, 1], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])

    # Calculate our kernel
    kernel_func = kernels.get_kernel_function(kernel="linear")
    our_result = cal_kernel_value(x, z, kernel_func)
    logging.debug("Value of our linear kernel: %s", our_result)

    # Calculate sklearn kernel
    sklearn_result = linear_kernel(x, z)
    logging.debug("Value of sklearn linear kernel: %s", sklearn_result)

    comparison = our_result == sklearn_result
    equal_arrays = comparison.all()
    if not equal_arrays:
        raise BaseException("False")


def test_sigmoid_kernel():
    """Compare the poly kernel of sklearn and our poly kernel."""
    x = np.array([[1, 2, 3, 4, 4], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    z = np.array([[3, 4, 5, 6, 1], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    # Calculate gamma value
    gamma_value = cal_gamma_value(x)

    # Calculate our kernel
    kernel_func = kernels.get_kernel_function(
        kernel="sigmoid", gamma=gamma_value, coef=1.0
    )
    our_result = cal_kernel_value(x, z, kernel_func)
    logging.debug("Value of our sigmoid kernel: %s", our_result)

    # Calculate sklearn kernel
    sklearn_result = sigmoid_kernel(x, z, gamma=gamma_value, coef0=1)
    logging.debug("Value of sklearn sigmoid kernel: %s", sklearn_result)

    comparison = our_result == sklearn_result
    equal_arrays = comparison.all()
    if not equal_arrays:
        raise BaseException("False")


def test_poly_kernel():
    """Compare the poly kernel of sklearn and our poly kernel."""
    x = np.array([[1, 2, 3, 4, 4], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    z = np.array([[3, 4, 5, 6, 1], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    # Calculate gamma value
    gamma_value = cal_gamma_value(x)

    # Calculate our kernel
    kernel_func = kernels.get_kernel_function(
        kernel="poly", degree=3.0, gamma=gamma_value, coef=0.0
    )
    our_result = cal_kernel_value(x, z, kernel_func)
    logging.debug("Value of our poly kernel: %s", our_result)

    # Calculate sklearn kernel
    sklearn_result = polynomial_kernel(
        x, z, degree=3.0, gamma=gamma_value, coef0=0.0
    )
    logging.debug("Value of sklearn poly kerel: %s", sklearn_result)

    comparison = our_result == sklearn_result
    equal_arrays = comparison.all()
    if not equal_arrays:
        raise BaseException("False")


def test_rbf_kernel():
    """Compare the poly kernel of sklearn and our poly kernel."""
    x = np.array([[1, 2, 3, 4, 4], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    z = np.array([[3, 4, 5, 6, 1], [1, 2, 3, 3, 4], [1, 2, 3, 3, 4]])
    # Calculate gamma value
    gamma_value = cal_gamma_value(x)

    # Calculate our kernel
    kernel_func = kernels.get_kernel_function(kernel="rbf", gamma=gamma_value)
    our_result = cal_kernel_value(x, z, kernel_func)
    logging.debug("Value of our rbf kernel: %s", our_result)

    # Calculate sklearn kernel
    sklearn_result = rbf_kernel(x, z, gamma=gamma_value)
    logging.debug("Value of sklearn rbf kernel: %s", sklearn_result)

    comparison = our_result == sklearn_result
    equal_arrays = comparison.all()
    if not equal_arrays:
        raise BaseException("False")


if __name__ == "__main__":
    test_linear_kernel()
    test_sigmoid_kernel()
    test_poly_kernel()
    test_rbf_kernel()
