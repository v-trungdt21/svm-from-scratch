"""Based class for implement svm kernels.
"""
import logging
from functools import partial

import numpy as np

from svm.utils import BaseException

logging.basicConfig(level=logging.DEBUG)


def linear_kernel(x, z):
    """Calculate the dot product btw two vectors.
    x, z: input arrays
    Return: <x, z>
    """
    return np.dot(x, z)


def polynomial_kernel(x, z, coef, gamma, degree):
    """Calculate the poly value btw two vectors.
    x, z: input arrays
    return: poly(x, z)
    """
    return (coef + gamma * np.dot(x, z)) ** degree


def rbf_kernel(x, z, gamma):
    """Calculate the rbf product btw two vectors.
    x, z: input arrays
    return: rbf(x, z)
    """
    if gamma < 0:
        raise "Gamma value must greater than zero."

    return np.exp(
        -1.0 * gamma * np.dot(np.subtract(x, z).T, np.subtract(x, z))
    )


def sigmoid_kernel(x, z, gamma, coef):
    """Calculate the sigmoid value btw two vectors.
    x, z: input arrays
    return: sigmoid(x, z)
    """
    # TODO: Check x: make sure input x can be transpose
    return np.tanh(gamma * np.dot(x, z) + coef)


def get_kernel_function(
    kernel="linear", degree=2.0, gamma=5.0, coef=1.0, x=None, z=None
):
    """Calculate the sigmoid value btw two vectors.
    Input:
      x, z: input arrays
      degree: Value in poly kernel
      gamma: Value in poly, sigmoid, rbf kernels
      coef: Value in poly, sigmoid kernels

    Return: cal_kernel_(x, z)
    """

    degree = float(degree)
    gamma = float(gamma)
    kernel = kernel.lower().strip()

    default_kernels = ["linear", "poly", "rbf", "sigmoid"]

    if kernel not in default_kernels:
        return BaseException(
            "SVM currently support ['linear', 'poly', \
            'rbf','sigmoid'] kernels, please choose again!"
        )
    elif kernel == "linear":
        return linear_kernel
    elif kernel == "poly":
        return partial(
            polynomial_kernel, coef=coef, gamma=gamma, degree=degree
        )
    elif kernel == "rbf":
        return partial(rbf_kernel, gamma=gamma)
    elif kernel == "sigmoid":
        return partial(sigmoid_kernel, gamma=gamma, coef=coef)
