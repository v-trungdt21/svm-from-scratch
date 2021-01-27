"""Based class for implement svm kernels.
"""
import logging
from functools import partial

import numpy as np

from svm.utils import BaseException

logging.basicConfig(level=logging.DEBUG)


def linear_kernel(x, z):
    """Calculate the dot product btw two vectors.
    Args
    ----------
        x, z: input arrays

    Return
    ----------
        <x, z>
    """
    return np.dot(x.T, z)


def polynomial_kernel(x, z, coef, gamma, degree):
    """Calculate the poly value btw two vectors.
    Args
    ----------
        x, z: input arrays

    Return
    ----------
        poly(x, z)
    """
    return (coef + gamma * np.dot(x.T, z)) ** degree


def rbf_kernel(x, z, gamma):
    """Calculate the rbf product btw two vectors.
    Args
    ----------
        x, z: input arrays

    Return
    ----------
        rbf(x, z)
    """
    if gamma < 0:
        raise BaseException("Gamma value must greater than zero.")

    return np.exp(-1.0 * gamma * np.linalg.norm(x - z) ** 2)


def cal_rbf(x, z, gamma):
    """Calculate the rbf product btw two vectors.
    Args
    ----------
        x, z: input arrays

    Return
    ----------
        rbf(x, z)
    """
    x = x.T
    z = z.T

    if x.ndim == 1 and z.ndim == 1:
        return rbf_kernel(x, z, gamma)
    elif x.ndim == 1:
        K = np.zeros(len(z))
        for i in range(len(z)):
            K[i] = rbf_kernel(x, z[i], gamma)
        return K
    elif z.ndim == 1:
        K = np.zeros(len(x))
        for i in range(len(x)):
            K[i] = rbf_kernel(x[i], z, gamma)
        return K
    else:
        K = np.zeros((x.shape[0], z.shape[0]))
        for i in range(x.shape[0]):
            for j in range(z.shape[0]):
                K[i][j] = rbf_kernel(x[i], z[j], gamma)
        return K


def sigmoid_kernel(x, z, gamma, coef):
    """Calculate the sigmoid value btw two vectors.

    Args:
    ----------
        x, z: input arrays

    Return:
    ----------
        sigmoid(x, z)
    """
    # TODO: Check x: make sure input x can be transpose
    return np.tanh(gamma * np.dot(x.T, z) + coef)


def get_kernel_function(kernel="rbf", degree=3.0, gamma=1.0, coef=0.0):
    """Calculate the sigmoid value btw two vectors.
    Args
    ----------
        x, z: input arrays
        degree: (int, default=3). Value in poly kernel
        gamma: ('scale', 'auto') or float. Value in poly, sigmoid, rbf kernels
        coef: (float, default=0.0). Value in poly, sigmoid kernels

    Return
    ----------
        cal_kernel_(x, z)
    """

    degree = float(degree)
    kernel = kernel.lower().strip()

    default_kernels = ["linear", "poly", "rbf", "sigmoid"]

    if kernel not in default_kernels:
        raise BaseException(
            "SVM currently support ['linear'1, 'poly', \
            'rbf','sigmoid'] kernels, please choose again!"
        )
    elif kernel == "linear":
        return linear_kernel
    elif kernel == "poly":
        return partial(
            polynomial_kernel, coef=coef, gamma=gamma, degree=degree
        )
    elif kernel == "rbf":
        return partial(cal_rbf, gamma=gamma)
    elif kernel == "sigmoid":
        return partial(sigmoid_kernel, gamma=gamma, coef=coef)
