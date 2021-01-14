"""Based class for implement svm kernels.
"""
import logging

import numpy as np

from svm.utils import BaseException

logging.basicConfig(level=logging.DEBUG)

def cal_kernel_linear(x, z):
    """Calculate the dot product btw two vectors.
    x, z: input arrays
    Return: <x, z>
    """
     # TODO: Check x.T: make sure input x can be transpose
    return np.dot(x.T, z)

def cal_kernel_polynomial(x, z, coef, gamma, degree):
    """Calculate the poly value btw two vectors.
    x, z: input arrays
    return: poly(x, z)
    """
     # TODO: Check x.T: make sure input x can be transpose
    return (coef + gamma * np.dot(x.T, z)) ** degree

def cal_kernel_rbf(x, z, gamma):
    """Calculate the rbf product btw two vectors.
    x, z: input arrays
    return: rbf(x, z)
    """
     # TODO: Check x.T: make sure input x can be transpose
    if gamma < 0:
        raise "Gamma value must greater than zero."

    return np.exp(
        -1.0 * gamma * np.dot(np.subtract(x, z).T, np.subtract(x, z))
    )

def cal_kernel_sigmoid(x, z, gamma, coef):
    """Calculate the sigmoid value btw two vectors.
    x, z: input arrays
    return: sigmoid(x, z)
    """
    # TODO: Check x.T: make sure input x can be transpose
    return np.tanh(gamma * np.dot(x.T, z) + coef)

def kernel_function(kernel='linear', degree=2, gamma=5,
                    coef=1, x=None, z=None):
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

    default_kernels = ["linear", "poly", "rbf", "sigmoid"]

    if kernel not in default_kernels:
        return BaseException(
            "SVM currently support ['linear', 'poly', \
            'rbf','sigmoid'] kernels, please choose again!")

    if kernel == 'linear':
        return cal_kernel_linear(x, z)
    elif kernel == 'poly':
        return cal_kernel_polynomial(x, z, coef, gamma, degree)
    elif kernel == 'rbf':
        return cal_kernel_rbf(x, z, gamma)
    elif kernel == 'sigmoid':
        return cal_kernel_sigmoid(x, z, gamma, coef)

    return None
