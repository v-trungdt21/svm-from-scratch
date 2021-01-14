"""Based class for implement svm kernels.
"""
import logging

import numpy as np

logging.basicConfig(level=logging.DEBUG)


class SVMKernel:
    """Defination of svm kernel. Calculate the ouput for K(x, z).
    With x, z are input of kernel function. Four kernels used in
    this class:
    1. linear
    2. polynomial
    3. rbf
    4. sigmoid
    4. Updating ...
    """

    def __init__(self, kernel_name="linear", coef=1, degree=3, gamma=5):
        default_kernels = ["linear", "poly", "rbf", "sigmoid"]
        if kernel_name not in default_kernels:
            raise "SVM currently support ['linear', 'poly', 'rbf', \
                  'sigmoid'] kernels, please choose again!"
        self.kernel_name = kernel_name
        self.coef = coef  # Value in poly, sigmoid kernels
        self.degree = float(degree)  # Value in poly kernel
        self.gamma = float(gamma)  # Value in poly, sigmoid, rbf kernels
        self.kernel_function = {
            "linear": self.linear,
            "poly": self.polynomial,
            "rbf": self.rbf,
            "tanh": self.sigmoid,
        }

    def linear(self, x, z):
        """Calculate the dot product btw two vectors.
        x, z: input arrays
        Return: <x, z>
        """
        return np.dot(x.T, z)

    def polynomial(self, x, z):
        """Calculate the poly value btw two vectors.
        x, z: input arrays
        return: poly(x, z)
        """
        return (self.coef + self.gamma * np.dot(x.T, z)) ** self.degree

    def rbf(self, x, z):
        """Calculate the rbf product btw two vectors.
        x, z: input arrays
        return: rbf(x, z)
        """
        if self.gamma < 0:
            raise "Gamma value must greater than zero."

        return np.exp(
            -1.0 * self.gamma * np.dot(np.subtract(x, z).T, np.subtract(x, z))
        )

    def sigmoid(self, x, z):
        """Calculate the sigmoid value btw two vectors.
        x, z: input arrays
        return: sigmoid(x, z)
        """
        return np.tanh(self.gamma * np.dot(x.T, z) + self.coef)
