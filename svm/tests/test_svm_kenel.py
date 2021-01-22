"""Based class for implement svm kernels.
"""
import logging
import timeit
from functools import partial

import numpy as np

from svm.core import kernels
from svm.utils import BaseException

logging.basicConfig(level=logging.DEBUG)


def test_speed_linear():
    """Test speed (check) for np linear vs math linear kernel."""
    x = np.random.randint(100, size=(100000))
    z = np.random.randint(100, size=(100000))

    logging.debug("Test speed for svm kernel linear.")
    start = timeit.default_timer()
    kf_linear = kernels.get_kernel_function(kernel="linear")
    kf_result = kf_linear(x=x, z=z)
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel linear: %s", stop - start)

    start = timeit.default_timer()
    direct_result = sum(x_i * z_i for x_i, z_i in zip(x, z))
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel direct: %s", stop - start)

    logging.debug("kf_result: %s, direct_result: %s", kf_result, direct_result)

    if kf_result != direct_result:
        raise BaseException("False")


def test_speed_poly():
    """Updating..."""
    pass


def test_speed_rbf():
    """Test speed (check) for svm kernel rbf scaler vs math linear kernel."""
    x = np.random.randint(100, size=(100000))
    z = np.random.randint(100, size=(100000))

    logging.debug("Test speed for svm kernel rbf.")
    start = timeit.default_timer()
    kf_linear = kernels.get_kernel_function(kernel="rbf", gamma=1.00)
    kf_result = kf_linear(x=x, z=z)
    logging.debug("Value of svm kernel rbf scale: %s", kf_result)
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel rbf scale: %s", stop - start)

    start = timeit.default_timer()
    kf_linear = kernels.get_kernel_function(kernel="rbf", gamma=1.00)
    kf_result = kf_linear(x=x, z=z)
    logging.debug("Value of svm kernel rbf: %s", kf_result)
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel rbf auto: %s", stop - start)


def test_speed_tanh():
    """Test speed (check) for np linear vs math linear kernel."""
    pass


if __name__ == "__main__":
    test_speed_rbf()
