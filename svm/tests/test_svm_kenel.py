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
    kf_result = partial(kernels.get_kernel_function, kernel="linear", x=x, z=z)
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel linear: %s", stop - start)

    start = timeit.default_timer()
    direct_result = sum(x_i * z_i for x_i, z_i in zip(x, z))
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel direct: %s", stop - start)

    logging.debug("kf_result: %s, direct_result: %s", kf_result, direct_result)

    if kf_result != direct_result:
        return BaseException("False")


def test_speed_poly():
    """Updating..."""
    pass


def test_speed_rbf():
    """Updating..."""
    pass


def test_speed_tanh():
    """Updating..."""
    pass
