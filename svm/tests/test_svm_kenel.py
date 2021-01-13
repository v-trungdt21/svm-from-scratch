"""Based class for implement svm kernels.
"""
import logging
import timeit

import numpy as np

from svm.core import svm_kernel

logging.basicConfig(level=logging.DEBUG)

def test_speed_linear():
    """Test speed (check) for np linear vs math linear kernel.
    """
    x = np.random.randint(100, size=(100000))
    z = np.random.randint(100, size=(100000))
    kernel = svm_kernel.SVMKernel(kernel_name='linear')

    logging.debug("Test speed for svm kernel linear.")
    start = timeit.default_timer()
    kf_result = kernel.kernel_function['linear'](x, z)
    stop = timeit.default_timer()
    logging.debug("Time consump by svm kernel np: %s", stop - start)

    start = timeit.default_timer()
    direct_result = sum(x_i*z_i for x_i, z_i in zip(x, z))
    start = timeit.default_timer()
    logging.debug("Time consump by svm kernel np: %s", stop - start)

    raise kf_result != direct_result

def test_speed_poly():
    """Updating...
    """
    pass

def test_speed_rbf():
    """Updating...
    """
    pass

def test_speed_tanh():
    """Updating...
    """
    pass
