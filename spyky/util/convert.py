import numpy as np


def to_kernel(k: str, size: int):
    """Converts a kernel, in string format, to a numpy array

    Args:
        k (st): A kernel in string format
        size (int): The size of the kernel

    Returns:
        array (numpy): A kernel in the shape `size` from string `k`
    """
    return np.array(
        [float(x) for x in k.replace("\n", "").split(" ")]
    ).reshape((size, size))


def mv_to_volts(mV):
    return mV / 1000


def pa_to_amperes(pa):
    return pa / 1000000000000


def pa_to_nanoamps(pa):
    return pa / 1000
