import numpy as np


def convert_kernel(k):
    return np.array(
        [float(x) for x in k.replace("\n", "").split(" ")]
    ).reshape((3, 3))
