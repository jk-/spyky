import sys
import random
import matplotlib.pyplot as plt

import numpy as np

from snn.util import convert_kernel
from snn.util import MNIST
from snn import SNN
from snn.neuron import LIFNeuron


if __name__ == "__main__":
    np.set_printoptions(
        suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize
    )

    mnist = MNIST()
    labels, images = mnist.load_data()
    magic, size, rows, cols = mnist.image_meta
    #
    # kernels = [
    #     "0 1 0 0 1 0 0 1 0",
    #     "0 0 0 1 1 1 0 0 0",
    #     "1 0 0 0 1 0 0 0 1",
    #     "0 0 1 0 1 0 1 0 0",
    #     "0 1 0 0 1 1 0 0 0",
    #     "0 0 0 1 1 0 0 1 0",
    #     "0 1 0 1 1 0 0 0 0",
    #     "0 0 0 0 1 1 0 1 0",
    #     "1 0 1 0 1 0 0 0 0",
    #     "1 0 0 0 1 0 1 0 0",
    #     "0 0 0 0 1 0 1 0 1",
    #     "0 0 1 0 1 0 0 0 1",
    #

    kernels = [
        "0 1 0 0 1 0 0 1 0",
        "0 0 0 1 1 1 0 0 0",
        "1 0 0 0 1 0 0 0 1",
        "0 0 1 0 1 0 1 0 0",
        "0 1 0 0 1 1 0 0 0",
        "0 0 0 1 1 0 0 1 0",
        "0 1 0 1 1 0 0 0 0",
        "0 0 0 0 1 1 0 1 0",
        "1 0 1 0 1 0 0 0 0",
        "1 0 0 0 1 0 1 0 0",
        "0 0 0 0 1 0 1 0 1",
        "0 0 1 0 1 0 0 0 1",
    ]

    image_at = 1
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]

    image = [x / 255 for x in image]
    image = np.reshape(image, (28, 28))

    snn = SNN(28 * 28, kernels, 10, time=100, dt=0.0125)
    snn.guess(image)

    for idx, feature in enumerate(snn.feature_maps):
        fig = plt.figure(idx)
        plt.title("Feature Map Kernel {}".format(idx))
        plt.imshow(feature, interpolation="nearest")

    plt.show()
