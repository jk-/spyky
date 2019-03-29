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
        "-1 1.16 -1 -1 1.16 -1 -1 1.16 -1",
        "-1 -1 -1 1.16 1.16 1.16 -1 -1 -1",
        "1.16 -1 -1 -1 1.16 -1 -1 -1 1.16",
        "-1 -1 1.16 -1 1.16 -1 1.16 -1 -1",
        "-1 1.16 -1 -1 1.16 1.16 -1 -1 -1",
        "-1 -1 -1 1.16 1.16 -1 -1 1.16 -1",
        "-1 1.16 -1 1.16 1.16 -1 -1 -1 -1",
        "-1 -1 -1 -1 1.16 1.16 -1 1.16 -1",
        "1.16 -1 1.16 -1 1.16 -1 -1 -1 -1",
        "1.16 -1 -1 -1 1.16 -1 1.16 -1 -1",
        "-1 -1 -1 -1 1.16 -1 1.16 -1 1.16",
        "-1 -1 1.16 -1 1.16 -1 -1 -1 1.16",
    ]

    image_at = 7
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]

    image = [x / 255 for x in image]
    image = np.reshape(image, (28, 28))

    snn = SNN(28 * 28, kernels, 10, time=100, dt=0.0125)
    snn.guess(image)

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=0.4)
    columns = 4
    rows = 3
    for i in range(1, columns * rows + 1):
        f_map_idx = i - 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(snn.feature_maps[f_map_idx], interpolation="nearest")
        plt.title("Feature Map {}".format(f_map_idx), fontsize=12)
        print(
            "Found {} spikes on kernel {}".format(
                snn.feature_maps[f_map_idx].sum(), f_map_idx
            )
        )
        plt.savefig("plots/feature_map")

    print(
        "size of hidden layer {}".format(
            np.array([x.shape[0] * x.shape[1] for x in snn.feature_maps]).sum()
        )
    )
    print("Label {}".format(labels[image_at - 1]))
    plt.show()
