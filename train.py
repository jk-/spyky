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
        "0 .16 0 0 .16 0 0 .16 0",
        # "0 0 0 .16 .16 .16 0 0 0",
        # ".16 0 0 0 .16 0 0 0 .16",
        # "0 0 .16 0 .16 0 .16 0 0",
        # "0 .16 0 0 .16 .16 0 0 0",
        # "0 0 0 .16 .16 0 0 .16 0",
        # "0 .16 0 .16 .16 0 0 0 0",
        # "0 0 0 0 .16 .16 0 .16 0",
        # ".16 0 .16 0 .16 0 0 0 0",
        # ".16 0 0 0 .16 0 .16 0 0",
        # "0 0 0 0 .16 0 .16 0 .16",
        # "0 0 .16 0 .16 0 0 0 .16",
    ]

    image_at = 7
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]

    image = [x / 255 for x in image]
    image = np.reshape(image, (28, 28))

    #
    # lp = 101.2  # minimum constant that doesn't trigger a spike
    # l0 = 2700
    # pixel = 244
    # i(k) = l0 + (k * lp)
    # I = (2700 + (pixel * lp)) * pA

    # lif = LIFNeuron(1, 100, 1)
    # lif.set_current(I)
    # lif.spike_train(None)
    #
    # plt.title("Input Image {}".format(labels[image_at - 1]))
    # print(lif.spikes.sum())
    #
    # plt.plot(lif.time, lif.V, label="Membrane Potential")[0]
    # plt.ylim(-0.070, 0.020)
    # plt.savefig(
    #     "plots/input_image_{}".format(labels[image_at - 1]),
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    # plt.show()
    # quit()

    print("Input Digit: {}".format(labels[image_at - 1]))
    snn = SNN(28 * 28, kernels, 10, time=100, dt=1)
    snn.guess(image)

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=0.4)
    columns = 4
    rows = 3
    for i in range(1, columns * rows + 1):
        f_map_idx = i - 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(snn.feature_maps[f_map_idx], interpolation="nearest")
        plt.title("Feature Map {}".format(f_map_idx + 1), fontsize=12)

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
