import sys
import random
import math
import matplotlib.pyplot as plt

import numpy as np

from snn.util import to_kernel, pa_to_nanoamps
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

    str_kernels = [
        "0 3800 0 0 3800 0 0 3800 0",
        "0 0 0 3800 3800 3800 0 0 0",
        "3800 0 0 0 3800 0 0 0 3800",
        "0 0 3800 0 3800 0 3800 0 0",
        "0 3800 0 0 3800 3800 0 0 0",
        "0 0 0 3800 3800 0 0 3800 0",
        "0 3800 0 3800 3800 0 0 0 0",
        "0 0 0 0 3800 3800 0 3800 0",
        "3800 0 3800 0 3800 0 0 0 0",
        "3800 0 0 0 3800 0 3800 0 0",
        "0 0 0 0 3800 0 3800 0 3800",
        "0 0 3800 0 3800 0 0 0 3800",
    ]

    kernels = []
    for kernel in str_kernels:
        kernels.append(to_kernel(kernel, 3))

    outputs = [x for x in range(0, 10)]

    w, h = (28, 28)

    layer_1_size = w * h
    layer_2_size = len(kernels) * (w - 2) * (h - 2)
    layer_3_size = len(outputs)

    T = 100
    dt = 1
    l1 = [LIFNeuron(x, T, dt) for x in range(0, layer_1_size)]
    # l2 = [LIFNeuron(x, T, dt) for x in range(0, layer_2_size)]  # 8112
    l2 = []

    image_at = 7
    image = images[
        (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    ]

    # plt.imshow(np.array(image).reshape(w, h), interpolation="nearest")
    # plt.title("MNIST Digit {}".format(labels[image_at - 1]), fontsize=12)
    # plt.savefig("plots/input_image")

    for idx, value in enumerate(image):
        l1[idx].set_current(value)
        l1[idx].spike_train()

    l1_reshape = np.array(l1).reshape(w, h)

    feature_map_idx = 0
    cur_n = 0
    window = [
        math.exp(-(x / 5)) - math.exp(-(x / 1.25)) for x in range(0, T + 1)
    ]

    feature_maps = np.zeros([12], dtype=object)
    for idx, feature_map in enumerate(feature_maps):
        feature_maps[idx] = np.zeros((26, 26))

    feature_map_idx = 0
    for kernel in kernels:
        kernel_width = kernel.shape[0]
        kernel_flat = kernel.flatten()
        print(
            "{}/{}: Feature Map {}".format(
                feature_map_idx + 1, len(kernels), kernel_flat
            )
        )

        for x in range(0, l1_reshape.shape[1] - 2):
            for y in range(0, l1_reshape.shape[0] - 2):
                neurons = l1_reshape[
                    x : x + kernel_width, y : y + kernel_width
                ]
                neurons = neurons.flatten()
                spike_count = 0
                for idx, neuron in enumerate(neurons):
                    signal = neuron.spikes
                    I = (
                        kernel_flat[idx]
                        * np.convolve(signal, window, "same")[0 : len(signal)]
                    )
                    n = LIFNeuron("L2 LIF: {}".format(cur_n), T, dt)
                    n.set_current_history(I)
                    n.spike_train()
                    spike_count += n.spike_count
                feature_maps[feature_map_idx][x][y] = spike_count
                cur_n += 1

        feature_map_idx += 1

    # for neuron_pointer in range(0, len(l2)):
    #     c = l2[neuron_pointer].spike_count
    #     if c > 0:
    #         print("found")

    # for map in feature_maps:
    #     plt.imshow(map, interpolation="nearest")
    #     plt.show()

    # snn = SNN(28 * 28, str_kernels, 10, time=100, dt=1)
    # snn.guess(image)
    #
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=0.4)
    columns = 4
    rows = 3
    for i in range(1, columns * rows + 1):
        f_map_idx = i - 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(feature_maps[f_map_idx], interpolation="nearest")
        plt.title("Feature Map {}".format(f_map_idx + 1), fontsize=12)
    fig.tight_layout()
    plt.savefig("plots/feature_map")
    plt.show()
