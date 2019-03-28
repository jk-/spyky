# for observing data
import sys
import random
import matplotlib.pyplot as plt

# parsing mnist
import numpy as np

from snn.util import convert_kernel
from snn.util import MNIST
from snn import SNN
from snn.neuron import LIFNeuron


if __name__ == "__main__":
    np.set_printoptions(
        suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize
    )

    T = 50  # total simulation time
    dt = 0.0125  # simulation step
    time = int(T / dt)
    inpt = 1.0  # voltage
    ninput = np.full((time), inpt)

    num_layers = 2
    num_neurons = 100

    neurons = []
    for layer in range(num_layers):
        neuron_layer = []
        for count in range(num_neurons):
            neuron_layer.append(LIFNeuron())
        neurons.append(neuron_layer)

    stimulus_len = len(ninput)
    layer = 0
    for neuron in range(num_neurons):
        offset = random.randint(
            0, 100
        )  # Simulates stimulus starting at different times
        stimulus = np.zeros_like(ninput)
        stimulus[offset:stimulus_len] = ninput[0 : stimulus_len - offset]
        neurons[layer][neuron].generate_spike(stimulus)

    plt.plot(neurons[0][0].time, neurons[0][0].Vm)
    plt.title("{} @ {}".format(neurons[0][0].type, "0/0"))
    plt.ylabel("Membrane Potential")
    plt.xlabel("Time (msec)")
    y_min = 0
    y_max = max(neurons[0][0].Vm) * 1.2
    if y_max == 0:
        y_max = 1
    plt.ylim([y_min, y_max])
    plt.show()
    # mnist = MNIST()
    # labels, images = mnist.load_data()
    # magic, size, rows, cols = mnist.image_meta
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
    # ]
    # snn = SNN(28 * 28, kernels, 10)
    #
    # image_at = 1
    # image = images[
    #     (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    # ]
    #
    # # normalize data set
    # image = [x / 255 for x in image]
    # image = np.reshape(image, (28, 28))
    # snn.set_input(image)
    #
    #

    # guess = snn.guess(image)

    # (train_labels, train_images) = mnist.load_data()

    # pulls an image at location
    # image_at = 1
    # image = images[
    #     (image_at - 1) * rows * cols : ((image_at - 1) + 1) * rows * cols
    # ]
    # # normalize data set
    # image = np.reshape(image, (28, 28))
    #
    # output = np.zeros((26, 26))
    #
    # kernel = ""
    # kernel = convert_kernel(kernel)
    #
    # kernel_width = kernel.shape[0]
    #
    # for x in range(0, image.shape[1] - 2):
    #     for y in range(0, image.shape[0] - 2):
    #         output[x, y] = (
    #             kernel * image[x : x + kernel_width, y : y + kernel_width]
    #         ).sum()
    #
    # print(output.shape)
    #
    # plt.title("Convolution")
    # plt.imshow(output, interpolation="nearest")
    # plt.show()
    # plt.title("Original")
    # plt.imshow(image, interpolation="nearest")
    # plt.show()
