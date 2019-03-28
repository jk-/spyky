import numpy as np
from snn.neuron import LIFNeuron
from snn.util import convert_kernel


class SNN(object):
    """
        inputs = 28*28 = 784
        hidden = 8,112 (26*26*12)
        output = 10 [0-9]
    """

    def __init__(self, neurons, kernels, outputs, *args, **kwargs):
        self.T = kwargs["time"]
        self.dt = kwargs["dt"]
        self.image_x = 0
        self.image_y = 0
        self.neurons = []
        self.kernels = kernels
        self.feature_maps = np.zeros([12], dtype=object)
        for idx, feature_map in enumerate(self.feature_maps):
            self.feature_maps[idx] = np.zeros((26, 26))

        self.image = None
        for x in range(0, neurons):
            self.neurons.append(LIFNeuron(x, self.dt))

    def guess(self, image):
        print("* Setting up neurons")
        self.image_x = image.shape[1]
        self.image_y = image.shape[0]
        self.image_idx = np.array(
            [x for x in range(0, image.shape[0] * image.shape[1])]
        ).reshape(image.shape[0], image.shape[1])
        flatten = image.flatten()
        for key, pixel_voltage in enumerate(flatten):
            self.neurons[key].input_voltage = pixel_voltage
        self.convolution()

    # 2d convolution
    def convolution(self):
        # loop through the image_idx with convolution of kernel
        # with a stride of 1. Get stride of image_idx
        # then flatten the stride
        # loop through flattened stride, generate a spike of neurons
        # grab the get_neurons_that_spiked
        # set the amount of spiked to feaature_map[key] at pixel with number

        print("** 2D Convolution")
        feature_map_idx = 0
        for kernel_str in self.kernels:
            kernel = convert_kernel(kernel_str, 3)
            kernel_width = kernel.shape[0]

            strides = 0
            for x in range(0, self.image_idx.shape[1] - 2):
                for y in range(0, self.image_idx.shape[0] - 2):
                    neurons_to_fire = self.image_idx[
                        x : x + kernel_width, y : y + kernel_width
                    ]
                    neurons_to_fire = neurons_to_fire.flatten()
                    count_neuron_spikes = 0
                    for idx, neuron_idx in enumerate(neurons_to_fire):
                        self.neurons[neuron_idx].generate_spike(self.T)
                        count_neuron_spikes += self.neurons[
                            neuron_idx
                        ].spikes.sum()
                    self.feature_maps[feature_map_idx][x][
                        y
                    ] = count_neuron_spikes
                    strides += 1
            feature_map_idx += 1
