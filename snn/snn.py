import numpy as np
from snn.neuron import LIF


class SNN(object):
    """
        inputs = 28*28 = 784
        hidden = 8,112 (26*26*12)
        output = 10 [0-9]
    """

    def __init__(self, neurons, kernels, outputs):
        self.T = 100  # 100ms time
        self.neurons = []
        for x in range(0, neurons):
            lif = LIF()  # this does not go here
            self.neurons.append(lif)

    def set_input(self, image):
        flatten = image.flatten()
        for key, cv in enumerate(flatten):
            self.neurons[key].current_v = cv

    def print_neurons(self):
        for key, neuron in enumerate(self.neurons):
            print(key, float(self.neurons[key].current_v))
