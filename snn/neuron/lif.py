import numpy as np

# https://github.com/Shikhargupta/Spiking-Neural-Network/blob/master/training/learning.py
class LIFNeuron(object):
    def __init__(self):
        self.type = "Leaky Integrate and Fire"
        self.label = "LIF"

        self.dt = 0.0125  # simulation time

        self.t_rest = 0  # initial refactory time
        self.Vm = np.array([0])  # Activation potential Vm(t)
        self.spikes = np.array([0])  # output spikes
        self.time = np.array([0])  # Time duration for the neuron

        self.t = 0  # Neuron time step
        self.Rm = 1  # Resistance (kOhm)
        self.Cm = 10  # Capacitance (uF)
        self.tau_m = self.Rm * self.Cm  # Time constant
        self.tau_ref = 4  # refractory period (ms)
        self.Vth = 0.75  # spike threshold
        self.V_spike = 1  # spike delta (V)

    def generate_spike(self, neuron_input):
        duration = len(neuron_input)
        Vm = np.zeros(duration)  # potential (V) trace over time
        time = np.arange(
            int(self.t / self.dt), int(self.t / self.dt) + duration
        )
        spikes = np.zeros(duration)  # len(time)

        Vm[-1] = self.Vm[-1]

        # V(t) = Vm-1 + (-Vm-1 + nP * Rm) / tau_m * d(t)
        for i in range(duration):
            if self.t > self.t_rest:
                Vm[i] = (
                    Vm[i - 1]
                    + (-Vm[i - 1] + neuron_input[i - 1] * self.Rm)
                    / self.tau_m
                    * self.dt
                )

                if Vm[i] >= self.Vth:
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref

            self.t += self.dt

        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)
