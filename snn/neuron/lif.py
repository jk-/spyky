import numpy as np


class LIFNeuron(object):
    def __init__(self, idx, step):
        self.type = "Leaky Integrate and Fire"
        self.label = "LIF"
        self.idx = idx

        self.dt = step
        self.t_rest = 0  # initial refactory time
        self.Vm = np.array([0])  # Activation potential Vm(t)
        self.spikes = np.array([0])  # output spikes (t)
        self.time = np.array([0])  # Time duration for the neuron

        self.t = 0  # Neuron time step
        self.Rm = 1  # Resistance (kOhm)
        self.Cm = 30  # Capacitance (uF)
        self.tau_m = self.Rm * self.Cm  # Time constant
        self.tau_ref = 3  # refractory period (ms)
        self.Vth = 20  # spike threshold "action potential"
        self.V_spike = 1  # spike delta (V)
        self.voltage = 1

    def reset(self):
        self.Vm = np.array([0])
        self.spikes = np.array([0])
        self.time = np.array([0])

    def generate_spike(self, duration, input_voltage):
        Vm = np.zeros(duration)  # potential (V) trace over time
        time = np.arange(
            int(self.t / self.dt), int(self.t / self.dt) + duration
        )
        spikes = np.zeros(duration)  # len(time)

        Vm[-1] = self.Vm[-1]

        # Vm[i] = Vm[i-1] + (-Vm[i-1] + I*Rm) / tau_m * dt
        for i in range(duration):
            if self.t > self.t_rest:
                Vm[i] = (
                    Vm[i - 1]
                    + (-Vm[i - 1] + (self.voltage * input_voltage) * self.Rm)
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

    def __str__(self):
        return f"{self.label} ({self.idx})"

    __repr__ = __str__
