import numpy as np


class LIFNeuron(object):
    def __init__(self, idx, T, dt):
        self.type = "Leaky Integrate and Fire"
        self.label = "LIF"
        self.idx = idx

        self.T = T  # simulation time [ms]
        self.dt = dt  # time step size [ms]

        self.Cm = 300.0  # membrane capacitance [pF]
        self.gl = 30.0  # leak conductance [nS]
        self.potential = 20.0  # membrane potential [mV]
        self.El = -70.0  # resting value [mV]
        self.T_rest = 3  # rest holding period [ms]
        self.T_refactor = 0  # a counter for holding after spike
        self.I = 0  # current

        self.reset()

    def set_current(self, pixel):
        lp = 101.2  # scaling factor
        # maximum constant amplitude current that doesn't trigger a spike
        l0 = 2700
        # i(k) = l0 + (k * lp)
        self.I = np.zeros(len(self.time))
        self.I[0:101] = l0 + (pixel * lp)

    def set_current_history(self, history):
        self.I = history

    def reset(self):
        self.time = np.arange(0, self.T + self.dt, self.dt)  # history of steps
        self.spikes = np.array([])  # history of spikes
        self.spike_count = 0
        self.V = np.zeros(len(self.time))  # history of voltage
        self.V[0] = self.El  # set start of voltage to resting state
        self.fired = False

    def spike_train(self):
        spikes = np.zeros(len(self.time))
        for i in range(1, len(self.time)):
            if not self.fired:
                dV = (
                    self.I[i] - self.gl * (self.V[i - 1] - self.El)
                ) / self.Cm
                self.V[i] = self.V[i - 1] + dV * self.dt

                # in case we exceed threshold
                if self.V[i] > self.potential:
                    self.V[i - 1] = self.potential
                    self.V[i] = self.El  # set to resting value
                    self.fired = True
                    spikes[i] += 1  # count spike
            else:
                if self.T_refactor < self.T_rest:
                    self.T_refactor += 1
                else:
                    self.T_refactor = 0
                    self.fired = False
                self.V[i] = self.El

        self.spikes = np.append(self.spikes, spikes)
        self.spike_count = self.spikes.sum()

    def __str__(self):
        return f"{self.label} ({self.idx})"

    __repr__ = __str__
