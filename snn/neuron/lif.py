class LIF(object):

    # https://github.com/Shikhargupta/Spiking-Neural-Network/blob/master/training/learning.py

    def __init__(self):
        self.el = -70  # resting potential mV
        self.vt = 20  # membrane potential mV
        self.gl = 30  # nS
        self.current_v = float(0.0)

    def __str__(self):
        return f"LIF MODEL with volt {float(self.current_v)}"

        #
        # Vt := action potential
        #
        # V(t) >= Vt a spike is issued
        #     and reset to El
        #
        # Defaults:
        #     El = -70mV
        #     Vt = 20mV
        #     C = 300pF
        #     gl = 30nS
