from brian2 import Network, second, ms
import numpy as np
from src.network import STDPNetwork

class Simulation:
    def __init__(self, params):
        self.params = params
        self.network_model = STDPNetwork(
            N_exc=params['N_exc'],
            N_inh=params['N_inh'],
            taupre=params['taupre']*ms,
            taupost=params['taupost']*ms,
            wmax=params['wmax']
        )
        self.exc_neurons, self.inh_neurons, self.synapses_exc, self.synapses_inh = self.network_model.create_network()

    def run(self, duration=1*second, input_pattern=None):
        net = Network(self.exc_neurons, self.inh_neurons, self.synapses_exc, self.synapses_inh,
                      self.network_model.spike_mon_exc, self.network_model.spike_mon_inh, self.network_model.weight_mon)

        # Apply input pattern (e.g., to simulate learning)
        if input_pattern is not None:
            self.exc_neurons.I_ext = input_pattern * mV

        net.run(duration)
        return self.network_model.spike_mon_exc, self.network_model.spike_mon_inh, self.network_model.weight_mon

    def test_recall(self, test_pattern, duration=0.5*second):
        # Reset external input and test recall
        self.exc_neurons.I_ext = test_pattern * mV
        net = Network(self.exc_neurons, self.inh_neurons, self.synapses_exc, self.synapses_inh,
                      self.network_model.spike_mon_exc, self.network_model.spike_mon_inh)
        net.run(duration)
        return self.network_model.spike_mon_exc