from brian2 import NeuronGroup, Synapses, StateMonitor, SpikeMonitor, ms, mV, second
import numpy as np

class STDPNetwork:
    def __init__(self, N_exc=100, N_inh=25, taupre=20*ms, taupost=20*ms, wmax=0.1):
        self.N_exc = N_exc  # Number of excitatory neurons
        self.N_inh = N_inh  # Number of inhibitory neurons
        self.taupre = taupre
        self.taupost = taupost
        self.wmax = wmax

        # Neuron model parameters
        self.neuron_eqs = '''
        dv/dt = (-v + I_ext) / (10*ms) : volt
        I_ext : volt
        '''

        # STDP parameters
        self.stdp_eqs = '''
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        '''

        self.on_pre = '''
        v_post += w * mV
        apre += 0.01
        w = clip(w + apost, 0, wmax)
        '''

        self.on_post = '''
        apost += -0.01 * (taupre/taupost)
        w = clip(w + apre, 0, wmax)
        '''

    def create_network(self):
        # Create neuron groups
        self.exc_neurons = NeuronGroup(self.N_exc, self.neuron_eqs, threshold='v>20*mV', reset='v=0*mV', method='euler')
        self.inh_neurons = NeuronGroup(self.N_inh, self.neuron_eqs, threshold='v>20*mV', reset='v=0*mV', method='euler')

        # Create synapses with STDP
        self.synapses_exc = Synapses(self.exc_neurons, self.exc_neurons, model=self.stdp_eqs,
                                     on_pre=self.on_pre, on_post=self.on_post)
        self.synapses_exc.connect(p=0.1)  # Sparse connectivity
        self.synapses_exc.w = 'rand() * wmax'

        # Inhibitory synapses (no STDP)
        self.synapses_inh = Synapses(self.inh_neurons, self.exc_neurons, model='w : 1', on_pre='v_post -= w * mV')
        self.synapses_inh.connect(p=0.2)
        self.synapses_inh.w = '0.05'

        # Monitors
        self.spike_mon_exc = SpikeMonitor(self.exc_neurons)
        self.spike_mon_inh = SpikeMonitor(self.inh_neurons)
        self.weight_mon = StateMonitor(self.synapses_exc, 'w', record=True)

        return self.exc_neurons, self.inh_neurons, self.synapses_exc, self.synapses_inh