import matplotlib.pyplot as plt
import numpy as np

class Visualize:
    @staticmethod
    def plot_spikes(spike_mon_exc, spike_mon_inh, filename='data/output/spike_plot.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(spike_mon_exc.t/ms, spike_mon_exc.i, '.k', label='Excitatory')
        plt.plot(spike_mon_inh.t/ms, spike_mon_inh.i, '.r', label='Inhibitory')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_weights(weight_mon, filename='data/output/weight_plot.png'):
        plt.figure(figsize=(10, 5))
        for i in range(min(10, weight_mon.w.shape[0])):
            plt.plot(weight_mon.t/ms, weight_mon.w[i], label=f'Synapse {i}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Synaptic weight')
        plt.legend()
        plt.savefig(filename)
        plt.close()