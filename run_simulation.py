from src.simulation import Simulation
from src.analysis import Analysis
from src.visualize import Visualize
import numpy as np
from brian2 import ms, second

def main():
    # Simulation parameters
    params = {
        'N_exc': 100,
        'N_inh': 25,
        'taupre': 20,
        'taupost': 20,
        'wmax': 0.1
    }

    # Create a simple input pattern (e.g., first 10 neurons active)
    input_pattern = np.zeros(params['N_exc'])
    input_pattern[:10] = 10  # Strong input to first 10 neurons

    # Run learning phase
    sim = Simulation(params)
    spike_mon_exc, spike_mon_inh, weight_mon = sim.run(duration=1*second, input_pattern=input_pattern)

    # Test recall with partial input
    test_pattern = np.zeros(params['N_exc'])
    test_pattern[:5] = 10  # Partial cue
    recall_spikes = sim.test_recall(test_pattern, duration=0.5*second)

    # Analyze results
    recall_accuracy = Analysis.calculate_recall_accuracy(recall_spikes, input_pattern)
    weight_stability = Analysis.weight_stability(weight_mon)
    print(f"Recall accuracy: {recall_accuracy:.2f}")
    print(f"Average weight stability: {np.mean(weight_stability):.4f}")

    # Visualize results
    Visualize.plot_spikes(spike_mon_exc, spike_mon_inh)
    Visualize.plot_weights(weight_mon)

if __name__ == "__main__":
    main()