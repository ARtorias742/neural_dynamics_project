import numpy as np
from scipy.stats import pearsonr

class Analysis:
    @staticmethod
    def calculate_recall_accuracy(spike_mon, target_pattern):
        # Simplified recall accuracy: correlation between spike pattern and target
        spike_counts = np.bincount(spike_mon.i, minlength=len(target_pattern))
        if np.sum(spike_counts) == 0 or np.sum(target_pattern) == 0:
            return 0.0
        return pearsonr(spike_counts, target_pattern)[0]

    @staticmethod
    def weight_stability(weight_mon):
        # Calculate mean weight change over time
        weights = weight_mon.w
        weight_changes = np.diff(weights, axis=1)
        return np.mean(np.abs(weight_changes), axis=1)