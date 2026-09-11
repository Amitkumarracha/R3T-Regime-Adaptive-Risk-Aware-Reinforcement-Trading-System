"""
Walk-forward validation.
"""
import numpy as np

class WalkForwardValidator:
    def __init__(self, n_splits=5, train_window=252*2, test_window=252):
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        
    def create_windows(self, n_samples):
        windows = []
        start_idx = 0
        for _ in range(self.n_splits):
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            if test_end > n_samples:
                break
                
            windows.append((
                (start_idx, train_end),
                (train_end, test_end)
            ))
            start_idx += self.test_window # slide forward
            
        return windows
        
    def summarize(self, metrics_list):
        summary = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                vals = [m[key] for m in metrics_list]
                summary[f"{key}_mean"] = np.mean(vals)
                summary[f"{key}_std"] = np.std(vals)
        return summary

