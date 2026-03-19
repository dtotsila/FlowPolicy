import numpy as np

class BatchedTemporalEnsembler:
    def __init__(self, exp_weight=0.01):
        self.exp_weight = exp_weight
        self.buffer = {}

    def update(self, current_t, batched_chunk):
        # batched_chunk shape: (B, chunk_size, action_dim)
        for i in range(batched_chunk.shape[1]):
            t = current_t + i
            weight = np.exp(-self.exp_weight * i)
            if t not in self.buffer:
                self.buffer[t] = []
            self.buffer[t].append((batched_chunk[:, i, :], weight))

    def get_action(self, t):
        if t not in self.buffer:
            raise ValueError(f"No predictions found for timestep {t}")

        preds = self.buffer[t]
        weighted_sum = sum(p[0] * p[1] for p in preds)
        total_weight = sum(p[1] for p in preds)
        return weighted_sum / total_weight # Shape: (B, action_dim)