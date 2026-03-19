import torch

class Normalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        """Fits the mean and std to a given tensor of shape (N, ...)."""
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-6 # Prevent division by zero

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class DictNormalizer:
    def __init__(self):
        self.normalizers = {}

    def fit(self, key, data):
        if key not in self.normalizers:
            self.normalizers[key] = Normalizer()
        self.normalizers[key].fit(data)

    def normalize(self, key, x):
        return self.normalizers[key].normalize(x)

    def denormalize(self, key, x):
        return self.normalizers[key].denormalize(x)

    def state_dict(self):
        return {k: norm.state_dict() for k, norm in self.normalizers.items()}

    def load_state_dict(self, state_dict):
        self.normalizers = {}
        for key, stats in state_dict.items():
            self.normalizers[key] = Normalizer()
            self.normalizers[key].load_state_dict(stats)


def build_normalizer(train_dataset) -> DictNormalizer:
    """Fit a DictNormalizer on training data only."""
    normalizer = DictNormalizer()
    all_states = torch.stack([s for s, _ in train_dataset])
    all_actions = torch.stack([a for _, a in train_dataset])
    normalizer.fit("state", all_states)
    normalizer.fit("action", all_actions)
    return normalizer
