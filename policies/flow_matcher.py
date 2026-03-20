import torch
import torch.nn as nn

class FlowMatcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_loss(self, x1, state, condition=None):
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        x0 = torch.randn_like(x1)

        t_expanded = t.view(B, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1

        target_flow = x1 - x0

        # Pass condition to the model
        predicted_flow = self.model(xt, state, t, condition=condition)

        loss = torch.nn.functional.mse_loss(predicted_flow, target_flow)
        return loss

    @torch.no_grad()
    def sample(self, state, chunk_size, action_dim, sampling_steps=10, condition=None):
        B = state.shape[0]
        device = state.device

        x = torch.randn(B, chunk_size, action_dim, device=device)
        dt = 1.0 / sampling_steps

        for i in range(sampling_steps):
            t = torch.full((B,), i * dt, device=device)

            # Pass condition to the model
            v = self.model(x, state, t, condition=condition)

            x = x + v * dt

        return x