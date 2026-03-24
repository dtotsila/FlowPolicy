import torch
import torch.nn as nn
from torchdiffeq import odeint


class FlowMatcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.source_dist = torch.distributions.Normal(
            torch.zeros(model.action_dim),
            torch.ones(model.action_dim)
        )

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
    def sample(self, state, chunk_size, action_dim, sampling_steps=10, condition=None, method="rk4", atol=1e-6, rtol=1e-6):
        if condition is not None:
            condition = condition.to(state.device, non_blocking=True)

        x_0 = self.source_dist.sample(
            (state.shape[0], chunk_size)).to(state.device)

        timesteps = torch.linspace(
            0.0, 1.0, sampling_steps+1, device=state.device)

        def ode_func(t, x_t):
            _t = t.expand(x_t.size(0)).unsqueeze(-1)
            v_t = self.model.forward(x_t, state, _t, condition)
            return v_t

        trajectory = odeint(ode_func, x_0, timesteps,
                            method=method, atol=atol, rtol=rtol)

        return trajectory[-1]
