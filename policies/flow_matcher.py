import torch
import torch.nn as nn

class FlowMatcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_loss(self, x_1, state):
        """Computes the Flow Matching MSE loss during training."""
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample random noise x_0
        x_0 = torch.randn_like(x_1)

        # Sample random time t in [0, 1]
        t = torch.rand((batch_size, 1), device=device)

        # Reshape t to match x_0 dimensions for broadcasting : [B, 1, 1]
        t_expanded = t.unsqueeze(-1)

        # Compute x_t: straight line path between x_0 and x_1
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Compute the target vector field (velocity)
        target_velocity = x_1 - x_0

        # Predict the vector field using the neural network
        predicted_velocity = self.model(x_t, state, t)

        # MSE loss
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(self, state, chunk_size, action_dim, sampling_steps=50):
        """Generate actions from pure nouse using an Euler ODE solver."""
        batch_size = state.shape[0]
        device = state.device

        # Start with pure gaussian noise at t=9
        x_t = torch.randn((batch_size, chunk_size, action_dim), device=device)

        # Set the time step size (dt
        dt = 1.0 / sampling_steps

        # Euler integration loop
        for i in range(sampling_steps):
            # Current time t
            t = torch.ones((batch_size, 1), device=device) * (i / sampling_steps)

            # Predict Velocity
            v_t = self.model(x_t, state, t)

            # Integrate
            x_t = x_t + v_t * dt

        # Final x_t at t =1 is our generated action chunk
        return x_t