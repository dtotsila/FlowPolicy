import torch
import torch.nn as nn
import math
from models.embeddings import SinusoidalPositionEmbedding

class DiTPolicy(nn.Module):
    def __init__(self, action_dim, state_dim, chunk_size, hidden_dim=256, num_layers=4, num_heads=4, num_classes=None):
        super().__init__()

        # 1. Input Embeddings
        self.action_emb = nn.Linear(action_dim, hidden_dim)
        self.state_emb = nn.Linear(state_dim, hidden_dim)

        # Time gets a sinusoidal embedding followed by an MLP
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Optional Class Embedding for Multi-Tasking
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # Learned positional embedding for the sequence of actions
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))

        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Output Projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, noisy_actions, state, t, condition=None):
        # Format t to ensure it's [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Embed inputs and add sequence positional encoding to actions
        x = self.action_emb(noisy_actions) + self.pos_emb # [B, chunk_size, hidden_dim]
        s_emb = self.state_emb(state)                     # [B, hidden_dim]
        t_emb = self.time_emb(t).squeeze(1)               # [B, hidden_dim]

        # Combine state and time
        cond_sum = s_emb + t_emb

        # Add class conditioning if provided
        if self.num_classes is not None and condition is not None:
            c_emb = self.class_emb(condition)             # [B, hidden_dim]
            cond_sum += c_emb

        # Create single conditioning token
        cond_token = cond_sum.unsqueeze(1)                # [B, 1, hidden_dim]

        # Prepend the condition token to the action sequence
        seq = torch.cat([cond_token, x], dim=1)           # [B, 1 + chunk_size, hidden_dim]

        # Process through Transformer
        out_seq = self.transformer(seq)                   # [B, 1 + chunk_size, hidden_dim]

        # Drop the condition token and project back to action space
        action_out = out_seq[:, 1:, :]                    # [B, chunk_size, hidden_dim]
        return self.output_proj(action_out)               # [B, chunk_size, action_dim]