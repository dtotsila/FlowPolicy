import torch
import torch.nn as nn
import math
from models.embeddings import SinusoidalPositionEmbedding

class DiTPolicy(nn.Module):
    def __init__(self, action_dim, state_dim, chunk_size, hidden_dim=256, num_layers=4, num_heads=4, num_classes=None, use_vision=False):
        super().__init__()
        self.action_dim = action_dim
        self.use_vision = use_vision

        # Input Embeddings
        self.action_emb = nn.Linear(self.action_dim, hidden_dim)
        self.state_emb = nn.Linear(state_dim, hidden_dim)

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.num_classes = num_classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, hidden_dim)
        if self.use_vision:
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),   # Output: [B, 32, 20, 20]
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: [B, 64, 9, 9]
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: [B, 64, 7, 7]
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, hidden_dim)            # Project to hidden_dim
            )

        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Projection
        self.output_proj = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, noisy_actions, state, t, condition=None, image=None):
        if t.dim() > 1:
            t = t.squeeze()

        x = self.action_emb(noisy_actions) + self.pos_emb
        s_emb = self.state_emb(state)
        t_emb = self.time_emb(t)

        # Base conditioning
        cond_sum = s_emb + t_emb

        # Class conditioning
        if self.num_classes is not None:
            if condition is None:
                raise ValueError("Class conditioning is required but 'condition' was not provided.")
            c_emb = self.class_emb(condition)
            cond_sum += c_emb


        if self.use_vision:
            if image is None:
                raise ValueError("Vision conditioning is enabled but 'image' was not provided.")
            v_emb = self.vision_encoder(image)
            cond_sum += v_emb

        cond_token = cond_sum.unsqueeze(1)
        seq = torch.cat([cond_token, x], dim=1)

        out_seq = self.transformer(seq)
        action_out = out_seq[:, 1:, :]
        return self.output_proj(action_out)