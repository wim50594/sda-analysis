import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_users: int | tuple[int, int], d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        if isinstance(num_users, tuple):
            self.n_senders, self.n_receivers = num_users
        else:
            self.n_senders, self.n_receivers = num_users, num_users

        self.d_model = d_model

        # Eingabe-Embedding für X
        self.input_embed = nn.Linear(self.n_senders, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Projektion: für jeden Nutzer Vorhersage an alle Empfänger
        self.P_proj = nn.Linear(d_model, self.n_senders * self.n_receivers)

    def forward(self, x):
        """
        x: [batch_size, seq_len, num_users] (one-hot Sendeaktionen)
        return:
            Y_pred: [B, T, N], vorhergesagte empfangene Nachrichten
            P_t: [B, T, N, N], dynamische Wahrscheinlichkeitsmatrizen pro Runde
        """
        B, N_senders = x.shape

        # --- Input-Embedding ---
        x_embed = self.input_embed(x)     # [B, d_model]
        x_embed = x_embed.unsqueeze(1)           # [B, T=1, d_model]

        # --- Transformer ---
        h = self.transformer(x_embed)            # [B, T, d_model]
        h = self.layer_norm(h)

        # --- Dynamisches P_t ---
        P_flat = self.P_proj(h)                  # [B, T, N_senders * N_receivers]
        P_t = P_flat.view(B, 1, self.n_senders, self.n_receivers)
        P_t = F.softmax(P_t, dim=-1)            # Wahrscheinlichkeiten pro Sender

        # --- Y_pred = X @ P_t ---
        P_t_squeezed = P_t.squeeze(1)
        Y_pred = torch.bmm(x.unsqueeze(1), P_t_squeezed).squeeze(1)  # [B, N_receivers]

        return Y_pred, P_t