import torch, math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEnc(nn.Module):
    def __init__(self, input_size: int = 1, latent_size:int=4, d_model: int = 64, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super(TransformerEnc, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)  # max length 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            # dim_feedforward=128,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_size)
        self.fc_logvar = nn.Linear(d_model, latent_size)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        B, T, D = x.shape
        assert T <= 512, f"Maximum allowed Sequence length is 512 < {T}"
        x = self.input_proj(x) * math.sqrt(self.d_model) # (B, T, D) -> (B, T, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # (B, T, D)
        xh = x[:, -1, :]  # (B, d_model)
        mu = self.fc_mu(xh)
        logvar = self.fc_logvar(xh)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar # (B, D)

class CausalForecaster(nn.Module):
    def __init__(self, input_size:int, latent_size:int, d_model:int, nhead:int, num_layers:int, dropout:float=0.1):
        super(CausalForecaster, self).__init__()
        self.D = input_size
        self.latent_size = latent_size
        self.encoders = nn.ModuleList([
            TransformerEnc(input_size=1, latent_size=latent_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout) for _ in range(self.D)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(latent_size*self.D, 1) for _ in range(self.D)
        ])

    def get_group_lasso_penalty(self):
        penalty = 0.0
        for i in range(self.D):
            weights = self.decoders[i].weight.squeeze() # Shape: (D * latent_size)
            for j in range(self.D):
                start = j * self.latent_size
                end = (j + 1) * self.latent_size
                w_block = weights[start:end]
                penalty += torch.norm(w_block, p=2) 
        return penalty

    def forward(self, x, y):
        # x: (B, T, D)
        if y.dim()==3: y = y.squeeze(1) 
        z_list = []
        mu_list = []
        logvar_list = []
        for i in range(self.D):
            x_i = x[:, :, i].unsqueeze(-1) # (B, T, 1)
            z_i, mu_i, logvar_i = self.encoders[i](x_i)
            z_list.append(z_i)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)

        z_all = torch.cat(z_list, dim=1) # (B, D * latent_size)
        
        total_mse = 0
        total_kl = 0
        for i in range(self.D):
            y_i = y[:, i].unsqueeze(-1)
            pred_i = self.decoders[i](z_all) 
            total_mse += F.mse_loss(pred_i, y_i)
            kl = -0.5 * torch.sum(1 + logvar_list[i] - mu_list[i].pow(2) - logvar_list[i].exp(), dim=1)
            total_kl += kl.mean()
        return total_mse, total_kl
    
    def get_adaptive_threshold_matrix(self):
        off_diagonal_norms = []
        with torch.no_grad():
            for i in range(self.D):
                weights = self.decoders[i].weight.data.squeeze()
                for j in range(self.D):
                    if i==j:
                        continue
                    start = j * self.latent_size
                    end = (j + 1) * self.latent_size
                    w_block = weights[start:end]
                    norm_val = torch.norm(w_block, p=2).item()
                    off_diagonal_norms.append(norm_val)
            off_diagonal_norms = np.array(off_diagonal_norms).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(off_diagonal_norms)
            centers = sorted(gmm.means_.flatten())

            noise_idx, signal_idx = np.argmin(centers), np.argmax(centers)
            low_bound, high_bound = centers[noise_idx], centers[signal_idx]
            search_space = np.linspace(low_bound, high_bound, 1000).reshape(-1, 1)
            probs = gmm.predict_proba(search_space)
            signal_dominance = probs[:, signal_idx] > probs[:, noise_idx]

            if np.any(signal_dominance):
                flip_index = np.argmax(signal_dominance)
                threshold = search_space[flip_index][0]
            else:
                threshold = (low_bound + high_bound) / 2
        return threshold

    def get_causal_matrix(self, threshold='adaptive'):
        matrix = torch.zeros(self.D, self.D)
        for i in range(self.D):
            weights = self.decoders[i].weight.data.squeeze()
            for j in range(self.D):
                start = j * self.latent_size
                end = (j + 1) * self.latent_size
                w_block = weights[start:end]
                matrix[i, j] = torch.norm(w_block, p=2)

        if threshold == 'adaptive':
            threshold = self.get_adaptive_threshold_matrix()
            print(f"Adaptive Threshold calculated: {threshold:.6f}", flush=True)
            return matrix.float(), (matrix > threshold).float(), threshold
        else:
            return matrix.float(), (matrix > threshold).float(), threshold