# Implementatin of the PAR model in the article : Towards Efficient and Real-Time Piano
# Transcription Using Neural Autoregressive Models

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallMLP(nn.Module):
    """
    Small fully-connected net mapping scalar condition -> vector of length out_dim.
    Matches 'three FC layers with 16 hidden units' per training details. :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        assert num_layers >= 1
        layers = []
        cur = in_dim
        # num_layers = total linear layers. Use ReLU between layers (paper: ReLU activation).
        for i in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, in_dim) -> out: (N, out_dim)
        return self.net(x)
    
class FiLMLayer(nn.Module):
    """
    Frequency-conditioned FiLM layer exactly following the paper:
    - computes condition c = k/F for k in {0..F-1}
    - two separate small nets f and g (for gamma and beta), each is SmallMLP
      with 3 FC layers and 16 hidden units per training details. 

    Input tensor shape: (B, C, F, T)
    Output tensor shape: (B, C, F, T)  (gamma*X + beta applied per-frequency row)
    """
    def __init__(self, channels: int, mlp_hidden: int = 16, mlp_layers: int = 3):
        super().__init__()
        self.channels = channels
        # f and g are separate as described in paper (no shared trunk). :contentReference[oaicite:6]{index=6}
        self.gamma_net = SmallMLP(in_dim=1, hidden_dim=mlp_hidden, out_dim=channels, num_layers=mlp_layers)
        self.beta_net  = SmallMLP(in_dim=1, hidden_dim=mlp_hidden, out_dim=channels, num_layers=mlp_layers)

        # Optional: initialize gamma bias to 1 and beta bias to 0 is an implementation choice;
        # the paper does NOT specify initialization. We therefore do NOT force that here to stick to the article. :contentReference[oaicite:7]{index=7}

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, F, T)
        returns: (B, C, F, T)
        """
        B, C, F, T = x.shape
        device = x.device
        dtype = x.dtype

        # condition vector k/F for k=0..F-1 -> shape (F,1)
        cond = torch.arange(F, dtype=dtype, device=device).unsqueeze(1) / float(F)  # (F,1)

        # compute gamma_k and beta_k for all k at once:
        # -> (F, C)
        gamma = self.gamma_net(cond)  # (F, C)
        beta  = self.beta_net(cond)   # (F, C)

        # transpose to (C, F) then reshape to (1, C, F, 1) for broadcasting over batch and time
        gamma = gamma.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1, C, F, 1)
        beta  = beta.transpose(0, 1).unsqueeze(0).unsqueeze(-1)   # (1, C, F, 1)

        return gamma * x + beta


# -------------------------
# Conv-FiLM block (Conv -> FiLM -> BatchNorm -> activation)
# -------------------------
class ConvFiLMBlock(nn.Module):
    """
    Single Conv-FiLM block: Conv2d(3x3) -> FiLM -> BatchNorm2d -> ReLU.
    BatchNorm after FiLM matches Figure 3 in paper. :contentReference[oaicite:8]{index=8}
    """
    def __init__(self, in_ch, out_ch, mlp_hidden=16, mlp_layers=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLMLayer(out_ch, mlp_hidden=mlp_hidden, mlp_layers=mlp_layers)
        self.bn = nn.BatchNorm2d(out_ch)   


    def forward(self, x):
        # x: (B, in_ch, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.film(x)
        x = self.bn(x)
        return x

    

class AcousticModule(nn.Module):
    def __init__(self, cnn_channels=48, hidden_size=768, hidden_unit_per_pitch=188):
        """
        Acoustic Module following the PAR model architecture.
        
        Args:
            cnn_channels: Number of CNN channels (48 for PAR, varies for compact)
            hidden_size: Size of hidden layer before pitch split (768 for PAR)
        """
        super().__init__()
        
        self.cnn_channels = cnn_channels
        self.hidden_size = hidden_size
        self.n_pitches = 88  # Number of Piano keys
        self.hidden_unit_per_pitch = hidden_unit_per_pitch
        # First Conv-FiLM block (1 input channel -> cnn_channels)
        self.conv_film_1 = ConvFiLMBlock(1, cnn_channels)
        self.maxpool_1 = nn.MaxPool2d((2, 1))  # Reduce freq by 2
        self.dropout_1 = nn.Dropout(0.25)
        
        # Second Conv-FiLM block (cnn_channels -> cnn_channels)
        self.conv_film_2 = ConvFiLMBlock(cnn_channels, cnn_channels)
        self.maxpool_2 = nn.MaxPool2d((2, 1))  # Reduce freq by 2
        self.dropout_2 = nn.Dropout(0.25)
        
        # Third Conv-FiLM block (cnn_channels -> cnn_channels)
        self.conv_film_3 = ConvFiLMBlock(cnn_channels, cnn_channels)
        
        # After 2 maxpools: 700 -> 350 -> 175
        freq_after_pools = 175
        combined_features = cnn_channels * freq_after_pools  # 48 * 175 = 8400
        
        # Layer normalization (applied before FC layers)
        self.layer_norm = nn.LayerNorm(combined_features)
        
        # Fully connected layers (timewise)
        self.fc1 = nn.Linear(combined_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 88 * hidden_unit_per_pitch)
        
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, 1, 700, n_frames) - log mel spectrogram
        Returns:
            pitch_features: (batch, 88, hidden_size, n_frames)
        """
        batch_size, _, _, n_frames = mel_spec.shape
        
        # First Conv-FiLM block + maxpool + dropout
        x = self.conv_film_1(mel_spec)  # (B, 48, 700, T)
        x = self.maxpool_1(x)            # (B, 48, 350, T)
        x = self.dropout_1(x)
        
        # Second Conv-FiLM block + maxpool + dropout
        x = self.conv_film_2(x)          # (B, 48, 350, T)
        x = self.maxpool_2(x)            # (B, 48, 175, T)
        x = self.dropout_2(x)
        
        # Third Conv-FiLM block (no maxpool after)
        x = self.conv_film_3(x)          # (B, 48, 175, T)
        
        # Reshape: combine channel and frequency dimensions
        # (B, C, F, T) -> (B, C*F, T)
        x = x.reshape(batch_size, -1, n_frames)  # (B, 8400, T)
        
        # Permute for timewise operations: (B, C*F, T) -> (B, T, C*F)
        x = x.permute(0, 2, 1)  # (B, T, 8400)
        
        # Layer normalization
        x = self.layer_norm(x)  # (B, T, 8400)
        
        # Timewise FC layers
        x = self.fc1(x)         # (B, T, 768)
        x = F.relu(x)
        x = self.fc2(x)         # (B, T, 88*768)
        
        # Split into 88 pitch segments
        x = x.reshape(batch_size, n_frames, 88, self.hidden_size)  # (B, T, 88, H)
        x = x.permute(0, 2, 3, 1)  # (B, 88, H, T)
        
        return x
    
    
class RecursiveContextNet(nn.Module):
    """
    Enhanced Recursive Context: embeds [last_state, duration, velocity] into 4D vector.
    Uses two FC layers with 16 hidden units as described in the paper.
    """
    def __init__(self, hidden_dim=16, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, context_input):
        """
        Args:
            context_input: (B, 88, 3) - [state, duration, velocity] per pitch
        Returns:
            (B, 88, 4) - embedded context vector
        """
        x = F.relu(self.fc1(context_input))
        x = self.fc2(x)
        return x
    

class NoteStateSequenceModule(nn.Module):
    """
    Note State Sequence Module from PAR model.
    - 88 pitch-wise LSTMs sharing the same parameters
    - Enhanced recursive context with [state, duration, velocity]
    - Outputs 5-state softmax per pitch
    """
    def __init__(self, hidden_size=768, lstm_units=48):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_units = lstm_units
        
        # Enhanced recursive context encoder (3 inputs -> 4D embedding)
        self.context_net = RecursiveContextNet(hidden_dim=16, out_dim=4)
        
        # Pitch-wise LSTMs (parameter shared across 88 pitches)
        # Input: hidden_size + 4 (context dim)
        self.lstm1 = nn.LSTM(hidden_size + 4, lstm_units, batch_first=False)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=False)
        
        # Timewise FC to 5 states (onset, re-onset, sustain, offset, off)
        self.fc_states = nn.Linear(lstm_units, 5)
        
    def forward(self, pitch_features, prev_context):
        """
        Args:
            pitch_features: (B, 88, H, T) - from acoustic module split by pitch
            prev_context: (B, 88, 3) - [last_state, duration, velocity] from previous frame
        Returns:
            state_probs: (B, 88, 5, T) - softmax probabilities over 5 states
        """
        B, num_pitches, H, T = pitch_features.shape
        
        # Encode recursive context: (B, 88, 3) -> (B, 88, 4)
        context_emb = self.context_net(prev_context)  # (B, 88, 4)
        
        # Expand for all time steps: (B, 88, 4) -> (B, 88, 4, T)
        context_emb = context_emb.unsqueeze(-1).expand(B, 88, 4, T)
        
        # Concatenate with pitch features: (B, 88, H+4, T)
        x = torch.cat([pitch_features, context_emb], dim=2)
        
        # Reshape for pitch-wise LSTM: (B, 88, H+4, T) -> (T, B*88, H+4)
        x = x.permute(3, 0, 1, 2)  # (T, B, 88, H+4)
        x = x.reshape(T, B * 88, H + 4)
        
        # Two LSTM layers (shared across all 88 pitches)
        x, _ = self.lstm1(x)  # (T, B*88, lstm_units)
        x, _ = self.lstm2(x)  # (T, B*88, lstm_units)
        
        # Timewise FC to 5 states
        x = self.fc_states(x)  # (T, B*88, 5)
        
        # Reshape back: (T, B*88, 5) -> (B, 88, 5, T)
        x = x.reshape(T, B, 88, 5)
        x = x.permute(1, 2, 3, 0)  # (B, 88, 5, T)
        
        # Softmax over states dimension
        state_probs = F.softmax(x, dim=2)
        
        return state_probs
    
class PARModel(nn.Module):
    def __init__(self, cnn_channels=48, hidden_size=768, lstm_units=48):
        super().__init__()
        self.acoustic = AcousticModule(cnn_channels, hidden_size)
        self.note_sequence = NoteStateSequenceModule(hidden_size, lstm_units)
    
    def forward(self, mel_spec, prev_context):
        """
        Args:
            mel_spec: (B, 1, 700, T) - log mel spectrogram
            prev_context: (B, 88, 3) - [state, duration, velocity]
        Returns:
            state_probs: (B, 88, 5, T) - softmax over 5 states
        """
        # Acoustic feature extraction
        pitch_features = self.acoustic(mel_spec)  # (B, 88, H, T)
        
        # Note state prediction
        state_probs = self.note_sequence(pitch_features, prev_context)
        
        return state_probs
    
# Création
model = PARModel(cnn_channels=48, hidden_size=768, lstm_units=48)

# Affichage structure
print(model)


# Calculer le nombre de paramètres par couche
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Parameters: {param.numel()}")

# Paramètre totaux
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")