# Implementatin of the PAR model in the article : Towards Efficient and Real-Time Piano
# Transcription Using Neural Autoregressive Models

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        assert num_layers >= 1
        layers = []
        cur = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class FiLMLayer(nn.Module):
    def __init__(self, channels: int, mlp_hidden: int = 16, mlp_layers: int = 3):
        super().__init__()
        self.channels = channels
        self.gamma_net = SmallMLP(in_dim=1, hidden_dim=mlp_hidden, out_dim=channels, num_layers=mlp_layers)
        self.beta_net  = SmallMLP(in_dim=1, hidden_dim=mlp_hidden, out_dim=channels, num_layers=mlp_layers)

    def forward(self, x: torch.Tensor):
        B, C, F, T = x.shape
        device = x.device
        dtype = x.dtype

        # condition vector k/F for k=0..F-1 -> shape (F,1)
        cond = torch.arange(F, dtype=dtype, device=device).unsqueeze(1) / float(F)  # (F,1)

        # compute gamma_k and beta_k for all k at once:
        gamma = self.gamma_net(cond)  # (F, C)
        beta  = self.beta_net(cond)   # (F, C)

        # transpose to (C, F) then reshape to (1, C, F, 1) for broadcasting over batch and time
        gamma = gamma.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1, C, F, 1)
        beta  = beta.transpose(0, 1).unsqueeze(0).unsqueeze(-1)   # (1, C, F, 1)

        return gamma * x + beta

class ConvFiLMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mlp_hidden=16, mlp_layers=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLMLayer(out_ch, mlp_hidden=mlp_hidden, mlp_layers=mlp_layers)
        self.bn = nn.BatchNorm2d(out_ch)   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.film(x)
        x = self.bn(x)
        return x

class AcousticModule(nn.Module):
    def __init__(self, cnn_channels=48, hidden_size=768, hidden_unit_per_pitch=188):
        super().__init__()
        
        self.cnn_channels = cnn_channels
        self.hidden_size = hidden_size
        self.n_pitches = 88
        self.hidden_unit_per_pitch = hidden_unit_per_pitch
        
        # Conv-FiLM blocks
        self.conv_film_1 = ConvFiLMBlock(1, cnn_channels)
        self.maxpool_1 = nn.MaxPool2d((2, 1))
        self.dropout_1 = nn.Dropout(0.25)
        
        self.conv_film_2 = ConvFiLMBlock(cnn_channels, cnn_channels)
        self.maxpool_2 = nn.MaxPool2d((2, 1))
        self.dropout_2 = nn.Dropout(0.25)
        
        self.conv_film_3 = ConvFiLMBlock(cnn_channels, cnn_channels)
        
        # After 2 maxpools: 700 -> 350 -> 175
        freq_after_pools = 175
        combined_features = cnn_channels * freq_after_pools  # 48 * 175 = 8400
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(combined_features)
        
        # Timewise FC layers
        self.fc1 = nn.Linear(combined_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 88 * hidden_unit_per_pitch)
        
    def forward(self, mel_spec):
        batch_size, _, _, n_frames = mel_spec.shape
        
        # First Conv-FiLM block + maxpool + dropout
        x = self.conv_film_1(mel_spec)  # (B, 48, 700, T)
        x = self.maxpool_1(x)            # (B, 48, 350, T)
        x = self.dropout_1(x)
        
        # Second Conv-FiLM block + maxpool + dropout
        x = self.conv_film_2(x)          # (B, 48, 350, T)
        x = self.maxpool_2(x)            # (B, 48, 175, T)
        x = self.dropout_2(x)
        
        # Third Conv-FiLM block
        x = self.conv_film_3(x)          # (B, 48, 175, T)
        
        # Reshape and permute
        x = x.reshape(batch_size, -1, n_frames)  # (B, 8400, T)
        x = x.permute(0, 2, 1)  # (B, T, 8400)
        
        # Layer normalization
        x = self.layer_norm(x)  # (B, T, 8400)
        
        # Timewise FC layers
        x = self.fc1(x)         # (B, T, 768)
        x = F.relu(x)
        x = self.fc2(x)         # (B, T, 88*768)
        
        # Split into 88 pitch segments
        x = x.reshape(batch_size, n_frames, 88, self.hidden_unit_per_pitch)  # (B, T, 88, H)
        x = x.permute(0, 2, 3, 1)  # (B, 88, H, T)
        
        return x
    

class RecursiveContextNet(nn.Module):
    """
    Enhanced Recursive Context with TWO hidden layers as described in paper:
    "embedded with two layers of small fully connected layer with 16 units,
    and projected into 4 dimension with a linear layer"
    """
    def __init__(self, hidden_dim=16, out_dim=4):
        super().__init__()
        # Two FC layers with 16 units: 3 → 16 → ReLU → 16 → ReLU → 4
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, context_input):
        """
        Args:
            context_input: (B, 88, 3) - [state, duration, velocity] per pitch
                         duration and velocity should be normalized [0,1]
        Returns:
            (B, 88, 4) - embedded context vector
        """
        x = F.relu(self.fc1(context_input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NoteStateSequenceModule(nn.Module):
    """
    Note State Sequence Module with teacher forcing support.
    """
    def __init__(self, hidden_unit_per_pitch=188, lstm_units=48):
        super().__init__()
        self.hidden_unit_per_pitch = hidden_unit_per_pitch
        self.lstm_units = lstm_units
        self.n_states = 5  # onset, re-onset, sustain, offset, off
        
        # Enhanced recursive context encoder
        self.context_net = RecursiveContextNet(hidden_dim=16, out_dim=4)
        
        # Pitch-wise LSTMs (parameter shared)
        self.lstm1 = nn.LSTM(hidden_unit_per_pitch + 4, lstm_units, batch_first=False)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=False)
        
        # Timewise FC to 5 states
        self.fc_states = nn.Linear(lstm_units, self.n_states)
        
    def forward_step(self, pitch_features_t, prev_context):
        """
        Single step forward for autoregressive inference.
        
        Args:
            pitch_features_t: (B, 88, H) - acoustic features at time t
            prev_context: (B, 88, 3) - [state, duration, velocity] from previous frame
        Returns:
            state_logits_t: (B, 88, 5) - logits for state prediction
            context_emb_t: (B, 88, 4) - embedded context for next step
        """
        # Encode recursive context: (B, 88, 3) -> (B, 88, 4)
        context_emb = self.context_net(prev_context)  # (B, 88, 4)
        
        # Concatenate with pitch features: (B, 88, H+4)
        x = torch.cat([pitch_features_t, context_emb], dim=2)  # (B, 88, H+4)
        
        # Reshape for LSTM: (88, B, H+4)
        x = x.permute(1, 0, 2)  # (88, B, H+4)
        
        # Two LSTM layers
        x, _ = self.lstm1(x)  # (88, B, lstm_units)
        x, _ = self.lstm2(x)  # (88, B, lstm_units)
        
        # Timewise FC to 5 states: (88, B, 5)
        x = self.fc_states(x)
        
        # Reshape back: (B, 88, 5)
        state_logits = x.permute(1, 0, 2)  # (B, 88, 5)
        
        return state_logits, context_emb
        
    def forward(self, pitch_features, prev_context, teacher_forcing=True, target_states=None):
        """
        Args:
            pitch_features: (B, 88, H, T) - from acoustic module
            prev_context: (B, 88, 3) - initial context (all zeros for first frame)
            teacher_forcing: bool - whether to use ground truth context
            target_states: (B, 88, T) - ground truth states for teacher forcing
        Returns:
            state_probs: (B, 88, 5, T) - softmax probabilities over 5 states
        """
        B, num_pitches, H, T = pitch_features.shape
        
        # List to store outputs at each timestep
        state_logits_list = []
        context_emb_list = []
        
        # Current context for autoregressive generation
        current_context = prev_context
        
        for t in range(T):
            # Get features at time t: (B, 88, H)
            pitch_features_t = pitch_features[..., t]
            
            # Forward step
            state_logits_t, context_emb_t = self.forward_step(pitch_features_t, current_context)
            state_logits_list.append(state_logits_t)
            context_emb_list.append(context_emb_t)
            
            # Update context for next step
            if teacher_forcing and target_states is not None:
                # Use ground truth states for next context
                # Extract state, duration, velocity from target_states
                # This requires additional processing - for now using predicted states
                pass
            
            # During inference or without teacher forcing: use predicted states
            # For now, we need to compute the next context from predictions
        
        # Stack outputs: (B, 88, 5, T)
        state_logits = torch.stack(state_logits_list, dim=-1)
        
        # Apply softmax
        state_probs = F.softmax(state_logits, dim=2)
        
        return state_probs

class VelocityModel(nn.Module):
    """
    Separate velocity prediction model as described in paper:
    "For estimating note velocity, we used another model with the same 
    specifications as the note model, as done in the onsets and frames model [4]. 
    The LSTM of the velocity model takes both the output of the acoustic 
    module from the note model and the velocity model."
    """
    def __init__(self, hidden_unit_per_pitch=188, lstm_units=48):
        super().__init__()
        self.hidden_unit_per_pitch = hidden_unit_per_pitch
        self.lstm_units = lstm_units
        
        # Velocity model has its own LSTM processing velocity features
        self.lstm1 = nn.LSTM(hidden_unit_per_pitch, lstm_units, batch_first=False)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=False)
        
        # Predict velocity (0-127 range, but normalized to [0,1] in training)
        self.fc_velocity = nn.Linear(lstm_units, 1)
        
    def forward(self, acoustic_features, note_state_probs=None):
        """
        Args:
            acoustic_features: (B, 88, H, T) - from acoustic module
            note_state_probs: (B, 88, 5, T) - optional, from note model for joint processing
        Returns:
            velocity_pred: (B, 88, 1, T) - predicted velocity (normalized [0,1])
        """
        B, num_pitches, H, T = acoustic_features.shape
        
        # Reshape for LSTM: (T, B*88, H)
        x = acoustic_features.permute(3, 0, 1, 2)  # (T, B, 88, H)
        x = x.reshape(T, B * 88, H)
        
        # If note_state_probs provided, we could concatenate - but paper says
        # "The LSTM of the velocity model takes both the output of the acoustic 
        # module from the note model and the velocity model"
        # This suggests separate processing - implementing separate velocity LSTM
        
        # Two LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Predict velocity: (T, B*88, 1)
        velocity = self.fc_velocity(x)
        
        # Reshape back: (B, 88, 1, T)
        velocity = velocity.reshape(T, B, 88, 1)
        velocity = velocity.permute(1, 2, 3, 0)  # (B, 88, 1, T)
        
        # Sigmoid activation for normalized velocity [0,1]
        velocity = torch.sigmoid(velocity)
        
        return velocity

class PARModel(nn.Module):
    def __init__(self, cnn_channels=48, hidden_size=768, lstm_units=48, hidden_unit_per_pitch=188):
        super().__init__()
        self.n_pitches = 88
        self.n_states = 5
        
        # Shared acoustic module
        self.acoustic = AcousticModule(cnn_channels, hidden_size, hidden_unit_per_pitch)
        
        # Note state prediction model
        self.note_sequence = NoteStateSequenceModule(hidden_unit_per_pitch, lstm_units)
        
        # Separate velocity prediction model
        self.velocity_model = VelocityModel(hidden_unit_per_pitch, lstm_units)
        
        # For computing context: max duration in seconds (paper: clipped to 5 secs)
        self.max_duration_seconds = 5.0
        
    def forward(self, mel_spec, prev_context, teacher_forcing=True, target_states=None, 
                target_durations=None, target_velocities=None):
        """
        Args:
            mel_spec: (B, 1, 700, T) - log mel spectrogram
            prev_context: (B, 88, 3) - initial context [state, duration, velocity]
            teacher_forcing: bool - use ground truth for autoregressive context
            target_states: (B, 88, T) - ground truth states for teacher forcing
            target_durations: (B, 88, T) - ground truth durations for teacher forcing
            target_velocities: (B, 88, T) - ground truth velocities for teacher forcing
        Returns:
            state_probs: (B, 88, 5, T) - note state probabilities
            velocity_pred: (B, 88, 1, T) - predicted velocity
        """
        # Extract acoustic features
        pitch_features = self.acoustic(mel_spec)  # (B, 88, H, T)
        
        # Predict note states (with teacher forcing)
        state_probs = self.note_sequence(
            pitch_features, prev_context, 
            teacher_forcing=teacher_forcing, 
            target_states=target_states
        )  # (B, 88, 5, T)
        
        # Predict velocity using separate model
        # The paper mentions the velocity model takes both acoustic module outputs,
        # but for simplicity we'll use the same acoustic features
        velocity_pred = self.velocity_model(pitch_features, state_probs)  # (B, 88, 1, T)
        
        return state_probs, velocity_pred
    
    def init_context(self, batch_size, device):
        """Initialize context for autoregressive generation"""
        # [state (0=off), duration (0), velocity (0)]
        return torch.zeros(batch_size, self.n_pitches, 3, device=device)

    
