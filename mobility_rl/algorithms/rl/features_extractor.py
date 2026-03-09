import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class MCAFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom BaseFeaturesExtractor for the Dict observation space.
    - Branch A: Processes the 1D scalar vector (num_scalars) through MLP.
    - Branch B: Processes the 2D history sequence (history_len, num_scalars) through Conv1D.
    - Fuses embeddings and outputs a fixed size feature vector for the RL head.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We assume features_dim is the combined size of the fusion layer output.
        super(MCAFeaturesExtractor, self).__init__(observation_space, features_dim)

        scalar_dim = observation_space.spaces["scalars"].shape[0]
        history_shape = observation_space.spaces["history"].shape # (T, num_scalars)
        
        # Branch A: Scalars Encoder (MLP)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Output dim = 64

        # Branch B: Temporal Sequence Encoder (1D CNN)
        # Input to Conv1D is (batch_size, channels, length) so we need to transpose (batch, T, feat)
        # Conv1D views 'channels' as features, 'length' as time steps. 
        # Wait, standard is (batch, in_channels, seq_len). 
        # Our history shape: (T, num_scalars). We want to convolve over time, so channels = num_scalars, length = T.
        in_channels = history_shape[1]
        seq_len = history_shape[0]

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate Flatten output size dynamically
        # Conv1d output length = (seq_len - kernel_size) / stride + 1 
        #                      = (5 - 2) / 1 + 1 = 4
        # Flat size = out_channels * out_len = 32 * 4 = 128
        out_len = ((seq_len - 2) // 1) + 1
        flat_size = 32 * out_len
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU()
        )
        # Output dim = 128
        
        # Fusion dimensionality check
        combined_dim = 64 + 128
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        scalars = observations["scalars"]
        history = observations["history"]

        # Branch A
        scalar_features = self.scalar_encoder(scalars)
        
        # Branch B
        # history is (batch, T, features). PyTorch Conv1d needs (batch, features, T)
        history = history.transpose(1, 2)
        temporal_conv = self.temporal_encoder(history)
        temporal_features = self.temporal_proj(temporal_conv)
        
        # Fusion
        combined = torch.cat([scalar_features, temporal_features], dim=1)
        return self.fusion(combined)
