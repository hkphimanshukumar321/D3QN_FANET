import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MAGAT_D3QN_QNetwork(nn.Module):
    """
    Multi-Agent Graph Attention Dueling Q-Network (MAGAT-D3QN).
    
    1. Spatio-Temporal Attention: Multi-Headed Graph Attention Networks (GATConv) 
       learn dynamic importance weights for neighboring UAVs (Channel/Topology Awareness).
    2. Dueling Architecture: Action-Value outputs are bifurcated into a State-Value 
       stream V(s) and an Advantage stream A(s, a).
       
    - node_in_dim: size of local state per UAV (8 here).
    - hidden_dim: size of message passing embedding.
    - num_actions: discrete MAC actions (2: TDMA/CSMA).
    - heads: number of attention heads for GAT layers.
    """
    def __init__(self, node_in_dim=8, hidden_dim=64, num_actions=2, heads=4):
        super(MAGAT_D3QN_QNetwork, self).__init__()
        
        # 1. Multi-Head Graph Attention Layers (replaces basic GCNConv)
        # We concatenate the heads, so output of conv1 is hidden_dim * heads
        self.conv1 = GATConv(node_in_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False) # Average heads for final embedding
        
        # 2. Dueling Q-Architecture Streams
        # Value Stream V(s): How good is the graph state globally?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage Stream A(s, a): How good is each MAC action relatively?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x, edge_index):
        # Apply Multi-Head Attention Spatial Encoders
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU is standard practice for GAT
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Dueling Bifurcation
        values = self.value_stream(x)           # Shape: (N_UAVs, 1)
        advantages = self.advantage_stream(x)   # Shape: (N_UAVs, num_actions)
        
        # Recombine: Q(s, a) = V(s) + ( A(s, a) - mean(A(s, a)) )
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
