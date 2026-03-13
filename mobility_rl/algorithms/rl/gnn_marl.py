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
        self.conv1 = GATConv(node_in_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False) # Average heads for final embedding
        
        # 2. Recurrent Memory Block (GRU) for Temporal Dependency (POMDP Fading)
        self.memory_block = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 3. Dueling Q-Architecture Streams
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
        
        # Apply Temporal Memory (GRU)
        # GRU expects shape (batch, seq_len, features).
        # We treat each node as an independent sequence of length 1 per graph transition.
        x_seq = x.unsqueeze(1) # Shape: (N_UAVs, 1, hidden_dim)
        x_out, _ = self.memory_block(x_seq)
        x = x_out.squeeze(1)   # Shape: (N_UAVs, hidden_dim)
        
        # Dueling Bifurcation
        values = self.value_stream(x)           # Shape: (N_UAVs, 1)
        advantages = self.advantage_stream(x)   # Shape: (N_UAVs, num_actions)
        
        # Recombine: Q(s, a) = V(s) + ( A(s, a) - mean(A(s, a)) )
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class QMIXMixer(nn.Module):
    """
    QMIX monotonic mixing network for centralised training (CTDE).

    Takes one Q-value per agent (the Q of the action they took) and a global
    state, and combines them into a single Q_total via hyper-networks whose
    weights are forced non-negative — guaranteeing monotonicity:
        dQ_total / dQ_i  >=  0   for all agents i

    This lets each UAV act greedily on its own Q-values at execution time
    while the team is trained against a shared Q_total during centralised
    training — solving the credit-assignment problem without changing the
    per-agent forward pass at all.

    Args:
        num_agents  : number of UAVs (N).
        state_dim   : dimension of the global state fed to the hyper-networks.
                      Simplest choice: node_in_dim * num_agents (concat all node feats).
        mixer_embed : hidden size of the hyper-networks (default 32 is sufficient).

    Usage (training loop only):
        mixer    = QMIXMixer(num_agents=6, state_dim=48, mixer_embed=32)
        q_total  = mixer(agent_qs, global_state)   # (batch, 1)
        loss     = F.mse_loss(q_total, target_q_total)
    """

    def __init__(self, num_agents: int, state_dim: int, mixer_embed: int = 32):
        super(QMIXMixer, self).__init__()

        self.num_agents  = num_agents
        self.mixer_embed = mixer_embed

        # Hyper-net W1: state -> weights (num_agents -> mixer_embed), kept >= 0
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, num_agents * mixer_embed),
        )

        # Hyper-net W2: state -> weights (mixer_embed -> 1), kept >= 0
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, mixer_embed),
        )

        # Biases (no non-negativity constraint needed)
        self.hyper_b1 = nn.Linear(state_dim, mixer_embed)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, 1),
        )

    def forward(self, agent_qs, global_state):
        """
        Args:
            agent_qs     : (batch, num_agents)  — chosen-action Q per agent.
            global_state : (batch, state_dim)   — global observation / concat node feats.
        Returns:
            q_total      : (batch, 1)
        """
        B = agent_qs.size(0)

        # Layer 1 — abs() enforces non-negative weights -> monotonicity
        w1 = torch.abs(self.hyper_w1(global_state))        # (B, N * embed)
        w1 = w1.view(B, self.num_agents, self.mixer_embed)  # (B, N, embed)
        b1 = self.hyper_b1(global_state).unsqueeze(1)       # (B, 1, embed)

        qs     = agent_qs.unsqueeze(1)                      # (B, 1, N)
        hidden = F.elu(torch.bmm(qs, w1) + b1)             # (B, 1, embed)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(global_state))        # (B, embed)
        w2 = w2.unsqueeze(2)                                # (B, embed, 1)
        b2 = self.hyper_b2(global_state)                    # (B, 1)

        q_total = torch.bmm(hidden, w2).squeeze(2) + b2    # (B, 1)
        return q_total