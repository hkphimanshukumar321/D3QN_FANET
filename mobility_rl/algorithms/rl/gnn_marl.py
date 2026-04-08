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

    - node_in_dim: size of local state per UAV.
    - hidden_dim: size of message passing embedding.
    - num_actions: discrete MAC actions (2: TDMA/CSMA).
    - heads: number of attention heads for GAT layers.
    """

    def __init__(
        self,
        node_in_dim=8,
        hidden_dim=64,
        num_actions=2,
        heads=4,
        use_graph=True,
        use_attention=True,
        use_gru=True,
        use_burst_history=True,
    ):
        super(MAGAT_D3QN_QNetwork, self).__init__()
        self.use_graph = use_graph
        self.use_attention = use_attention
        self.use_gru = use_gru
        self.use_burst_history = use_burst_history

        # 1. Spatial encoder
        if self.use_graph and self.use_attention:
            self.conv1 = GATConv(node_in_dim, hidden_dim, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
            self.pre_mlp = None
            self.post_mlp = None
        else:
            self.conv1 = None
            self.conv2 = None
            self.pre_mlp = nn.Linear(node_in_dim, hidden_dim)
            self.post_mlp = nn.Linear(hidden_dim, hidden_dim)

        # 2. Recurrent Memory Block (GRU) for Temporal Dependency
        if self.use_gru:
            self.memory_block = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        else:
            self.memory_block = None

        # Persistent hidden state for temporal memory across environment steps
        self.hidden_state = None

        # 3. Dueling Q-Architecture Streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def reset_memory(self):
        """
        Reset GRU hidden state.
        Call this at the start of every new episode.
        """
        self.hidden_state = None

    def detach_memory(self):
        """
        Detach GRU hidden state from the current computation graph.
        Useful during training to avoid backpropagating indefinitely through time.
        """
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()

    def _init_hidden_if_needed(self, x):
        """
        Initialize or resize hidden state if needed.
        x is expected to have shape (N_UAVs, hidden_dim) before GRU unsqueeze.
        """
        batch_size = x.size(0)
        hidden_dim = x.size(1)

        need_init = (
            self.hidden_state is None or
            self.hidden_state.size(1) != batch_size or
            self.hidden_state.size(2) != hidden_dim or
            self.hidden_state.device != x.device
        )

        if need_init:
            self.hidden_state = torch.zeros(
                1, batch_size, hidden_dim,
                device=x.device,
                dtype=x.dtype
            )

    def _edge_aggregate(self, x, edge_index):
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), dtype=x.dtype, device=x.device))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return x + agg / deg

    def forward(self, x, edge_index, alive_mask=None):
        if not self.use_burst_history and x.size(-1) >= 24:
            x = x.clone()
            x[:, 21:24] = 0.0

        # Apply spatial encoder
        if self.use_graph and self.use_attention:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
        else:
            x = F.relu(self.pre_mlp(x))
            if self.use_graph:
                x = self._edge_aggregate(x, edge_index)
            x = F.relu(self.post_mlp(x))

        # Apply Temporal Memory (GRU) with persistent hidden state
        # Shape before GRU: (N_UAVs, hidden_dim)
        if self.use_gru:
            self._init_hidden_if_needed(x)

            # GRU expects (batch, seq_len, features)
            x_seq = x.unsqueeze(1)  # Shape: (N_UAVs, 1, hidden_dim)

            x_out, self.hidden_state = self.memory_block(x_seq, self.hidden_state)
            x = x_out.squeeze(1)     # Shape: (N_UAVs, hidden_dim)

        # Dueling Bifurcation
        values = self.value_stream(x)           # Shape: (N_UAVs, 1)
        advantages = self.advantage_stream(x)   # Shape: (N_UAVs, num_actions)

        # Recombine: Q(s, a) = V(s) + ( A(s, a) - mean(A(s, a)) )
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        if alive_mask is not None:
            # Mask out dead clusters
            # q_values shape is (N_UAVs, num_actions), alive_mask is (N_UAVs,)
            q_values = q_values * alive_mask.unsqueeze(-1).float()

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
    """

    def __init__(self, num_agents: int, state_dim: int, mixer_embed: int = 32):
        super(QMIXMixer, self).__init__()

        self.num_agents = num_agents
        self.mixer_embed = mixer_embed

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, num_agents * mixer_embed),
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, mixer_embed),
        )

        self.hyper_b1 = nn.Linear(state_dim, mixer_embed)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixer_embed),
            nn.ReLU(),
            nn.Linear(mixer_embed, 1),
        )

    def forward(self, agent_qs, global_state, alive_mask=None):
        B = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(B, self.num_agents, self.mixer_embed)
        
        if alive_mask is not None:
            # alive_mask: (B, N) -> mask out contributions of dead clusters
            w1 = w1 * alive_mask.unsqueeze(-1).float()

        b1 = self.hyper_b1(global_state).unsqueeze(1)

        qs = agent_qs.unsqueeze(1)
        hidden = F.elu(torch.bmm(qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.unsqueeze(2)
        b2 = self.hyper_b2(global_state)

        q_total = torch.bmm(hidden, w2).squeeze(2) + b2
        return q_total
