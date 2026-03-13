import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from algorithms.rl.features_extractor import MCAFeaturesExtractor


class DuelingQNetwork(nn.Module):
    """
    Proper Dueling Q-Network head to replace SB3's default flat MLP.

    Splits the feature vector from MCAFeaturesExtractor into:
        V(s)    — scalar state value
        A(s, a) — per-action advantage

    Recombined as: Q(s,a) = V(s) + ( A(s,a) - mean_a[ A(s,a) ] )

    This is the only thing missing from the original MCA-D3QN — everything
    else in create_mca_d3qn() is unchanged.

    Args:
        features_dim : output dim of MCAFeaturesExtractor (256).
        n_actions    : number of discrete actions in the environment.
        hidden_dim   : width of the V and A stream hidden layers.
    """

    def __init__(self, features_dim: int, n_actions: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        # Value stream V(s) — how good is this state overall?
        self.value_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream A(s, a) — how much better is each action relatively?
        self.advantage_stream = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values     = self.value_stream(features)        # (batch, 1)
        advantages = self.advantage_stream(features)    # (batch, n_actions)
        # Mean-subtracted recombination — standard Dueling formula
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values                                 # (batch, n_actions)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the network in training or evaluation mode.
        Required by newer versions of Stable Baselines 3.
        """
        self.train(mode)


class DuelingMultiInputPolicy(MultiInputPolicy):
    """
    Thin SB3 policy subclass that swaps the default flat Q-network
    for our DuelingQNetwork — everything else (optimizer, feature
    extractor wiring, double-DQN logic) stays exactly as SB3 provides.
    """

    def make_q_net(self) -> DuelingQNetwork:
        # net_arch is passed via policy_kwargs but unused here —
        # the Dueling streams use hidden_dim=128 matching the original net_arch.
        # SB3 doesn't assign self.features_dim on this class, so we read it directly from the extractor.
        features_dim = getattr(self, "features_dim", getattr(self.features_extractor, "features_dim", 256))
        
        return DuelingQNetwork(
            features_dim=features_dim,
            n_actions=self.action_space.n,
            hidden_dim=128,
        ).to(self.device)


def create_mca_d3qn(env, learning_rate=1e-3, buffer_size=100000,
                    batch_size=64, gamma=0.99, exploration_fraction=0.1,
                    target_update_interval=1000, seed=42):
    """
    Creates the Mobility-Channel-Aware Dueling Double DQN (MCA-D3QN)
    using Stable-Baselines3 DQN with a proper Dueling Q-network head.

    SB3's DQN natively provides:
        - Double Q-learning        (on by default)
        - Replay buffer
        - Epsilon-greedy schedule

    DuelingMultiInputPolicy adds:
        - True V(s) / A(s,a) split via DuelingQNetwork
          (previously approximated by a flat MLP — now fully realised)

    MCAFeaturesExtractor provides:
        - Multi-branch feature extraction (mobility + channel awareness)
        - 256-dim joint embedding fed into both Dueling streams
    """

    policy_kwargs = dict(
        features_extractor_class=MCAFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # net_arch is intentionally omitted — DuelingQNetwork owns
        # its own hidden layers (128-dim streams), keeping parity
        # with the original net_arch=[128, 128].
    )

    model = DQN(
        DuelingMultiInputPolicy,        # ← only change to create_mca_d3qn body
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=100,
        batch_size=batch_size,
        tau=1.0,                        # Hard update — intentional, keep as-is
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        tensorboard_log="./results/tensorboard_logs/",
    )

    return model