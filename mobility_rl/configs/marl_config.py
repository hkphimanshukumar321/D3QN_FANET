# marl_config.py
# Centralized configuration for Multi-Agent RL (MARL) training and evaluation

import os

from configs.cluster_config import ClusterConfig as CC

class MARLConfig:
    # --------------------------------------------------
    # Observation / Action
    # --------------------------------------------------
    NUM_AGENTS = CC.C_MAX
    OBS_DIM = CC.OBS_DIM_CLUSTER
    NUM_ACTIONS = CC.NUM_ACTIONS

    # --------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------
    EPISODES = 2000
    MAX_STEPS_PER_EP = 100     # Steps per episode
    BATCH_SIZE = 64
    REPLAY_SIZE = 50000
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 1500       # Steps for exponential decay
    TARGET_UPDATE_FREQ = 1000  # Steps between target net sync
    HIDDEN_DIM = 64
    GNN_HEADS = 4
    MAGAT_USE_GRAPH = True
    MAGAT_USE_ATTENTION = True
    MAGAT_USE_GRU = True
    MAGAT_USE_BURST_HISTORY = True

    # --------------------------------------------------
    # Reward weights are now managed centrally in ClusterConfig
    # See algorithms/rl/rewards.py for math formulation.
    # --------------------------------------------------
    W_THROUGHPUT = CC.ALPHA_LOCAL_THROUGHPUT + CC.ALPHA_INTER_THROUGHPUT
    W_DELAY = CC.BETA_LOCAL_DELAY + CC.BETA_INTER_DELAY
    W_DROPS = CC.PHI_QUEUE_OVERFLOW
    W_COLLISIONS = CC.GAMMA_COLLISION

    # --------------------------------------------------
    # QMIX-specific
    # --------------------------------------------------
    QMIX_EMBED_DIM = 32       # Hypernetwork embedding dimension

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    @staticmethod
    def get_checkpoint_dir():
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints_unified"))
        os.makedirs(base, exist_ok=True)
        return base
