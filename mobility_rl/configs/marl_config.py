# marl_config.py
# Centralized configuration for Multi-Agent RL (MARL) training and evaluation

import os

class MARLConfig:
    # --------------------------------------------------
    # Observation / Action
    # --------------------------------------------------
    OBS_DIM = 16               # Per-agent observation features
    NUM_ACTIONS = 2            # 0=TDMA, 1=CSMA/CA

    # --------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------
    EPISODES = 500
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

    # --------------------------------------------------
    # Reward Weights
    # --------------------------------------------------
    W_THROUGHPUT = 1.0
    W_DELAY = 0.5
    W_DROPS = 0.3
    W_COLLISIONS = 0.2
    W_LINK_UTIL = 0.1

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
