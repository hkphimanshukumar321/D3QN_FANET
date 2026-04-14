# cluster_config.py
# Configuration for Dynamic Cluster-Based Decentralized Control
#
# Mathematical reference:
#   S_{i,k}(t) = w_d f_d + w_s f_s + w_m f_m + w_q f_q
#   cluster(i,t) = argmax_k S_{i,k}(t)  subject to hysteresis

import os


class ClusterConfig:
    """All tunable constants for the clustering subsystem."""

    # --------------------------------------------------
    # Cluster count bounds
    # --------------------------------------------------
    C_MIN = 3
    C_MAX = 10
    C_INIT = 5

    # --------------------------------------------------
    # Cluster size bounds (triggers split / merge)
    # --------------------------------------------------
    N_MIN = 2
    N_MAX = 12

    # --------------------------------------------------
    # Spatial radii (meters)
    # --------------------------------------------------
    R_C = 100.0
    R_I = 250.0

    # --------------------------------------------------
    # Interference graph SINR threshold
    # --------------------------------------------------
    TAU_I = 10.0

    # --------------------------------------------------
    # Membership score weights
    #   S_{i,k} = w_d*f_d + w_s*f_s + w_m*f_m + w_q*f_q
    # --------------------------------------------------
    W_DIST = 0.40
    W_SINR = 0.25
    W_MOB = 0.20
    W_LOAD = 0.15

    # --------------------------------------------------
    # Hysteresis thresholds (anti-oscillation)
    # --------------------------------------------------
    THETA_JOIN = 0.65
    THETA_LEAVE = 0.35

    # --------------------------------------------------
    # Cluster-head health & handover
    # --------------------------------------------------
    HEALTH_THRESHOLD = 0.20
    A_ENERGY = 0.30
    A_DEGREE = 0.25
    A_MOBSTAB = 0.20
    A_QUEUE = 0.15
    A_RISK = 0.10

    # --------------------------------------------------
    # Energy model (simple linear drain)
    # --------------------------------------------------
    E_INIT = 100.0
    E_TX_COST = 0.01
    E_IDLE_COST = 0.001

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------
    T_CLUSTER = 5

    # --------------------------------------------------
    # Observation space
    # --------------------------------------------------
    OBS_DIM_CLUSTER = 24
    NEIGHBOR_SUMMARY_DIM = 3
    MAX_NEIGHBORS = 6

    # --------------------------------------------------
    # Burst split control
    # --------------------------------------------------
    BURST_TOTAL_TIME = 0.5
    T1_MIN = 0.05
    T2_MIN = 0.05
    RHO_ACTION_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    DEFAULT_RHO = 0.5
    COORD_SYNC_MODE = "tail_aligned"
    COORD_GUARD_TIME = 0.05
    INTRA_CLUSTER_PAYLOAD_TYPES = ("data", "control", "retransmission")
    INTER_CLUSTER_COORD_PAYLOAD_TYPES = (
        "schedule_exchange",
        "relay_control",
        "topology_update",
        "handover_notice",
    )
    COORD_CAPACITY_PER_SEC = 4.0
    LOCAL_CTRL_CAPACITY_PER_SEC = 6.0
    RELAY_SERVICE_CAPACITY_PER_SEC = 3.0
    COORD_BACKLOG_INCREMENT = 0.25
    RELAY_DEMAND_INCREMENT = 0.20
    LOCAL_CTRL_DEMAND_INCREMENT = 0.15
    MAX_COORD_BACKLOG = 10.0
    MAX_RELAY_DEMAND = 10.0
    MAX_LOCAL_CTRL_DEMAND = 10.0

    # --------------------------------------------------
    # Action space
    # --------------------------------------------------
    NUM_MAC_MODES = 2
    NUM_CHANNELS = 1
    NUM_CW_CLASSES = 1
    NUM_RHO_LEVELS = len(RHO_ACTION_LEVELS)

    #   |A| = NUM_MAC_MODES * NUM_CHANNELS * NUM_CW_CLASSES * NUM_RHO_LEVELS
    NUM_ACTIONS = NUM_MAC_MODES * NUM_CHANNELS * NUM_CW_CLASSES * NUM_RHO_LEVELS

    # --------------------------------------------------
    # Reward coefficients
    # --------------------------------------------------
    ALPHA_LOCAL_THROUGHPUT = 0.22
    ALPHA_INTER_THROUGHPUT = 0.13
    BETA_LOCAL_DELAY = 0.10
    BETA_INTER_DELAY = 0.05
    GAMMA_COLLISION = 0.20
    DELTA_ENERGY = 0.05
    ETA_INTERFERENCE = 0.12
    PSI_COORD_FAILURE = 0.08
    PHI_QUEUE_OVERFLOW = 0.10
    XI_AOI = 0.00
    ZETA_HANDOVER = 0.05
    LAMBDA_FAIRNESS = 0.10

    # --------------------------------------------------
    # Normalization ceilings
    # --------------------------------------------------
    MAX_THROUGHPUT_MBPS = 5.0
    MAX_DELAY_MS = 200.0
    MAX_COLLISIONS = 500
    MAX_ENERGY_COST = 1.0
    MAX_QUEUE_OVERFLOW_RATIO = 1.0
    MAX_COORD_FAILURE_RATIO = 1.0

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    @staticmethod
    def get_cluster_log_dir():
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "results", "cluster_logs")
        )
        os.makedirs(base, exist_ok=True)
        return base
