# config.py
# Centralized Configuration for FANET Simulation Experiments

# ==========================================
# Global Simulation & Node Parameters
# ==========================================
MAC_SELECTION = ["TDMA", "CSMA_CA", "TABULAR", "DQN", "PPO", "A2C", "MCA_D3QN", "MARL_GNN"]  # Protocols for RL evaluation
N = 50                                   # Number of senders / nodes
SIM_TIME_S = 15                               # Simulation duration (seconds)
SLOT_TIME_S = 9e-6                            # Discrete time slot granularity
PHY_RATE_BPS = 5e6                            # Data rate (channel/link capacity in bps)
PAYLOAD_BYTES = 1000                      # Packet payload size
QMAX = 100                                   # Queue/buffer capacity per node
SEED = 42                                     # Random seed baseline

# ==========================================
# Experiment Sweep Settings (Load)
# ==========================================
SWEEP_MIN_PPS = 100                           # Minimum total traffic load (packets/sec)
SWEEP_MAX_PPS = 1500                          # Maximum total traffic load (packets/sec)
SWEEP_STEPS = 30                               # Number of granular steps in the sweep

# ==========================================
# Traffic Generation Model
# ==========================================
# Traffic follows a Poisson process with memoryless exponential inter-arrival times.
# Arrivals are independently sampled per node. No burst or correlated traffic is modeled.
# Destinations are explicitly not mapped (evaluates abstract channel contention / common sink).

# ==========================================
# Logging Settings
# ==========================================
LOG_INTERVAL_SLOTS = 1000                     # Snapshot interval for buffer tracking

# ==========================================
# CSMA/CA Tunable Parameters
# ==========================================
CW_MIN = 15
CW_MAX = 1023
DIFS_SLOTS = 4
SIFS_SLOTS = 2
ACK_TIMEOUT_SLOTS = 10
MAX_RETRY = 1

# Features
RTS_CTS_ENABLED = True
ACK_ENABLED = True

# Control Frames (in slots - approximations)
RTS_SLOTS = 3
CTS_SLOTS = 3
ACK_SLOTS = 2

# ==========================================
# TDMA Tunable Parameters
# ==========================================
TDMA_GUARD_TIME_S = 1e-6                      # Gap between assigned node slots
TDMA_GUARD_TIME_UNIT = "s"                    # Time unit for guard

# ==========================================
# Q-Learning RL Selector Parameters
# ==========================================
ENABLE_RL_SELECTOR = True  # Set False globally to avoid UI engine polling, managed instead by run_experiments.py
RL_DECISION_INTERVAL_S = 1.0  # seconds between RL MAC selection queries
RL_ALPHA = 1.0        # 1.0 = instant learning of the current best observation
RL_GAMMA = 0.0        # 0.0 = independent points, no future discounting (it's a static CSV sweep)
RL_EPSILON = 0.0      # 0.0 = never randomly explore
RL_WT = 0.5           # match MARL_Config.W_THROUGHPUT
RL_WD = 0.1           # match MARL_Config.W_DELAY
RL_STATE_MODE = "traffic_rate_bin"
RL_TRAFFIC_BINS = 30  # Needs to match SWEEP_STEPS exactly so each traffic load point gets its own bin

# ==========================================
# 3D Mobility Parameters
# ==========================================
ENABLE_MOBILITY = True

# Bounding cube (meters)
AREA_X = 200   # A
AREA_Y = 200   # B
AREA_Z = 50    # C

# Sink/base station position (fixed)
SINK_X = 200
SINK_Y = 200
SINK_Z = 25

# Mobility model: "gauss_markov" | "random_waypoint" | "random_walk" | "circular"
MOBILITY_MODEL = "random_walk"

# Gauss–Markov specific
GM_ALPHA = 0.5   # memory / correlation (0 = fully random, 1 = deterministic)

# Random Waypoint specific
RWP_PAUSE_TIME = 1.0  # seconds at each waypoint

# Circular / Spiral specific
CIRC_RADIUS = 100.0       # orbit radius (m)
CIRC_OMEGA_MEAN = 0.1     # mean angular velocity (rad/s)
CIRC_OMEGA_STD = 0.02     # std dev of angular velocity
CIRC_CLIMB_RATE = 0.5     # vertical climb/descent rate (m/s)

# Speed configuration (variable speed — must have)
SPEED_MODE = "uniform"        # "uniform" | "gaussian" | "per_node_uniform" | "piecewise"
V_MIN = 5.0                   # m/s minimum speed
V_MAX = 35                 # m/s maximum speed
V_MEAN = 15.0                 # m/s (gaussian mode)
V_STD = 5.0                   # m/s (gaussian mode)
SPEED_UPDATE_INTERVAL = 5.0   # seconds between speed changes

# Link model (Phase-1: range-gated binary)
COMM_RANGE_R = 200         # meters

# Phase-2 optional path-loss (disabled by default)
ENABLE_PATHLOSS = True
PATHLOSS_K = 0.00001
PATHLOSS_ETA = 2.0

# Optional propagation delay (disabled by default)
ENABLE_PROP_DELAY = True

# ==========================================
# Fade Model Configuration (Multi-Fading + Modulation)
# ==========================================
ENABLE_FADING = True
FADING_MODEL = "nakagami"         # "awgn" | "rayleigh" | "rician" | "nakagami"
NAKAGAMI_M = 2.0                  # Nakagami shape (m=1 → Rayleigh)
NAKAGAMI_OMEGA = 1.0              # Nakagami spread (avg power)
RICIAN_K = 3.0                    # Rician K-factor (dB)
TX_POWER_DBM = 20.0              # Transmit power
NOISE_POWER_DBM = -80          # Noise floor
BER_THRESHOLD = 1e-3              # Max acceptable BER
MODULATION = "QPSK"               # "BPSK" | "QPSK" | "8PSK" | "16QAM" | "64QAM" | "256QAM"

# Mobility time step resolution (seconds)
MOBILITY_DT = 0.1

# ==========================================
# Experiment Execution Controls
# ==========================================
RUN_TABULAR_QLEARNING = True
RUN_QLEARNING_SELECTOR = True  # Set to True to enable the legacy tabular selector logic
RUN_CUSTOM_RL = True   # MCA-D3QN
RUN_DQN = True
RUN_PPO = True
RUN_A2C = True
RUN_MARL_GNN = True

# MARL Execution config
ENABLE_MARL = True
DECENTRALIZED_COMM = True  # If True, UAVs calculate pathloss to their closest peer instead of SINK

# MARL Baselines
RUN_MARL_IQL = True
RUN_MARL_VDN = True
RUN_MARL_QMIX = True

# ==========================================
# Results / Logging Controls
# ==========================================
RESULTS_ROOT = "results"
AUTO_CREATE_RUN_FOLDER = True
SAVE_IMAGES = True
SAVE_CSV = True
SAVE_LOGS = True
SAVE_METADATA = True
SAVE_CHECKPOINTS = True
# ==========================================
# Multiprocessing & Parallel Execution
# ==========================================
ENABLE_MULTIPROCESSING = True
NUM_CPU_WORKERS = "auto"           # Use string "auto" or an integer (e.g., 4)
CPU_UTILIZATION_FRACTION = 0.8
PARALLELIZE_OVER = ["loads"]       # Which parameter loops to parallelize 

# ==========================================
# Hardware / Device Execution
# ==========================================
ENABLE_GPU = True
GPU_DEVICE_ID = 0
FORCE_CPU = False
TRAIN_ON_GPU = True
EVAL_ON_GPU = True
ENABLE_RESOURCE_LOGGING = True
RESULTS_PER_WORKER = True          # If True, parallel workers write safely isolated folders


