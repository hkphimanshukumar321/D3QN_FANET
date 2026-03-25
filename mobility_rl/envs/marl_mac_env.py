from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

from configs import config as params
from configs.marl_config import MARLConfig
from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.link import compute_distances, compute_link_up
from algorithms.mac.baseline import Config, Logger, simulate_tdma, simulate_csma_ca

# Optional fading imports
try:
    from algorithms.channel.fading import BERCalculator, AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
    from algorithms.mobility.link import compute_fading_success_prob
    _HAS_FADING = True
except ImportError:
    _HAS_FADING = False


class MARLMacEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for Decentralized Multi-Agent RL MAC Selection.

    Each UAV observes its local state (16 features) and independently selects
    a MAC protocol. A majority vote determines the shared network protocol.
    All agents receive a cooperative reward based on global network performance.

    Supports fading channels when ENABLE_FADING is True in config.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed=None):
        super().__init__()
        self.N = params.N
        self.agents = [f"uav_{i}" for i in range(self.N)]
        self.possible_agents = self.agents[:]

        # Action space: 0=TDMA, 1=CSMA
        self.action_spaces = {a: spaces.Discrete(MARLConfig.NUM_ACTIONS) for a in self.agents}

        # Observation space: 16 features per agent
        self.obs_dim = MARLConfig.OBS_DIM
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for a in self.agents
        }

        self.seed_val = seed if seed is not None else params.SEED
        self.rng = np.random.default_rng(self.seed_val)

        self.current_step = 0
        self.max_steps = MARLConfig.MAX_STEPS_PER_EP
        self.speed_engine = None
        self.mobility_model = None

        # Fading
        self.fading_channel = None
        self.ber_calc = None
        self._init_fading()

        # Per-step tracking
        self._prev_actions = {}
        self._prev_reward = 0.0
        self._fading_gains = np.ones(self.N)
        self._succ_probs = np.ones(self.N)
        self._distances = np.zeros(self.N)

        # Communication log
        self.comm_log = []

    def _init_fading(self):
        """Initialize fading channel objects if enabled."""
        if not _HAS_FADING or not getattr(params, 'ENABLE_FADING', False):
            return
        self.ber_calc = BERCalculator(modulation=getattr(params, 'MODULATION', 'BPSK'))
        f_model = getattr(params, 'FADING_MODEL', 'awgn').lower()
        if f_model == "rayleigh":
            self.fading_channel = RayleighChannel()
        elif f_model == "rician":
            self.fading_channel = RicianChannel(K=getattr(params, 'RICIAN_K', 3.0))
        elif f_model == "nakagami":
            self.fading_channel = NakagamiChannel(
                m=getattr(params, 'NAKAGAMI_M', 2.0),
                omega=getattr(params, 'NAKAGAMI_OMEGA', 1.0)
            )
        else:
            self.fading_channel = AWGNChannel()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(self.seed_val)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self._prev_actions = {a: 0 for a in self.agents}
        self._prev_reward = 0.0
        self.comm_log = []

        # Initialize mobility
        self.speed_engine = SpeedEngine(
            n_nodes=self.N,
            v_min=params.V_MIN, v_max=params.V_MAX,
            mode=params.SPEED_MODE, v_mean=params.V_MEAN, v_std=params.V_STD,
            update_interval=params.SPEED_UPDATE_INTERVAL,
            rng=self.rng
        )
        self.mobility_model = create_mobility_model(
            name=params.MOBILITY_MODEL,
            n_nodes=self.N,
            bounds=(params.AREA_X, params.AREA_Y, params.AREA_Z),
            speed_engine=self.speed_engine,
            rng=self.rng,
            gm_alpha=params.GM_ALPHA
        )

        # Compute initial distances and fading
        self._update_link_state()

        observations = self._get_obs()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _update_link_state(self):
        """Compute distances, link status, and fading-aware success probabilities."""
        pos = self.mobility_model.positions
        
        if getattr(params, 'DECENTRALIZED_COMM', False):
            # Calculate N x N distance matrix, find closest peer for each UAV
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)  # Exclude self
            self._distances = np.sqrt(np.min(dist_sq, axis=1))
        else:
            sink = np.array([params.SINK_X, params.SINK_Y, params.SINK_Z])
            self._distances = compute_distances(pos, sink)
            
        self._link_up = compute_link_up(self._distances, params.COMM_RANGE_R)

        if self.fading_channel is not None and self.ber_calc is not None:
            payload_bits = getattr(params, 'PAYLOAD_BYTES', 1500) * 8
            self._succ_probs = compute_fading_success_prob(
                self._distances, self.fading_channel, self.ber_calc,
                params.TX_POWER_DBM, params.NOISE_POWER_DBM,
                payload_bits, self.rng
            )
            self._fading_gains = self.fading_channel.sample_gain(self.N, self.rng)
        else:
            self._succ_probs = np.ones(self.N)
            self._fading_gains = np.ones(self.N)

    def _get_obs(self):
        """Build 16-feature observation per agent."""
        sink = np.array([params.SINK_X, params.SINK_Y, params.SINK_Z])
        pos = self.mobility_model.positions
        vel = self.mobility_model.velocities

        # Neighbor computation (within comm range, O(N^2) but vectorized)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        R2 = params.COMM_RANGE_R ** 2
        neighbor_mask = (dist_sq <= R2) & ~np.eye(self.N, dtype=bool)
        n_neighbors = neighbor_mask.sum(axis=1)
        mean_neighbor_dist = np.zeros(self.N)
        mean_neighbor_speed = np.zeros(self.N)

        speeds = np.linalg.norm(vel, axis=1)
        for i in range(self.N):
            nbrs = np.where(neighbor_mask[i])[0]
            if len(nbrs) > 0:
                mean_neighbor_dist[i] = np.sqrt(dist_sq[i, nbrs]).mean()
                mean_neighbor_speed[i] = speeds[nbrs].mean()

        obs = {}
        for i, agent in enumerate(self.possible_agents):
            # 16 features
            load = getattr(params, 'SWEEP_MAX_PPS', 400) / max(self.N, 1)
            prev_act = float(self._prev_actions.get(agent, 0))

            node_feat = np.array([
                pos[i][0] / max(params.AREA_X, 1),          # 1. x (normalized)
                pos[i][1] / max(params.AREA_Y, 1),          # 2. y
                pos[i][2] / max(params.AREA_Z, 1),          # 3. z
                vel[i][0] / 30.0,                            # 4. vx (normalized by max speed)
                vel[i][1] / 30.0,                            # 5. vy
                vel[i][2] / 30.0,                            # 6. vz
                self._distances[i] / 2000.0,                 # 7. distance to target (sink or peer)
                n_neighbors[i] / max(self.N, 1),             # 8. neighbor count (normalized)
                mean_neighbor_dist[i] / 1000.0,              # 9. mean neighbor distance
                0.0,                                          # 10. queue occupancy (placeholder)
                load / 1000.0,                                # 11. traffic load
                self._succ_probs[i],                          # 12. link success probability
                prev_act,                                     # 13. previous MAC action
                self._prev_reward / 10.0,                     # 14. previous reward (normalized)
                mean_neighbor_speed[i] / 30.0,                # 15. mean neighbor speed
                self._fading_gains[i],                        # 16. fading gain
            ], dtype=np.float32)
            obs[agent] = node_feat

        return obs

    def get_global_graph_state(self):
        """
        Extracts topological data for GNN models.
        Returns: (node_features, edge_index)
        """
        obs_dict = self._get_obs()
        x = np.stack([obs_dict[a] for a in self.possible_agents])

        pos = self.mobility_model.positions
        R = params.COMM_RANGE_R
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        valid_edges = (dist_sq <= R**2) & ~np.eye(self.N, dtype=bool)
        row, col = np.where(valid_edges)

        if len(row) > 0:
            edge_index = np.stack([row, col], axis=0).astype(np.int64)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        return x, edge_index

    def step(self, actions):
        # 1. Advance mobility
        self.mobility_model.update(params.MOBILITY_DT)
        self._update_link_state()

        # 2. Majority vote
        votes = [actions.get(a, 0) for a in self.agents]
        vote_tdma = sum(1 for v in votes if v == 0)
        vote_csma = sum(1 for v in votes if v == 1)
        chosen_mac = 1 if vote_csma > vote_tdma else 0

        # 3. MAC simulation
        cfg = Config(
            N=self.N,
            sim_time_s=0.5,
            slot_time_s=params.SLOT_TIME_S,
            phy_rate_bps=params.PHY_RATE_BPS,
            payload_bytes=params.PAYLOAD_BYTES,
            QMAX=params.QMAX,
            seed=self.seed_val + self.current_step,
            cw_min=params.CW_MIN,
            cw_max=params.CW_MAX,
            difs_slots=params.DIFS_SLOTS,
            sifs_slots=params.SIFS_SLOTS,
            ack_slots=params.ACK_SLOTS,
            ack_timeout_slots=params.ACK_TIMEOUT_SLOTS,
            max_retry=params.MAX_RETRY,
            log_interval_slots=params.LOG_INTERVAL_SLOTS,
            rts_cts_enabled=params.RTS_CTS_ENABLED,
            ack_enabled=params.ACK_ENABLED,
            rts_slots=params.RTS_SLOTS,
            cts_slots=params.CTS_SLOTS,
            tdma_guard_time_s=params.TDMA_GUARD_TIME_S,
        )
        log = Logger()
        load = getattr(params, 'OFFERED_PPS', getattr(params, 'SWEEP_MAX_PPS', 400))

        if chosen_mac == 0:
            simulate_tdma(cfg, load, log)
        else:
            simulate_csma_ca(cfg, load, log)

        # 4. Cooperative reward
        throughput = log.get_throughput_bps(0.5) / 1e6
        drops = log.pkts_dropped_qfull + log.pkts_dropped_mac
        collisions = log.collision_events
        delay_ms = log.get_avg_end_to_end_delay_s() * 1000
        pkts_gen = max(log.pkts_generated, 1)

        t_norm = min(throughput / 10.0, 1.0)
        d_norm = min(delay_ms / 100.0, 1.0)
        drop_rate = min(drops / pkts_gen, 1.0)
        col_norm = min(collisions / 1000.0, 1.0)
        link_util = float(np.mean(self._succ_probs))

        reward = (
            MARLConfig.W_THROUGHPUT * t_norm
            - MARLConfig.W_DELAY * d_norm
            - MARLConfig.W_DROPS * drop_rate
            - MARLConfig.W_COLLISIONS * col_norm
            + MARLConfig.W_LINK_UTIL * link_util
        )

        self._prev_reward = reward
        self._prev_actions = dict(actions)

        # 5. Communication log entry
        fading_info = getattr(params, 'FADING_MODEL', 'none') if getattr(params, 'ENABLE_FADING', False) else 'none'
        self.comm_log.append({
            "step": self.current_step,
            "mac": "TDMA" if chosen_mac == 0 else "CSMA",
            "votes_tdma": vote_tdma,
            "votes_csma": vote_csma,
            "throughput_mbps": round(throughput, 4),
            "delay_ms": round(delay_ms, 4),
            "drops": drops,
            "collisions": collisions,
            "avg_succ_prob": round(float(np.mean(self._succ_probs)), 4),
            "avg_fading_gain": round(float(np.mean(self._fading_gains)), 4),
            "fading_model": fading_info,
        })

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        rewards = {a: float(reward) for a in self.agents}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {
            "chosen_mac": chosen_mac, "throughput": throughput, "drops": drops,
            "delay_ms": delay_ms, "collisions": collisions, "fading": fading_info,
            "avg_succ_prob": float(np.mean(self._succ_probs)),
        } for a in self.agents}

        if terminated:
            self.agents = []

        observations = self._get_obs()
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
