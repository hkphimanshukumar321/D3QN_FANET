from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

from configs import config as params
from algorithms.mobility.speed import SpeedEngine
from algorithms.mobility.models import create_mobility_model
from algorithms.mobility.link import compute_distances
from algorithms.mac.baseline import Config, Logger, simulate_tdma, simulate_csma_ca

class MARLMacEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for Multi-Agent RL MAC Selection.
    UAVs output independent MAC choices. We take a majority vote to 
    dynamically switch the shared network protocol, rewarding all UAVs 
    for global throughput and penalizing collective packet drops.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, seed=None):
        super().__init__()
        self.N = params.N
        self.agents = [f"uav_{i}" for i in range(self.N)]
        self.possible_agents = self.agents[:]
        
        # Action space: 0 for TDMA, 1 for CSMA (Agents vote)
        self.action_spaces = {a: spaces.Discrete(2) for a in self.agents}
        
        # Observation space: node local physics features
        # [x, y, z, vx, vy, vz, load_pps, distance_to_sink] -> 8 features
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            for a in self.agents
        }
        
        self.seed_val = seed if seed is not None else params.SEED
        self.rng = np.random.default_rng(self.seed_val)
        
        self.current_step = 0
        self.max_steps = 100
        self.speed_engine = None
        self.mobility_model = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(self.seed_val)
            
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
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
        
        observations = self._get_obs()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _get_obs(self):
        sink = np.array([params.SINK_X, params.SINK_Y, params.SINK_Z])
        dists = compute_distances(self.mobility_model.positions, sink)
        
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            pos = self.mobility_model.positions[i]
            vel = self.mobility_model.velocities[i]
            dist = dists[i]
            load = params.SWEEP_MAX_PPS / self.N 
            
            node_feat = np.array([
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                load, dist
            ], dtype=np.float32)
            obs[agent] = node_feat
            
        return obs
        
    def get_global_graph_state(self):
        """
        Extracts the required topological data for PyTorch Geometric GNNs.
        Returns the Node Feature Matrix (X) and Edge Index (A).
        """
        obs_dict = self._get_obs()
        x = np.stack([obs_dict[a] for a in self.possible_agents])
        
        pos = self.mobility_model.positions
        R = params.COMM_RANGE_R
        edges = []
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d <= R:
                        edges.append([i, j])
                        
        if len(edges) > 0:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            
        return x, edge_index

    def step(self, actions):
        # 1. Advance simulator physics
        self.mobility_model.update(params.MOBILITY_DT)
        
        # 2. Extract MARL agent votes
        votes = [actions.get(a, 0) for a in self.agents]
        vote_tdma = sum(1 for v in votes if v == 0)
        vote_csma = sum(1 for v in votes if v == 1)
        chosen_mac = 1 if vote_csma > vote_tdma else 0
        
        # 3. Fast Forward Simulation 
        cfg = Config(N=self.N, sim_time_s=1.0, QMAX=params.QMAX, tdma_guard_time_s=params.TDMA_GUARD_TIME_S)
        log = Logger()
        load = params.SWEEP_MAX_PPS
        
        if chosen_mac == 0:
            simulate_tdma(cfg, load, log)
        else:
            simulate_csma_ca(cfg, load, log)
            
        # 4. Cooperative Shared Reward 
        throughput = log.get_throughput_bps(1.0) / 1e6
        drops = log.pkts_dropped_qfull + log.pkts_dropped_mac
        reward = throughput - (drops * 0.05)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        rewards = {a: float(reward) for a in self.agents}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {"chosen_mac": chosen_mac, "throughput": throughput, "drops": drops} for a in self.agents}
        
        if terminated:
            self.agents = []
            
        observations = self._get_obs()
        return observations, rewards, terminations, truncations, infos
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
