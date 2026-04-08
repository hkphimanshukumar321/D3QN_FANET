import numpy as np

from configs.cluster_config import ClusterConfig as CC
from envs.burst_scheduler import VALID_RHO_LEVELS, decode_action, encode_action
from envs.marl_mac_env import MARLMacEnv
from envs.sarl_central_env import SARLCentralEnv


def test_burst_action_respects_timing_constraints():
    assert VALID_RHO_LEVELS, "Expected at least one valid rho level"
    for rho in VALID_RHO_LEVELS:
        t1 = rho * CC.BURST_TOTAL_TIME
        t2 = (1.0 - rho) * CC.BURST_TOTAL_TIME
        assert t1 >= CC.T1_MIN
        assert t2 >= CC.T2_MIN
        assert abs((t1 + t2) - CC.BURST_TOTAL_TIME) < 1e-9

    action = decode_action(encode_action(1, len(VALID_RHO_LEVELS) // 2))
    assert action.mac_mode == 1
    assert action.rho in VALID_RHO_LEVELS


def test_env_reports_burst_metrics():
    env = MARLMacEnv(seed=42)
    env.reset(seed=42)
    action_id = encode_action(0, len(VALID_RHO_LEVELS) // 2)
    _, _, _, _, infos = env.step({a: action_id for a in env.possible_agents})

    alive_infos = [infos[a] for a in env.possible_agents if infos[a]["alive"]]
    assert alive_infos, "Expected live cluster infos after one step"
    sample = alive_infos[0]
    for key in ["rho", "t1_time", "t2_time", "throughput_mbps", "inter_coord_success", "chosen_mac"]:
        assert key in sample, f"Missing burst metric key: {key}"
    assert abs(sample["t1_time"] + sample["t2_time"] - CC.BURST_TOTAL_TIME) < 1e-9


def test_rho_changes_execution_metrics():
    low_rho_action = encode_action(0, 0)
    high_rho_action = encode_action(0, len(VALID_RHO_LEVELS) - 1)

    env_low = MARLMacEnv(seed=123)
    env_low.reset(seed=123)
    _, _, _, _, infos_low = env_low.step({a: low_rho_action for a in env_low.possible_agents})

    env_high = MARLMacEnv(seed=123)
    env_high.reset(seed=123)
    _, _, _, _, infos_high = env_high.step({a: high_rho_action for a in env_high.possible_agents})

    low_alive = next(info for info in infos_low.values() if info.get("alive", False))
    high_alive = next(info for info in infos_high.values() if info.get("alive", False))

    assert low_alive["t1_time"] != high_alive["t1_time"]
    assert (
        low_alive["throughput_mbps"] != high_alive["throughput_mbps"]
        or low_alive["inter_coord_success"] != high_alive["inter_coord_success"]
    )


def test_sarl_central_env_uses_joint_multidiscrete_action():
    env = SARLCentralEnv(seed=7)
    obs, _ = env.reset(seed=7)
    assert env.action_space.nvec.shape[0] == CC.C_MAX
    assert obs["scalars"].shape[0] == CC.C_MAX * CC.OBS_DIM_CLUSTER

    action = np.full(CC.C_MAX, encode_action(1, len(VALID_RHO_LEVELS) // 2), dtype=np.int64)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs["scalars"].shape[0] == CC.C_MAX * CC.OBS_DIM_CLUSTER
    assert isinstance(reward, float)
    assert "raw_infos" in info
