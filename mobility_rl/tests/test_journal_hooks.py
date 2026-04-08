import numpy as np

from envs.marl_mac_env import MARLMacEnv


def test_env_exposes_step_records_and_summary():
    env = MARLMacEnv(seed=7)
    obs, infos = env.reset(
        options={
            "topology_preset": "compact_dense",
            "traffic_profile": "bursty_on_off",
            "failure_schedule": [{"step": 0, "target": "random"}],
        }
    )
    actions = {agent: 0 for agent in env.possible_agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    records = env.get_last_step_records()
    summary = env.get_last_step_summary()
    assert records
    assert "avg_graph_degree" in summary
    assert "failure_events" in summary
    assert all("rho" in record for record in records)


def test_graph_mode_none_hides_policy_edges():
    env = MARLMacEnv(seed=8)
    env.reset(options={"graph_mode": "none"})
    _, edge_index, alive_mask = env.get_global_graph_state()
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 0
    assert alive_mask.shape[0] > 0


def test_observation_noise_and_staleness_preserve_shape():
    env = MARLMacEnv(seed=9)
    obs0, _ = env.reset(options={"obs_staleness_steps": 1, "obs_noise_std": 0.01})
    actions = {agent: 0 for agent in env.possible_agents}
    obs1, *_ = env.step(actions)
    for agent in env.possible_agents:
        assert obs0[agent].shape == obs1[agent].shape
        assert obs0[agent].dtype == np.float32
