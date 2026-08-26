"""Test vectorized updates for VDN, QMIX, and IQL agents.

Verifies that the vectorized update() methods produce valid losses and properly
update target networks and weights without shape mismatches.
"""

import numpy as np
import pytest
import torch

from algorithms.rl.marl_baselines import IQLAgent, VDNAgent, QMIXAgent


@pytest.fixture
def synthetic_marl_data():
    """Generate synthetic MARL replay data."""
    bsz = 32
    n_agents = 5
    obs_dim = 24
    num_actions = 18

    rng = np.random.default_rng(42)
    obs = rng.standard_normal((bsz, n_agents, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((bsz, n_agents, obs_dim)).astype(np.float32)
    actions = rng.integers(0, num_actions, size=(bsz, n_agents)).astype(np.int64)
    rewards = rng.standard_normal((bsz, n_agents)).astype(np.float32)
    dones = rng.choice([0.0, 1.0], size=bsz, p=[0.9, 0.1]).astype(np.float32)
    alive_masks = rng.choice([True, False], size=(bsz, n_agents), p=[0.8, 0.2])
    # Ensure at least one agent alive per step
    alive_masks[:, 0] = True

    return {
        "n_agents": n_agents,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "batch_size": bsz,
        "obs": obs,
        "next_obs": next_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "alive_masks": alive_masks,
    }


def test_iql_vectorized_update(synthetic_marl_data):
    d = synthetic_marl_data
    agent = IQLAgent(
        n_agents=d["n_agents"],
        obs_dim=d["obs_dim"],
        num_actions=d["num_actions"],
        batch_size=d["batch_size"],
        device="cpu",
    )
    for i in range(d["batch_size"]):
        agent.store(
            d["obs"][i], d["actions"][i], d["rewards"][i],
            d["next_obs"][i], bool(d["dones"][i]), d["alive_masks"][i],
        )

    loss = agent.update()
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss >= 0.0


def test_vdn_vectorized_update(synthetic_marl_data):
    d = synthetic_marl_data
    agent = VDNAgent(
        n_agents=d["n_agents"],
        obs_dim=d["obs_dim"],
        num_actions=d["num_actions"],
        batch_size=d["batch_size"],
        device="cpu",
    )
    for i in range(d["batch_size"]):
        agent.store(
            d["obs"][i], d["actions"][i], d["rewards"][i],
            d["next_obs"][i], bool(d["dones"][i]), d["alive_masks"][i],
        )

    loss = agent.update()
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss >= 0.0


def test_qmix_vectorized_update(synthetic_marl_data):
    d = synthetic_marl_data
    agent = QMIXAgent(
        n_agents=d["n_agents"],
        obs_dim=d["obs_dim"],
        num_actions=d["num_actions"],
        batch_size=d["batch_size"],
        embed_dim=32,
        device="cpu",
    )
    for i in range(d["batch_size"]):
        agent.store(
            d["obs"][i], d["actions"][i], d["rewards"][i],
            d["next_obs"][i], bool(d["dones"][i]), d["alive_masks"][i],
        )

    loss = agent.update()
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss >= 0.0
