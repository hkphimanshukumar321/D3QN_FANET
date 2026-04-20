"""Quick smoke test: MAPPO on-policy episode update."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from algorithms.rl.mappo_agent import MAPPOAgent

def test_mappo_episode_update():
    agent = MAPPOAgent(4, 10, 6, hidden_dim=32, batch_size=2, ppo_epochs=2)
    obs = np.random.randn(4, 10).astype(np.float32)
    mask = np.ones(4, dtype=np.bool_)
    rewards = np.array([0.5, -0.1, 0.3, 0.2], dtype=np.float32)

    # Collect 10 transitions (simulating one episode)
    for _ in range(10):
        out = agent.select_actions(obs, alive_mask=mask, store=True)
        actions, logprobs, val, global_obs = out
        agent.store(obs, global_obs, actions, logprobs, val, rewards, False, alive_mask=mask)

    assert len(agent.buffer["rewards"]) == 10, "Buffer should have 10 transitions"

    # On-policy update at episode end
    loss = agent.update()
    print(f"Update loss after 10 transitions: {loss}")
    assert not np.isnan(loss), "Loss should not be NaN"
    assert len(agent.buffer["rewards"]) == 0, "Buffer should be cleared after update"

    # Verify model still produces valid outputs
    eval_actions = agent.select_actions(obs, alive_mask=mask, store=False)
    assert len(eval_actions) == 4, "Should return 4 actions"
    assert all(isinstance(a, int) for a in eval_actions), "Actions should be ints"

    print("PASS: MAPPO on-policy episode update works correctly")


def test_mappo_nan_guard():
    agent = MAPPOAgent(4, 10, 6, hidden_dim=32, batch_size=2)

    # Force NaN into buffer
    obs = np.random.randn(4, 10).astype(np.float32)
    mask = np.ones(4, dtype=np.bool_)
    nan_rewards = np.array([float("nan")] * 4, dtype=np.float32)

    for _ in range(5):
        out = agent.select_actions(obs, alive_mask=mask, store=True)
        actions, logprobs, val, global_obs = out
        agent.store(obs, global_obs, actions, logprobs, val, nan_rewards, False, alive_mask=mask)

    loss = agent.update()
    assert loss == 0.0, "NaN guard should return 0.0 and skip update"
    print("PASS: MAPPO NaN guard works correctly")


if __name__ == "__main__":
    test_mappo_episode_update()
    test_mappo_nan_guard()
