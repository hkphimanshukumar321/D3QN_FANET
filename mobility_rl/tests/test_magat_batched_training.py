"""Test numerical equivalence between per-sample and batched MAGAT forward passes.

Verifies that forward_batched() on a PyG Batch of N disjoint graphs produces
identical Q-values to calling forward() N times individually (with reset_memory()
before each call).
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data, Batch


@pytest.fixture
def magat_net():
    """Create a deterministic MAGAT-D3QN network."""
    from algorithms.rl.gnn_marl import MAGAT_D3QN_QNetwork
    torch.manual_seed(42)
    net = MAGAT_D3QN_QNetwork(
        node_in_dim=24,
        hidden_dim=64,
        num_actions=18,
        heads=2,
        use_graph=True,
        use_attention=True,
        use_gru=True,
        use_burst_history=True,
    )
    net.eval()
    return net


@pytest.fixture
def synthetic_batch():
    """Generate 4 synthetic graph transitions with varying topology."""
    rng = np.random.default_rng(123)
    n_nodes = 10  # C_MAX
    obs_dim = 24

    samples = []
    for i in range(4):
        # Random node features
        x = rng.standard_normal((n_nodes, obs_dim)).astype(np.float32)
        next_x = rng.standard_normal((n_nodes, obs_dim)).astype(np.float32)

        # Random sparse edges (3-8 edges per graph)
        n_edges = rng.integers(3, 9)
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = np.stack([src, dst]).astype(np.int64)
        next_edge_index = np.stack([
            rng.integers(0, n_nodes, size=n_edges),
            rng.integers(0, n_nodes, size=n_edges),
        ]).astype(np.int64)

        # Alive mask: 6-10 nodes alive
        n_alive = rng.integers(6, n_nodes + 1)
        alive_mask = np.zeros(n_nodes, dtype=bool)
        alive_mask[:n_alive] = True
        rng.shuffle(alive_mask)

        next_alive_mask = np.zeros(n_nodes, dtype=bool)
        next_alive_mask[:rng.integers(6, n_nodes + 1)] = True
        rng.shuffle(next_alive_mask)

        # Random actions (0-17 for 18 actions)
        actions = rng.integers(0, 18, size=n_nodes).astype(np.int64)
        rewards = rng.standard_normal(n_nodes).astype(np.float32)
        done = bool(i == 3)  # Last sample is terminal

        samples.append((x, edge_index, alive_mask, actions, rewards,
                        next_x, next_edge_index, next_alive_mask, done))

    return samples


def _run_per_sample(net, samples, device="cpu"):
    """Run the old per-sample loop: forward() with reset_memory() per sample."""
    q_values_per_sample = []
    for (bx, bei, bmask, ba, br, bnx, bnei, bnmask, bd) in samples:
        bxt = torch.tensor(bx, dtype=torch.float32, device=device)
        bet = torch.tensor(bei, dtype=torch.long, device=device)
        bmask_t = torch.tensor(bmask, dtype=torch.bool, device=device)

        net.reset_memory()
        q_all = net(bxt, bet, bmask_t)
        q_values_per_sample.append(q_all)

    return torch.cat(q_values_per_sample, dim=0)


def _run_batched(net, samples, device="cpu"):
    """Run the new batched path: forward_batched() on a PyG Batch."""
    data_list = []
    masks_list = []
    for (bx, bei, bmask, *_rest) in samples:
        data_list.append(Data(
            x=torch.tensor(bx, dtype=torch.float32),
            edge_index=torch.tensor(bei, dtype=torch.long),
        ))
        masks_list.append(torch.tensor(bmask, dtype=torch.bool))

    batch = Batch.from_data_list(data_list).to(device)
    mask_cat = torch.cat(masks_list).to(device)

    net.reset_memory()
    q_batched = net.forward_batched(batch.x, batch.edge_index, batch.batch, mask_cat)
    return q_batched


def test_forward_batched_matches_per_sample(magat_net, synthetic_batch):
    """Q-values from batched forward must match per-sample forward."""
    device = "cpu"
    magat_net = magat_net.to(device)

    q_individual = _run_per_sample(magat_net, synthetic_batch, device)
    q_batched = _run_batched(magat_net, synthetic_batch, device)

    assert q_individual.shape == q_batched.shape, (
        f"Shape mismatch: individual {q_individual.shape} vs batched {q_batched.shape}"
    )
    assert torch.allclose(q_individual, q_batched, atol=1e-5), (
        f"Q-value mismatch!\n"
        f"Max abs diff: {(q_individual - q_batched).abs().max().item():.8f}\n"
        f"Mean abs diff: {(q_individual - q_batched).abs().mean().item():.8f}"
    )


def test_batched_loss_matches_per_sample_loss(magat_net, synthetic_batch):
    """Loss computed via batched path must closely match per-sample loop loss."""
    import torch.nn.functional as F

    device = "cpu"
    net = magat_net.to(device)
    gamma = 0.99

    # --- Per-sample loss (old code) ---
    losses = []
    for (bx, bei, bmask, ba, br, bnx, bnei, bnmask, bd) in synthetic_batch:
        bxt = torch.tensor(bx, dtype=torch.float32, device=device)
        bet = torch.tensor(bei, dtype=torch.long, device=device)
        bmask_t = torch.tensor(bmask, dtype=torch.bool, device=device)
        ba_t = torch.tensor(ba, dtype=torch.long, device=device)
        br_t = torch.tensor(br, dtype=torch.float32, device=device)
        bnxt = torch.tensor(bnx, dtype=torch.float32, device=device)
        bnet = torch.tensor(bnei, dtype=torch.long, device=device)
        bnmask_t = torch.tensor(bnmask, dtype=torch.bool, device=device)

        net.reset_memory()
        q_all = net(bxt, bet, bmask_t)
        q_a = q_all[torch.arange(q_all.size(0), device=device), ba_t]

        with torch.no_grad():
            net.reset_memory()
            q_next = net(bnxt, bnet, bnmask_t).max(1)[0]
            target = br_t + gamma * q_next * (1.0 - float(bd))

        active_idx = bmask_t
        if torch.any(active_idx):
            losses.append(F.mse_loss(q_a[active_idx], target[active_idx]))

    loss_per_sample = torch.stack(losses).mean()

    # --- Batched loss (new code) ---
    from torch_geometric.data import Data, Batch as PyGBatch

    data_curr = []
    data_next = []
    actions_b, rewards_b, masks_b, masks_next_b, dones_b = [], [], [], [], []

    for (bx, bei, bmask, ba, br, bnx, bnei, bnmask, bd) in synthetic_batch:
        n_nodes = bx.shape[0]
        data_curr.append(Data(
            x=torch.tensor(bx, dtype=torch.float32),
            edge_index=torch.tensor(bei, dtype=torch.long),
        ))
        data_next.append(Data(
            x=torch.tensor(bnx, dtype=torch.float32),
            edge_index=torch.tensor(bnei, dtype=torch.long),
        ))
        actions_b.append(torch.tensor(ba, dtype=torch.long))
        rewards_b.append(torch.tensor(br, dtype=torch.float32))
        masks_b.append(torch.tensor(bmask, dtype=torch.bool))
        masks_next_b.append(torch.tensor(bnmask, dtype=torch.bool))
        dones_b.append(torch.full((n_nodes,), float(bd), dtype=torch.float32))

    batch_c = PyGBatch.from_data_list(data_curr).to(device)
    batch_n = PyGBatch.from_data_list(data_next).to(device)
    act_cat = torch.cat(actions_b).to(device)
    rew_cat = torch.cat(rewards_b).to(device)
    mask_cat = torch.cat(masks_b).to(device)
    mask_next_cat = torch.cat(masks_next_b).to(device)
    done_cat = torch.cat(dones_b).to(device)

    net.reset_memory()
    q_all_b = net.forward_batched(batch_c.x, batch_c.edge_index, batch_c.batch, mask_cat)
    q_a_b = q_all_b[torch.arange(q_all_b.size(0), device=device), act_cat]

    with torch.no_grad():
        net.reset_memory()
        q_next_b = net.forward_batched(
            batch_n.x, batch_n.edge_index, batch_n.batch, mask_next_cat
        ).max(1)[0]
        target_b = rew_cat + gamma * q_next_b * (1.0 - done_cat)

    active_b = mask_cat
    loss_batched = F.mse_loss(q_a_b[active_b], target_b[active_b])

    # Note: The batched loss computes MSE over ALL alive nodes pooled together,
    # while the per-sample loss averages per-graph MSEs. These are NOT identical
    # unless all graphs have the same number of alive nodes. We verify they are
    # in the same ballpark (same order of magnitude).
    assert abs(loss_batched.item() - loss_per_sample.item()) < max(
        loss_per_sample.item() * 0.5, 0.1
    ), (
        f"Loss mismatch too large: batched={loss_batched.item():.6f} "
        f"vs per_sample={loss_per_sample.item():.6f}"
    )


def test_forward_batched_single_graph_matches_forward(magat_net):
    """A batch of 1 graph should exactly match a regular forward() call."""
    device = "cpu"
    net = magat_net.to(device)

    rng = np.random.default_rng(99)
    n_nodes = 10
    x = torch.tensor(rng.standard_normal((n_nodes, 24)).astype(np.float32))
    edge_index = torch.tensor(
        np.stack([rng.integers(0, n_nodes, 5), rng.integers(0, n_nodes, 5)]).astype(np.int64)
    )
    alive = torch.ones(n_nodes, dtype=torch.bool)

    # Regular forward
    net.reset_memory()
    q_single = net(x, edge_index, alive)

    # Batched forward with 1 graph
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    net.reset_memory()
    q_batched = net.forward_batched(batch.x, batch.edge_index, batch.batch, alive)

    assert torch.allclose(q_single, q_batched, atol=1e-5), (
        f"Single-graph mismatch: max diff = {(q_single - q_batched).abs().max().item():.8f}"
    )
