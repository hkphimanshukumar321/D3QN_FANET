# Decentralization Redesign Implementation Plan

This document provides the complete, sequential blueprint to migrate the UAV/FANET repository from a globally pooled MAC-selection framework into a true decentralized cluster-based interference graph control study.

---

## 1. Architectural Audit

**A. Current Implementation Files:**
- **Current MARL env:** `envs/marl_mac_env.py` (Local observations, but global action via majority vote execution).
- **Current SARL env/wrapper:** `envs/marl_sarl_wrapper.py` (Mean-pools the 16-D local state into 14-D, broadcasts a single SARL action to all. Not a true central graph formulation).
- **Current reward logic:** End of `step()` in `marl_mac_env.py`. Fully centralized, global latency, global throughput. All agents get the identical scalar cooperative reward.
- **Current MAGAT model:** `algorithms/rl/gnn_marl.py` (Good GAT + GRU setup, uses CTDE via QMIX).
- **Current MCA-D3QN model:** `algorithms/rl/custom_mca_d3qn.py` (Used natively as a SARL/Central model).
- **Current training & evaluation loop:** `experiments/run_unified_experiment.py` (Runs the environment over steps, tracks identical logging across MARL and SARL).

**B. Where Centralization Still Exists:**
- **Action Execution:** The majority vote in `marl_mac_env.py:step()` completely centralizes execution (`chosen_mac = 1 if vote_csma > vote_tdma else 0`). Even independent agents vote for a *global* MAC setting applied to everyone.
- **Reward Assignment:** All agents receive identical global rewards based on the aggregate performance of the shared `Config`, preventing agents from learning niche localized topological actions.
- **MAC Simulator Context:** The MAC simulation context (`simulate_tdma` or `simulate_csma_ca`) is run once globally per step, not per cluster.
- **SARL Wrapper:** The current wrapper pools data with `mean()` rather than exposing the full joint distribution, masking true centralization.

**C. What Can Be Retained:**
- **Mobility and Channel Models:** `mobility/`, `link.py`, fading models, SpeedEngine.
- **MAGAT Architecture:** `gnn_marl.py` handles GAT and GRU nicely. QMIX logic is perfect for CTDE given local rewards.
- **Logging Pipeline:** `experiment_tracking.py` and WandB/JSON/Optuna integrations.

**D. What Must Be Modified or Replaced:**
- **Deprecate:** `marl_sarl_wrapper.py` (It is a hack, not a true centralized controller).
- **Modify:** `marl_mac_env.py` -> Must simulate clusters. It needs to calculate disjoint cluster queues/nodes and run MAC simulators independently per cluster.
- **Create New:** 
  - `envs/clustering.py` (Dynamic cluster graph builder and state tracker).
  - `envs/sarl_central_env.py` (Dedicated central SARL with global action over $c$ dimensions).
  - `algorithms/rl/rewards.py` (Isolated reward mathematical mapping).

---

## 2. Mathematical Formulation

**1. UAV Set:** 
There are $n$ UAVs: $U(t) = \{1, 2, ..., n\}$.
State per UAV: position $p_i(t)$, velocity $v_i(t)$, queue $q_i(t)$, offered load $\lambda_i(t)$, channel estimate $\hat{\gamma}_i(t)$, collision history $\kappa_i(t)$, residual energy $E_i(t)$, optional urgency $A_i(t)$.

**2. Dynamic Clusters:** 
$c(t)$ clusters partitioning the network: $U(t) = C_1(t) \cup C_2(t) \cup ... \cup C_c(t)(t)$. Exactly one cluster per UAV. Exactly one leader (cluster-head) $L_k(t) \in C_k(t)$ per cluster.

**3. Cluster Membership Score:** 
Assigned via: $\text{cluster}(i,t) = \text{argmax}_k S_{i,k}(t)$
Where $S_{i,k}(t) = w_d f_d(i,k,t) + w_s f_s(i,k,t) + w_m f_m(i,k,t) + w_q f_q(i,k,t)$
Subject to hysterisis thresholds $\theta_{join}$ and $\theta_{leave}$.

**4. Radii:** 
$R_c$: Membership coordination radius.
$R_I$: Inter-cluster interference range ($R_I \geq R_c$).

**5. Cluster Interference Graph:** 
$G_t = (V_t, E_t)$ where $V_t = \{1, 2, ..., c(t)\}$. 
Edges $E_t$: $(k, m) \in E_t \iff ||p_{L_k}(t) - p_{L_m}(t)|| \leq R_I$ or $\text{SINR}_{k,m}(t) \leq \tau_I$.

**6. Agents:**
The decentralized agents are the cluster-heads $L_k(t)$. Number of agents equals $c(t)$.

**7. Observation Space:** 
Local observation for head $k$: 
$o_k(t) = [|C_k(t)|, q_k(t), d_k^{HOL}(t), \lambda_k(t), \hat{\gamma}_k(t), \kappa_k(t), E_k(t), v_k(t), \text{deg}_k(t), \text{summary}(N_k(t)), \text{health}_k(t), \text{handover}_k(t)]$

**8. Action Space:** 
Local action per head: $a_k(t) = (m_k(t), ch_k(t), \omega_k(t))$. Joint vector is $a(t) = [a_1(t), ..., a_c(t)]$. No majority voting bottleneck.

**9. Dec-POMDP & CTDE:**
This framework is a Dec-POMDP (Decentralized Partially Observable Markov Decision Process). Training via MAGAT uses QMIX: a Centralized Training Decentralized Execution (CTDE) mechanism.
- The majority-vote old design is mathematically just a multi-headed SARL agent with a bizarre reduction step. It lacked per-agent degrees of freedom. This paradigm fixes it.

**10. Reward Structure:** 
Local Reward: $r_k(t) = \alpha T_k(t) - \beta D_k(t) - \gamma C_k(t) - \delta Ecost_k(t) - \eta I_k(t) - \xi A_k(t) - \zeta H_k(t)$
Team objective: $R_{team}(t) = \sum r_k(t) + \lambda J(t)$

**11. Fault Tolerance Formulation:** 
On failure, $L_k(t+) = \text{argmax}_{u \in C_k(t) \setminus F_k(t)} H_u(t)$ based on $H$ (energy, centrality, mobility stability, queue backlog).

---

## 3. Clustering Subsystem Design

*   **Initialization:** K-Means on initial spatial coordinates to group $n$ nodes into $c_{start}$ clusters. Evaluate K-Means centroids to pick the initial $c_{start}$ cluster heads.
*   **Bounding $c(t)$:** Adaptive, but securely bounded within $[c_{min}, c_{max}]$ (e.g., $c_{min}=3, \ c_{max}=10$) to prevent fragmentation or total consolidation. Note: MARL input dimensions must use padding/masking to accommodate variable $c(t)$ agents if using standard PettingZoo.
*   **Assignment & Reassociation:** Checked every $T_{cluster}$ interval. An edge node calculates $S_{i,m}(t)$ for neighboring cluster $m$. Associates with $m$ if $S_{i,m} - S_{i,k} > (\theta_{join} - \theta_{leave})$.
*   **Split:** If member count $> N_{max}$, isolate the subset of nodes furthest from the head and spawn a new cluster head (if $c < c_{max}$).
*   **Merge:** If member count $< N_{min}$, flag the cluster. Head negotiates merge with closest friendly head. Members merge.
*   **Cluster Head Selection & Health:** $health_k \propto E_R \cdot \frac{1}{\Delta v}$. If health dips below threshold $\theta_{health}$ or dies (energy 0), trigger `handover_flag`.
*   **Summary:** Node $k$'s neighbors' summaries $\text{summary}(N_k(t))$ are a masked aggregate of neighboring $|C_m(t)|$ and queues where bounding graph edges exist.

**Sanity Check:** Code must ensure $\sum|C_k(t)| == n$ always (no orphan nodes, no duplicates).

---

## 4. Environment Redesign

**1. MARL MAC Env Transformation (`envs/marl_mac_env.py`):**
- Drop the global `votes -> majority -> simulate_csma_ca` pipeline.
- Step signature: takes `actions: dict` (dict of size $c$, mapped by cluster head IDs).
- The environment spins up $c(t)$ separate `simulate_csma` or `simulate_tdma` isolated MAC sub-simulator blocks per step, mapping the specific UAV sub-sets into those discrete simulators.
- Inter-cluster interference: If two clusters use the same MAC channel/class and are connected by the $R_I$ graph, an interference heuristic artificially increases packet drops or collisions in their respective local simulators.
- Observations returned are exactly $o_k(t)$, per active cluster head. Note: If using `PettingZoo`, inactive agents are mapped as dead/terminated, or we pad to $c_{max}$.

**2. Sanity Checks for the Environment:**
- Two adjacent clusters choosing TDMA with the same timescale *must* exhibit modeled inter-cluster interference unless orthogonal channels are implemented.
- The outcome of UAV 1 must depend on Cluster 2's action if they are within $R_I$.

---

## 5. Reward Redesign

- **Isolated Rewards Function (`algorithms/rl/rewards.py`):**
  - **MARL:** Receives local components directly from the sub-MAC simulator for cluster $k$.
    $$r_k(t) = \alpha T_k - \beta D_k - \gamma C_k - \delta \Delta E_k - \eta (\sum_{m \in N_k} I_{k,m})$$
  - **SARL:** Computes scalar `R_SARL = sum(r_k) + Jain_Index([T_k])`.
  - **Normalization:** Cap delay to a maximum $D_{max}$ to prevent exploding negatives. Normalize throughput by maximum theoretical capacity.
  - **Differences:** Giving MARL raw identical team reward causes the lazy agent problem. MARL uses strictly private $r_k$, requiring QMIX to stitch them during CTDE. SARL directly sees the J(t) fairness impact.

---

## 6. Model-by-Model Compatibility Reviews

1. **MAGAT-D3QN:** Remains highly valid. Node inputs are now the cluster heads ($c(t)$ notes). The Edge Index perfectly mirrors $G_t$ inter-cluster graph. Handles variable $c$ via masking.
2. **IQL / VDN / QMIX:** All compatible. QMIX handles CTDE over the $c(t)$ rewards.
3. **MCA-D3QN:** Invalidated as a MARL concept. Must be repurposed purely as the Centralized SARL Baseline model (taking the joint state).
4. **SB3 DQN/PPO:** Compatible with the dedicated SARL env.
5. **Tabular baseline:** Requires aggressive state quantization. Valid for very small $c$.

*(Scientific Note: QMIX relies on static number of agents in the standard Pymarl framework. To handle variable $c(t)$, we must instantiate exactly $c_{max}$ agents and pass a death mask to the mixer for inactive clusters to zero their outputs.)*

---

## 7. File-by-File Migration Plan

*   **`envs/marl_mac_env.py`**: **[Modify heavily]**. Strip out global votes. Implement independent sub-MAC tracking.
*   **`envs/marl_sarl_wrapper.py`**: **[Deprecate]**. Replace with `envs/sarl_central_env.py`.
*   **`envs/sarl_central_env.py`**: **[New]**. Provides full global joint observation of all clusters. Takes multidiscrete action vector.
*   **`envs/clustering.py`**: **[New]**. Class to manage node assignment matrix, splits/merges, and inter-cluster distance graphs.
*   **`algorithms/rl/rewards.py`**: **[New]**. Pure mathematical reward evaluation logic for local vs global.
*   **`algorithms/rl/gnn_marl.py`**: **[Modify]**. Add observation masking logic for dead/inactive clusters.
*   **`algorithms/rl/features_extractor.py`**: **[Modify]**. Must parse cluster-level states, not global UAV states.
*   **`configs/marl_config.py`**: **[Modify]**. Add `R_c`, `R_I`, `C_MAX`, `C_MIN`, thresholds.
*   **`experiments/run_unified_experiment.py`**: **[Modify]**. Change RL setup to use correct env mappings. Track cluster sizes in WandB.

---

## 8. Experimental Protocol for Publishability

1. **Main Benchmark:** MAGAT-D3QN vs MCA-D3QN (Central) vs VDN vs Tabular.
2. **Topology-Sensitivity:** Keep traffic load static, but switch between uniform grid, random scattering, and two dense "hotspots". (Proves graph awareness works).
3. **Scalability:** Scale $n$ from 20 to 100. (Shows Central SARL collapsing due to action space dimensionality, while MARL holds steady).
4. **Robustness-to-Mismatch:** Train on $\tau_I = high$, Test on $\tau_I = low$. Is the policy overly brittle?
5. **Leader-Failure Robustness:** At $t=500$, force-fail $50\%$ of cluster heads. Measure recovery time and post-failure fairness (Jain's index).
6. **Clustering Sanity:** Plot $c(t)$ over time to prove the network doesn't helplessly oscillate or shatter into $N$ clusters.

---

## 9. Implementation Order

> [!CAUTION]
> Deviating from this order guarantees integration hell. Follow strictly sequentially.

*   **Phase 1: Clustering Core (No RL yet).** Build `clustering.py`. Given a set of $N$ moving nodes, verify they split, merge, and output valid graphs.
*   **Phase 2: MARL Env Redesign.** Rewire `marl_mac_env.py` to ingest the Phase 1 clusters and run parallel MACs.
*   **Phase 3: Reward Math.** Implement the private metrics inside the new environment steps.
*   **Phase 4: New SARL Baseline Env.** Build `sarl_central_env.py`.
*   **Phase 5: MAGAT & Model Patches.** Add masking for $c \leq c_{max}$ to MAGAT. Repurpose MCA-D3QN.
*   **Phase 6: Failover Logic.** Add the leader death simulation.
*   **Phase 7: Training & Plots.** Execute unified script and generate tradeoffs.

---

## 10. Sanity Check Verdict

1. **Is this now truly decentralized?** YES. Cluster actions are functionally independent, no global vote exists.
2. **Is SARL still a fair baseline?** YES. By providing a true central joint-action environment, SARL becomes an honest algorithmic upper-bound mapping to Central Control.
3. **Does MAGAT have a reason to win?** YES. The Graph explicitly encodes the inter-cluster interference radius; the model learns spatial isolation.
4. **Hidden centralization leaks?** *Risk*: Passing the full proximity graph to every local agent. MAGAT must only process the neighbor-edges, not the entire $N \times N$ matrix for the local node.
5. **Cluster-pathology risks?** *Risk*: Oscillating ping-pong memberships. Strictly enforce $\theta_{join} >> \theta_{leave}$.
6. **Final Comparisons defensible?** YES. This isolates graph-awareness versus mean-field approximation, proving structural knowledge enhances local autonomy.

> [!WARNING]
> The biggest scientific risk remaining is dealing with PettingZoo's fixed-agent requirement. If $c(t)$ changes dynamically, RL arrays will throw size errors. We must enforce padded `c_max` arrays, passing `dead` bitmasks to the models to ignore inactive empty clusters.
