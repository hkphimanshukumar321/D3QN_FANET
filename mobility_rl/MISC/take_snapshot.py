import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.marl_mac_env import MARLMacEnv
from utils.visualize_fanet import run_episode, FANETAnimator3D

env = MARLMacEnv()
rec = run_episode(env, max_steps=50, seed=42)

animator = FANETAnimator3D(rec, figsize=(10, 8))
animator._setup_figure()
# Add extra padding at the top for the title
animator.fig.subplots_adjust(top=0.92)
animator._update_frame(49)

# Change background to white just in case
animator.fig.patch.set_facecolor("white")
animator.ax.set_facecolor("white")

plt.savefig("alternate_tj_latex_template_ap/figures/early_simulation_snapshot.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved snapshot.")
