"""Render the MAGAT-D3QN architecture as a horizontal slab diagram (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",  # Use Computer Modern for math
    "text.usetex": False,      # Keep False to avoid dependency issues, but cm works well
})

fig, ax = plt.subplots(figsize=(16, 4.2))
ax.set_xlim(-0.5, 21)
ax.set_ylim(-1.8, 4.5)
ax.axis("off")
ax.set_aspect("equal")

def draw_slab(ax, x, y, w, h, depth, facecolor, edgecolor, label="", dim="", alpha=0.85):
    """Draw a 3D slab (front face + top face + right face)."""
    dx, dy = depth * 0.35, depth * 0.35
    # Front face
    front = plt.Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                        facecolor=facecolor, edgecolor=edgecolor, lw=0.8, alpha=alpha, zorder=2)
    ax.add_patch(front)
    # Top face (lighter)
    import matplotlib.colors as mc
    top_c = mc.to_rgba(facecolor)
    top_c = tuple(min(1, c*1.15) for c in top_c[:3]) + (alpha*0.7,)
    top = plt.Polygon([(x, y+h), (x+dx, y+h+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                      facecolor=top_c, edgecolor=edgecolor, lw=0.6, zorder=3)
    ax.add_patch(top)
    # Right face (darker)
    right_c = tuple(max(0, c*0.8) for c in mc.to_rgba(facecolor)[:3]) + (alpha*0.6,)
    right = plt.Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                        facecolor=right_c, edgecolor=edgecolor, lw=0.6, zorder=3)
    ax.add_patch(right)
    # Label
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=6,
                fontfamily="serif", zorder=4, fontweight="bold")
    # Dimension annotation
    if dim:
        ax.text(x+w/2+dx/2, y+h+dy+0.15, dim, ha="center", va="bottom",
                fontsize=5.5, fontfamily="serif", color="#555555", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=5)

def draw_circle(ax, x, y, r, label, fc="white", ec="#333"):
    c = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=1.0, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold",
            fontfamily="serif", zorder=6)

# ═══════════════ LAYOUT (left to right) ═══════════════
cx = 0.0

# --- INPUT ---
draw_slab(ax, cx, 0, 0.6, 2.8, 0.25, "#e1d5e7", "#9673a6", "Input\n$x \in \mathbb{R}^{N \\times 8}$", "$N \\times 8$")
inp_x = cx + 0.6

cx += 1.0
draw_slab(ax, cx, 0.3, 0.45, 2.2, 0.2, "#f3e5f5", "#9673a6", "Edge\nIndex", "$2 \\times E$")
edge_x = cx + 0.45

# Arrow
draw_arrow(ax, inp_x+0.08, 1.4, cx-0.08, 1.4, "#9673a6")

# --- GAT LAYER 1 (4 heads as parallel slabs) ---
cx += 0.85
draw_circle(ax, cx, 1.4, 0.18, "$\\oplus$", "#f5f5f5", "#b85450")
draw_arrow(ax, edge_x+0.08, 1.4, cx-0.2, 1.4, "#9673a6", 0.8, "-|>")

cx += 0.45
gat1_colors = ["#f8cecc", "#fce4ec", "#ffcdd2", "#ef9a9a"]
for i in range(4):
    draw_slab(ax, cx + i*0.22, 0.1+i*0.12, 0.35, 2.6-i*0.24, 0.15,
              gat1_colors[i], "#b85450", "" if i > 0 else "GAT\nH1-4", "64" if i==0 else "")
ax.text(cx+0.55, -0.3, "GATConv L1\n$8 \\to 256$, 4 heads", ha="center", fontsize=5.5,
        fontfamily="serif", color="#b85450")
gat1_end = cx + 4*0.22 + 0.35

cx = gat1_end + 0.25
draw_slab(ax, cx, 0.4, 0.28, 2.0, 0.12, "#ffecb3", "#f57f17", "ELU", "")
elu1_end = cx + 0.28

# --- GAT LAYER 2 ---
cx = elu1_end + 0.35
draw_circle(ax, cx, 1.4, 0.18, "$\\oplus$", "#f5f5f5", "#b85450")
draw_arrow(ax, elu1_end+0.05, 1.4, cx-0.2, 1.4)

cx += 0.4
draw_slab(ax, cx, 0.3, 0.5, 2.2, 0.2, "#f8cecc", "#b85450", "GAT\nL2", "$N \\times 64$")
ax.text(cx+0.35, -0.3, "GATConv L2\n$256 \\to 64$, 1 head", ha="center", fontsize=5.5,
        fontfamily="serif", color="#b85450")
gat2_end = cx + 0.5

cx = gat2_end + 0.25
draw_slab(ax, cx, 0.4, 0.28, 2.0, 0.12, "#ffecb3", "#f57f17", "ELU", "")
draw_arrow(ax, gat2_end+0.05, 1.4, cx-0.05, 1.4)
elu2_end = cx + 0.28

# --- GRU ---
cx = elu2_end + 0.5
draw_arrow(ax, elu2_end+0.05, 1.4, cx-0.05, 1.4)
draw_slab(ax, cx, 0.15, 0.7, 2.5, 0.25, "#d5e8d4", "#82b366", "GRU\n$64 \\to 64$", "$N \\times 64$")
# Recurrence loop
loop_y = -0.25
ax.annotate("", xy=(cx+0.55, 0.15), xytext=(cx+0.35, 0.15),
            arrowprops=dict(arrowstyle="-|>", color="#82b366", lw=1.0,
                           connectionstyle="arc3,rad=-0.5"))
ax.text(cx+0.45, loop_y, "$h_t$", ha="center", fontsize=6, color="#82b366",
        fontfamily="serif", style="italic")
gru_end = cx + 0.7

# --- SPLIT ---
cx = gru_end + 0.4
draw_arrow(ax, gru_end+0.08, 1.4, cx-0.12, 1.4)
draw_circle(ax, cx, 1.4, 0.1, "", "#1a1a2e", "#1a1a2e")
split_x = cx

# --- VALUE STREAM (top) ---
vx = split_x + 0.4
draw_arrow(ax, split_x+0.12, 1.7, vx-0.05, 2.6, "#7c3aed", 1.0)
draw_slab(ax, vx, 2.1, 0.45, 1.5, 0.18, "#f0e6ff", "#7c3aed", "FC\n$64 \\to 64$\nReLU", "64")
vx2 = vx + 0.45 + 0.2
draw_arrow(ax, vx+0.48, 2.85, vx2-0.05, 2.85, "#7c3aed")
draw_slab(ax, vx2, 2.3, 0.35, 1.1, 0.15, "#e8d5ff", "#7c3aed", "FC\n$64 \\to 1$", "1")
ax.text(vx+0.3, 3.85, "$V(s)$ stream", ha="center", fontsize=7,
        fontfamily="serif", color="#7c3aed", fontweight="bold")
v_end_x = vx2 + 0.35

# --- ADVANTAGE STREAM (bottom) ---
ax2 = split_x + 0.4
draw_arrow(ax, split_x+0.12, 1.1, ax2-0.05, 0.5, "#e65100", 1.0)
draw_slab(ax, ax2, -0.4, 0.45, 1.5, 0.18, "#ffe0b2", "#e65100", "FC\n$64 \\to 64$\nReLU", "64")
ax2b = ax2 + 0.45 + 0.2
draw_arrow(ax, ax2+0.48, 0.35, ax2b-0.05, 0.35, "#e65100")
draw_slab(ax, ax2b, -0.2, 0.35, 1.2, 0.15, "#ffcc80", "#e65100", "FC\n$64 \\to 2$", "2")
ax.text(ax2+0.3, -0.8, "$A(s,a)$ stream", ha="center", fontsize=7,
        fontfamily="serif", color="#e65100", fontweight="bold")
a_end_x = ax2b + 0.35

# --- DUELING AGGREGATION ---
agg_x = max(v_end_x, a_end_x) + 0.5
draw_circle(ax, agg_x, 1.4, 0.22, "+", "#e8eaf6", "#3949ab")
draw_arrow(ax, v_end_x+0.05, 2.85, agg_x-0.1, 1.65, "#7c3aed")
draw_arrow(ax, a_end_x+0.05, 0.35, agg_x-0.1, 1.18, "#e65100")
ax.text(agg_x, 0.85, r"$Q = V + [A - \overline{A}]$", ha="center", fontsize=7,
        fontfamily="serif", color="#3949ab")

# --- Q OUTPUT ---
qx = agg_x + 0.5
draw_arrow(ax, agg_x+0.24, 1.4, qx-0.05, 1.4, "#3949ab")
draw_slab(ax, qx, 0.3, 0.55, 2.2, 0.2, "#cce5ff", "#3949ab",
          r"$Q_i$" + "\n" + r"$\in \mathbb{R}^{N \times 2}$" + "\n[TDMA,\nCSMA]", r"$N \times 2$")
q_end = qx + 0.55

# --- ACTION SELECTION ---
act_x = q_end + 0.4
draw_arrow(ax, q_end+0.05, 1.4, act_x-0.05, 1.4, "#1565c0")
draw_slab(ax, act_x, 0.5, 0.5, 1.8, 0.18, "#bbdefb", "#1565c0",
          r"$\epsilon$-greedy" + "\nargmax\n$a_i$", r"$a \in \{0,1\}$")

# --- QMIX (below main flow) ---
mix_y = -1.5
mix_x = qx - 0.3
ax.text(mix_x, mix_y+0.55, "QMIX Mixer (Training Only)", ha="left", fontsize=6,
        fontfamily="serif", color="#2e7d32", fontweight="bold")
draw_slab(ax, mix_x, mix_y-0.3, 0.5, 0.6, 0.15, "#e1d5e7", "#9673a6", "Global\nState $s$", "")
draw_slab(ax, mix_x+0.7, mix_y-0.3, 0.45, 0.6, 0.12, "#fff9c4", "#f57f17", "$W_1$\nHyper", "")
draw_slab(ax, mix_x+1.35, mix_y-0.3, 0.45, 0.6, 0.12, "#fff9c4", "#f57f17", "$W_2$\nHyper", "")
draw_slab(ax, mix_x+2.0, mix_y-0.2, 0.5, 0.5, 0.15, "#c8e6c9", "#2e7d32", "$Q_{tot}$", "")
# QMIX boundary
rect = mpatches.FancyBboxPatch((mix_x-0.15, mix_y-0.5), 2.85, 1.2,
                                boxstyle="round,pad=0.05", facecolor="none",
                                edgecolor="#2e7d32", lw=1.0, ls="--", zorder=1)
ax.add_patch(rect)
# Arrow from Q to QMIX
draw_arrow(ax, qx+0.3, 0.28, mix_x+0.85, mix_y+0.35, "#3949ab", 0.8, "-|>")

# --- Per-agent boundary ---
pa_rect = mpatches.FancyBboxPatch((-0.3, -0.6), act_x+0.85, 4.8,
                                   boxstyle="round,pad=0.08", facecolor="none",
                                   edgecolor="#1a1a2e", lw=1.2, ls=(0, (8, 4)), zorder=0)
ax.add_patch(pa_rect)
ax.text(0.0, 4.15, "Per-Agent Network (Shared Weights — Decentralised Execution)",
        fontsize=7, fontfamily="serif", color="#1a1a2e", fontweight="bold")

# Title
ax.text(10.5, 4.4, "MAGAT-D3QN: Multi-Agent Graph Attention Dueling Double Deep Q-Network",
        ha="center", fontsize=11, fontfamily="serif", fontweight="bold", color="#1a1a2e")

plt.tight_layout()
out_dir = r"c:\Users\hkphi\OneDrive\Desktop\mobility_rl\alternate_tj_latex_template_ap\figures"
fig.savefig(f"{out_dir}/early_magat_architecture.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out_dir}/early_magat_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()
print("Saved early_magat_architecture.pdf/png")
