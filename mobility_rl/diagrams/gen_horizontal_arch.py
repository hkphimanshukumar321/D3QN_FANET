"""Generate horizontal MAGAT-D3QN architecture diagram in draw.io XML format."""
import xml.etree.ElementTree as ET

# Page dimensions for wide horizontal layout
PW, PH = 2400, 500

root = ET.Element("mxfile", host="app.diagrams.net", version="29.6.6")
diagram = ET.SubElement(root, "diagram", id="MAGAT-D3QN-Horiz", name="MAGAT-D3QN Architecture")
model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="900", grid="1", gridSize="10",
                      guides="1", tooltips="1", connect="1", arrows="1", fold="1",
                      page="1", pageScale="1", pageWidth=str(PW), pageHeight=str(PH),
                      math="0", shadow="0")
rt = ET.SubElement(model, "root")
ET.SubElement(rt, "mxCell", id="0")
ET.SubElement(rt, "mxCell", id="1", parent="0")

cell_id = [100]
def nid():
    cell_id[0] += 1
    return str(cell_id[0])

# 3D slab style matching the reference image
def slab_style(fill, stroke, w, h, depth=18):
    return (f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;"
            f"darkOpacity=0.20;darkOpacity2=0.08;size={depth};"
            f"fillColor={fill};strokeColor={stroke};shadow=0;"
            f"fontFamily=Times New Roman;fontSize=11;rounded=0;")

def add_slab(x, y, w, h, fill, stroke, label, dim_label="", depth=18):
    sid = nid()
    c = ET.SubElement(rt, "mxCell", id=sid, parent="1", vertex="1",
                      style=slab_style(fill, stroke, w, h, depth),
                      value=label)
    ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    if dim_label:
        did = nid()
        d = ET.SubElement(rt, "mxCell", id=did, parent="1", vertex="1",
                          style="text;html=1;align=center;verticalAlign=bottom;fontFamily=Times New Roman;fontSize=7;fontColor=#555555;strokeColor=none;fillColor=none;",
                          value=dim_label)
        ET.SubElement(d, "mxGeometry", x=str(x-5), y=str(y-18), width=str(w+10), height=str(16), **{"as": "geometry"})
    return sid

def add_arrow(src, tgt, color="#333333", width="1.5", dashed="0"):
    aid = nid()
    c = ET.SubElement(rt, "mxCell", id=aid, parent="1", edge="1", source=src, target=tgt,
                      style=f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;strokeColor={color};strokeWidth={width};endArrow=blockThin;endFill=1;curved=1;dashed={dashed};")
    ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
    return aid

def add_text(x, y, w, h, label, size=9, color="#1a1a2e", bold=True):
    tid = nid()
    bwrap = f"<b>{label}</b>" if bold else label
    c = ET.SubElement(rt, "mxCell", id=tid, parent="1", vertex="1",
                      style=f"text;html=1;align=center;verticalAlign=middle;fontFamily=Times New Roman;fontSize={size};fontColor={color};strokeColor=none;fillColor=none;",
                      value=bwrap)
    ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    return tid

def add_circle(x, y, r, label, fill="#ffffff", stroke="#333333"):
    cid = nid()
    c = ET.SubElement(rt, "mxCell", id=cid, parent="1", vertex="1",
                      style=f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontFamily=Times New Roman;fontSize=12;fontStyle=1;shadow=0;",
                      value=label)
    ET.SubElement(c, "mxGeometry", x=str(x-r), y=str(y-r), width=str(2*r), height=str(2*r), **{"as": "geometry"})
    return cid

# ═══════════════════════════════════════════════════
# Layout: horizontal left-to-right, y_center=220
# ═══════════════════════════════════════════════════
Y0 = 100  # top of main slab row
GAP = 22  # horizontal gap between slabs

# --- A. Input ---
x = 30
inp = add_slab(x, Y0+30, 55, 180, "#e1d5e7", "#9673a6",
               "<b>Input</b><br><font style='font-size:10px;'>x∈ℝ<sup>N×8</sup></font>", "N×8", depth=14)

x += 55 + GAP
edge_inp = add_slab(x, Y0+60, 40, 120, "#f3e5f5", "#9673a6",
                    "<b>Edge<br>Index</b><br><font style='font-size:10px;'>2×E</font>", "2×E", depth=10)

# --- B. GAT Layer 1 (4 heads, shown as 4 parallel slabs) ---
x += 40 + GAP + 15
# Operator circle
op1 = add_circle(x+5, Y0+120, 12, "⊕")
x += 30

gat1_ids = []
colors_gat = ["#f8cecc", "#fce4ec", "#ffcdd2", "#ef9a9a"]  # red tones for 4 heads
for i in range(4):
    sid = add_slab(x + i*18, Y0+20+i*8, 45, 200-i*16, colors_gat[i], "#b85450",
                   "" if i > 0 else "<font style='font-size:10px;'>GAT<br>H1-4</font>",
                   f"64" if i == 0 else "", depth=12)
    gat1_ids.append(sid)
add_text(x-5, Y0-20, 100, 16, "GATConv L1: 8→256", size=7, bold=False, color="#b85450")
add_text(x-5, Y0+225, 100, 14, "4 heads, concat", size=6, bold=False, color="#888888")

x += 4*18 + GAP
# ELU activation
elu1 = add_slab(x, Y0+50, 22, 140, "#ffecb3", "#f57f17", "<font style='font-size:10px;'>ELU</font>", "", depth=8)

# --- GAT Layer 2 (single head) ---
x += 22 + GAP
op2 = add_circle(x+5, Y0+120, 12, "⊕")
x += 30
gat2 = add_slab(x, Y0+40, 45, 160, "#f8cecc", "#b85450",
                "<font style='font-size:10px;'>GAT<br>L2</font>", "N×64", depth=14)
add_text(x-8, Y0+210, 70, 14, "256→64, 1 head", size=6, bold=False, color="#888888")

x += 45 + GAP
elu2 = add_slab(x, Y0+50, 22, 140, "#ffecb3", "#f57f17", "<font style='font-size:10px;'>ELU</font>", "", depth=8)

# --- C. GRU ---
x += 22 + GAP + 5
op3 = add_circle(x+5, Y0+120, 12, "→")
x += 30
gru = add_slab(x, Y0+30, 60, 180, "#d5e8d4", "#82b366",
               "<b>GRU</b><br><font style='font-size:10px;'>64→64<br>h<sub>t-1</sub>→h<sub>t</sub></font>", "N×64", depth=16)
# Recurrence loop arrow
loop_id = nid()
c = ET.SubElement(rt, "mxCell", id=loop_id, parent="1", edge="1", source=gru, target=gru,
                  style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#82b366;strokeWidth=1.5;endArrow=blockThin;endFill=1;curved=1;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.8;entryY=1;entryDx=0;entryDy=0;")
g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
arr = ET.SubElement(g, "Array", **{"as": "points"})
ET.SubElement(arr, "mxPoint", x=str(x+30), y=str(Y0+245))
ET.SubElement(arr, "mxPoint", x=str(x+48), y=str(Y0+245))

add_text(x+10, Y0+248, 50, 12, "h<sub>t</sub>", size=7, bold=False, color="#82b366")

# --- D. Split point ---
x += 60 + GAP + 10
split = add_circle(x, Y0+120, 8, "", "#1a1a2e", "#1a1a2e")

# --- Value stream (top branch) ---
vx = x + 25
v1 = add_slab(vx, Y0-10, 40, 110, "#f0e6ff", "#7c3aed",
              "<font style='font-size:10px;'>FC<br>64→64<br>ReLU</font>", "64", depth=12)
vx += 40 + 12
v2 = add_slab(vx, Y0+10, 30, 70, "#e8d5ff", "#7c3aed",
              "<font style='font-size:10px;'>FC<br>64→1</font>", "1", depth=10)
add_text(vx-30, Y0-30, 90, 14, "V(s) stream", size=7, bold=True, color="#7c3aed")

# --- Advantage stream (bottom branch) ---
ax_s = x + 25
a1 = add_slab(ax_s, Y0+140, 40, 110, "#ffe0b2", "#e65100",
              "<font style='font-size:10px;'>FC<br>64→64<br>ReLU</font>", "64", depth=12)
ax_s += 40 + 12
a2 = add_slab(ax_s, Y0+155, 30, 80, "#ffcc80", "#e65100",
              "<font style='font-size:10px;'>FC<br>64→2</font>", "2", depth=10)
add_text(ax_s-30, Y0+270, 90, 14, "A(s,a) stream", size=7, bold=True, color="#e65100")

# --- Dueling aggregation ---
agg_x = ax_s + 30 + GAP + 10
agg = add_circle(agg_x, Y0+120, 16, "+", "#e8eaf6", "#3949ab")
add_text(agg_x-50, Y0+145, 100, 20, "Q = V + [A − mean(A)]", size=6, bold=False, color="#3949ab")

# --- Q output ---
qx = agg_x + 30 + GAP
q_out = add_slab(qx, Y0+50, 50, 140, "#cce5ff", "#3949ab",
                 "<b>Q<sub>i</sub></b><br><font style='font-size:10px;'>ℝ<sup>N×2</sup><br>[TDMA,<br>CSMA]</font>", "N×2", depth=14)

# --- Action selection ---
act_x = qx + 50 + GAP
action = add_slab(act_x, Y0+60, 45, 120, "#bbdefb", "#1565c0",
                  "<font style='font-size:10px;'><b>ε-greedy</b><br>argmax<br>a<sub>i</sub></font>", "a∈{0,1}", depth=12)

# --- QMIX mixer (below main flow) ---
mix_y = Y0 + 310
mx = qx - 30
add_text(mx-10, mix_y-25, 250, 18, "QMIX Mixing Network (Centralised Training Only)", size=8, bold=True, color="#2e7d32")

gs = add_slab(mx, mix_y, 55, 70, "#e1d5e7", "#9673a6",
              "<font style='font-size:10px;'><b>Global<br>State s</b></font>", "", depth=10)
hw1 = add_slab(mx+75, mix_y, 50, 70, "#fff9c4", "#f57f17",
               "<font style='font-size:10px;'><b>W<sub>1</sub></b><br>HyperNet</font>", "", depth=10)
hw2 = add_slab(mx+145, mix_y, 50, 70, "#fff9c4", "#f57f17",
               "<font style='font-size:10px;'><b>W<sub>2</sub></b><br>HyperNet</font>", "", depth=10)
qt = add_slab(mx+215, mix_y+5, 55, 60, "#c8e6c9", "#2e7d32",
              "<font style='font-size:10px;'><b>Q<sub>tot</sub></b></font>", "", depth=10)

# QMIX boundary box
qmix_box_id = nid()
c = ET.SubElement(rt, "mxCell", id=qmix_box_id, parent="1", vertex="1",
                  style="rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#2e7d32;strokeWidth=1.5;dashed=1;dashPattern=6 3;",
                  value="")
ET.SubElement(c, "mxGeometry", x=str(mx-15), y=str(mix_y-30), width=str(300), height=str(115), **{"as": "geometry"})

# Per-agent boundary box
pa_box_id = nid()
c = ET.SubElement(rt, "mxCell", id=pa_box_id, parent="1", vertex="1",
                  style="rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#1a1a2e;strokeWidth=1.5;dashed=1;dashPattern=8 4;",
                  value="")
ET.SubElement(c, "mxGeometry", x="20", y=str(Y0-40), width=str(act_x+60), height=str(300), **{"as": "geometry"})
add_text(22, Y0+260, 320, 16, "Per-Agent Network (Shared Weights — Decentralised Execution)", size=8, bold=True, color="#1a1a2e")

# ═══════════════════════════════════════════════════
# Arrows (left to right flow)
# ═══════════════════════════════════════════════════
add_arrow(inp, op1, "#9673a6")
add_arrow(edge_inp, op1, "#9673a6", "1", "1")
add_arrow(op1, gat1_ids[0], "#333333")
add_arrow(gat1_ids[-1], elu1, "#333333")
add_arrow(elu1, op2, "#333333")
add_arrow(edge_inp, op2, "#9673a6", "1", "1")
add_arrow(op2, gat2, "#333333")
add_arrow(gat2, elu2, "#333333")
add_arrow(elu2, op3, "#333333")
add_arrow(op3, gru, "#333333")
add_arrow(gru, split, "#333333")
add_arrow(split, v1, "#7c3aed")
add_arrow(split, a1, "#e65100")
add_arrow(v2, agg, "#7c3aed")
add_arrow(a2, agg, "#e65100")
add_arrow(agg, q_out, "#3949ab")
add_arrow(q_out, action, "#1565c0")
# QMIX arrows
add_arrow(q_out, hw1, "#3949ab", "1.5", "1")
add_arrow(gs, hw1, "#9673a6", "1", "1")
add_arrow(gs, hw2, "#9673a6", "1", "1")
add_arrow(hw1, qt, "#f57f17")
add_arrow(hw2, qt, "#f57f17")
add_arrow(v1, v2, "#7c3aed")
add_arrow(a1, a2, "#e65100")

# Title
add_text(30, 20, 800, 30, "MAGAT-D3QN: Multi-Agent Graph Attention Dueling Double Deep Q-Network", size=14, color="#1a1a2e")
add_text(30, 48, 600, 18, "with QMIX Centralised Training &amp; Decentralised Execution (CTDE)", size=9, bold=False, color="#666666")

# Write output
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
out = r"c:\Users\hkphi\OneDrive\Desktop\mobility_rl\diagrams\MAGAT_D3QN_Architecture.drawio"
tree.write(out, encoding="unicode", xml_declaration=False)
print(f"Written horizontal architecture to {out}")
