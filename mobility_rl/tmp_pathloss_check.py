import numpy as np

print("=" * 60)
print("PATHLOSS ANALYSIS")
print("=" * 60)
print()
print("Formula:  P_succ = exp(-k * d^eta)")
print("Current config:  k=0.001, eta=2.0")
print("So:       P_succ = exp(-0.001 * d^2)")
print()

dists = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]

print(f"{'Dist(m)':>8} | {'P_succ':>10} | {'% pkts surviving':>18}")
print("-" * 45)
for d in dists:
    p = np.exp(-0.001 * d**2)
    print(f"{d:>7}m | {p:>10.6f} | {p*100:>17.4f}%")

print()
print("VERDICT: Pathloss is EXTREMELY aggressive!")
print("  At 30m only 41% of packets survive.")
print("  At 50m only 8.5% survive.")
print("  At 100m only 0.005% survive -> effectively 0 Mbps.")
print()
print("The data is NOT hardcoded — it's a real probabilistic")
print("simulation. Each packet's delivery is rolled against P_succ.")
print("With P_succ ~= 0.00005 at 100m, almost no packets get through.")
print()
print("=" * 60)
print("COMPARISON: What if k were smaller?")
print("=" * 60)
print()
for k_val, label in [(0.0001, "k=0.0001 (10x gentler)"), (0.00001, "k=0.00001 (100x gentler)")]:
    print(f"--- {label} ---")
    print(f"{'Dist(m)':>8} | {'P_succ':>10} | {'% pkts surviving':>18}")
    print("-" * 45)
    for d in dists:
        p = np.exp(-k_val * d**2)
        print(f"{d:>7}m | {p:>10.6f} | {p*100:>17.4f}%")
    print()
