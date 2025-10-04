import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Desired enzyme parameters (from research paper)
substrates = {
    "PFLNA": {"Km_mM": 0.80, "kcat_s": 42.02, "Vmax_uM_per_min": 11.85},
}

# 2. assay settings 
S0_mM     = 0.10        # initial substrate (mM)
t_end_min = 2.0         # simulate 2.0 minutes
dt_sec    = 15.0        # sample every 15 seconds
dt_min    = dt_sec / 60.0
times     = np.arange(0.0, t_end_min + 1e-12, dt_min)  # minutes

# 3. Output directory 
outdir = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/technion/igem/Technion Team iGEM 2025- Model"
)
os.makedirs(outdir, exist_ok=True)

# 4. Create subfolders for CSVs and PNGs
csv_dir = os.path.join(outdir, "P_of_t_research_2_min_csv")
png_dir = os.path.join(outdir, "P_of_t_research_2_min_graphs_png")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)


def simulate_series(Km_mM, kcat_s, Vmax_uM_per_min, S0_mM, times):
    # 5. Unit conversions
    Km_uM    = Km_mM * 1000.0            # mM -> µM
    S0_uM    = S0_mM * 1000.0            # mM -> µM
    kcat_min = kcat_s * 60.0             # s^-1 -> min^-1
    E0_uM    = Vmax_uM_per_min / kcat_min  # from Vmax = kcat * E0

    def dP_dt_uM(P_uM):
        S_uM = S0_uM - P_uM
        if S_uM < 0.0:
            S_uM = 0.0
        return (kcat_min * E0_uM * S_uM) / (Km_uM + S_uM)  # µM/min

    # RK4 on P_uM
    P_uM = 0.0
    series_uM = [P_uM]
    v_series_uM_per_min = [dP_dt_uM(P_uM)]  # v(0)

    for _ in range(1, len(times)):
        k1 = dt_min * dP_dt_uM(P_uM)
        k2 = dt_min * dP_dt_uM(P_uM + 0.5 * k1)
        k3 = dt_min * dP_dt_uM(P_uM + 0.5 * k2)
        k4 = dt_min * dP_dt_uM(P_uM + k3)
        P_uM += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        # Clamp to [0, S0_uM]
        if P_uM < 0.0:
            P_uM = 0.0
        if P_uM > S0_uM:
            P_uM = S0_uM
        series_uM.append(P_uM)
        v_series_uM_per_min.append(dP_dt_uM(P_uM))


    return {
        "E0_uM": E0_uM,
        "Km_uM": Km_uM,
        "kcat_s": kcat_s,
        "kcat_min": kcat_min,
        "times_min": times,
        "P_series_uM": series_uM,
        "v_series_uM_per_min": v_series_uM_per_min,
    }

def plot_P_from_csv(csv_path, dt_min, out_png, title="P vs Time"):
    """Read a CSV (comma-separated P values in µM), reconstruct time with dt_min, and save P vs time plot."""
    with open(csv_path, "r") as f:
        vals = f.read().strip().split(",")
    P_uM = [float(x) for x in vals]                # µM
    t_min = [i * dt_min for i in range(len(P_uM))] # minutes

    plt.figure()
    plt.plot(t_min, P_uM, marker="o", linewidth=1)
    plt.xlabel("Time (min)")
    plt.ylabel("Product, P (µM)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# Run all substrates
for name, params in substrates.items():
    result = simulate_series(**params, S0_mM=S0_mM, times=times)

    # Write P(t) (µM) as CSV (no header, comma-separated)
    csv_path = os.path.join(csv_dir, f"P_of_t_{name}_research.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(f"{v:.6f}" for v in result["P_series_uM"]))

    # Plot P(t) from CSV
    png_path = os.path.join(png_dir, f"P_vs_time_{name}.png")
    plot_P_from_csv(csv_path, dt_min, png_path, title=f"{name}: P vs Time")

    print(f"{name}: wrote {csv_path}")
    print(f"{name}: saved {png_path}")
    print(
        "   E0={E0:.6f} µM  |  Km={Km:.1f} µM  |  kcat={kcat_s:.2f} s^-1  |  kcat_min={kcat_min:.2f} min^-1"
        .format(
            E0=result["E0_uM"],
            Km=result["Km_uM"],
            kcat_s=result["kcat_s"],
            kcat_min=result["kcat_min"],
        )
    )