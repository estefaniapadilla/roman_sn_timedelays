"""
compare_fit_modes.py
=====================
Head-to-head comparison of the three delay-measurement methods on the SAME
lens systems (inner join on lens_index + image):

  GP cross-correlation  scripts/gp_benchmark.ecsv             (wall_time_s)
  SALT fast             scripts/delay_benchmark_fast.ecsv     (fit_time_s)
  SALT robust           scripts/delay_benchmark_robust.ecsv   (fit_time_s)

All three used identical simulations (same per-lens seed), so differences
are attributable to the method alone. "GP (good)" is the GP restricted to
its own quality == "good" flag — the subset the pipeline would trust for
priming Stage 3.

Outputs: figures/benchmark_comparison.png + a stats table on stdout.

(The previous version of this script re-ran measure_one() variants on a
lens subset; it is preserved in git history if needed.)

Usage: python scripts/compare_fit_modes.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Fixed color per method (validated categorical palette; never reassigned)
COLORS = {"GP": "#2a78d6", "fast": "#eda100", "robust": "#1baf7a"}
TEXT, MUTED = "#1a1a19", "#6b6a60"


def load():
    """Inner-join the three benchmarks on (lens_index, image)."""
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from roman_td.paths import find_benchmark
    gp = Table.read(find_benchmark("gp_only_benchmark.ecsv"))
    fast = Table.read(find_benchmark("delay_benchmark_fast.ecsv"))
    rob = Table.read(find_benchmark("delay_benchmark_robust.ecsv"))

    def key(t):
        return {(int(r["lens_index"]), str(r["image"])): r for r in t}

    kg, kf, kr = key(gp), key(fast), key(rob)
    common = sorted(set(kg) & set(kf) & set(kr))

    rows = []
    for k in common:
        g, f, r = kg[k], kf[k], kr[k]
        rows.append({
            "lens_index": k[0], "image": k[1],
            "true_delay": float(g["true_delay"]),
            "res_GP": float(g["residual"]), "res_fast": float(f["residual"]),
            "res_robust": float(r["residual"]),
            "t_GP": float(g["wall_time_s"]), "t_fast": float(f["fit_time_s"]),
            "t_robust": float(r["fit_time_s"]),
            "gp_quality": str(g["quality"]),
        })
    return Table(rows)


def stats(res, t):
    res = res[np.isfinite(res)]
    return {
        "n": len(res),
        "median": np.median(res),
        "p68": np.percentile(np.abs(res), 68),
        "lt2": np.mean(np.abs(res) < 2) * 100,
        "lt5": np.mean(np.abs(res) < 5) * 100,
        "t_med": np.median(t[np.isfinite(t)]),
    }


def main():
    tab = load()
    print(f"Common systems across all three benchmarks: {len(tab)}")

    methods = ["GP", "fast", "robust"]
    S = {m: stats(np.asarray(tab[f"res_{m}"]), np.asarray(tab[f"t_{m}"]))
         for m in methods}
    good = tab[tab["gp_quality"] == "good"]
    S["GP (good)"] = stats(np.asarray(good["res_GP"]), np.asarray(good["t_GP"]))

    print(f"\n{'method':<10} {'n':>4} {'median':>8} {'P68|res|':>9} "
          f"{'<2d':>6} {'<5d':>6} {'med time':>9}")
    for m, s in S.items():
        print(f"{m:<10} {s['n']:>4} {s['median']:>+7.2f}d {s['p68']:>8.2f}d "
              f"{s['lt2']:>5.0f}% {s['lt5']:>5.0f}% {s['t_med']:>8.1f}s")

    # ── figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), facecolor="white")
    fig.subplots_adjust(hspace=0.42, wspace=0.30, top=0.90, bottom=0.08,
                        left=0.08, right=0.97)
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(True, color="#e8e7e0", lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # (a) ECDF of |residual| — accuracy at every threshold at once
    ax = axes[0, 0]
    for m in methods:
        r = np.sort(np.abs(np.asarray(tab[f"res_{m}"])))
        r = r[np.isfinite(r)]
        ax.step(np.clip(r, 1e-2, None), np.arange(1, len(r) + 1) / len(r),
                color=COLORS[m], lw=2, label=m, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 300)
    ax.set_ylim(0, 1.02)
    ax.axvline(2, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(2, 1.01, " 2 d", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("|residual|  (days)", color=TEXT)
    ax.set_ylabel("fraction of systems below", color=TEXT)
    ax.set_title("Accuracy: |residual| distribution (ECDF)",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    # (b) residual vs true delay — where each method breaks down
    ax = axes[0, 1]
    for m in methods:
        ax.scatter(tab["true_delay"], tab[f"res_{m}"], s=14,
                   color=COLORS[m], alpha=0.55, label=m, zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("true delay  (days)", color=TEXT)
    ax.set_ylabel("residual  (days)", color=TEXT)
    ax.set_title("Residual vs true delay  (clipped to ±25 d)",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # (c) time per system — log-spaced histograms
    ax = axes[1, 0]
    all_t = np.concatenate([np.asarray(tab[f"t_{m}"]) for m in methods])
    all_t = all_t[np.isfinite(all_t) & (all_t > 0)]
    bins = np.geomspace(all_t.min() * 0.8, all_t.max() * 1.2, 30)
    for m in methods:
        t = np.asarray(tab[f"t_{m}"])
        ax.hist(t[np.isfinite(t)], bins=bins, histtype="step", lw=2,
                color=COLORS[m], label=m, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("time per system  (s)", color=TEXT)
    ax.set_ylabel("systems", color=TEXT)
    ax.set_title("Speed: per-system measurement time",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # (d) the trade-off: accuracy vs speed, one point per method
    ax = axes[1, 1]
    for m, s in S.items():
        base = m.split(" ")[0]
        filled = "(" not in m
        ax.scatter(s["t_med"], s["p68"], s=110, zorder=3,
                   color=COLORS[base] if filled else "white",
                   edgecolors=COLORS[base], linewidths=2)
        ax.annotate(f"{m}\n{s['lt2']:.0f}% < 2 d", (s["t_med"], s["p68"]),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=9, color=TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median time per system  (s)", color=TEXT)
    ax.set_ylabel("P68 |residual|  (days)", color=TEXT)
    ax.set_title("The trade-off: accuracy vs cost",
                 color=TEXT, fontsize=11, loc="left")
    ax.margins(x=0.25, y=0.25)

    fig.suptitle(f"Time-delay methods on {len(tab)} common systems "
                 f"(identical simulations)", color=TEXT, fontsize=13, x=0.08,
                 ha="left", y=0.96)
    out = os.path.join(FIGDIR, "benchmark_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
