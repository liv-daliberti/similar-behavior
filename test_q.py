#!/usr/bin/env python3
"""
Multi-panel visualization for Humanoid-v4 using only the merged CSV.
Panels: Mean Reward, Mean KL, Average Q-Mean Difference, Jacobian Difference,
         CI of Run CI Sizes (95%).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ↑3× larger fonts
plt.rcParams.update({
    "font.family":      "serif",
    "axes.titlesize":   72,
    "axes.labelsize":   66,
    "xtick.labelsize":  54,
    "ytick.labelsize":  54,
    "legend.fontsize":  60,
    "lines.linewidth":  3.0,
    "lines.markersize": 8,
})

METRICS = {
    "mean_reward":   ("Mean Reward",               "episode_reward"),
    "mean_KL":       ("Mean KL Divergence",        "episode_kl"),
    "q_infnorm":     ("Average Q-Mean Difference", "episode_q_meandiff"),
    "jacobian_diff": ("Jacobian Difference",       "episode_jacobian_diff"),
    "ci_of_ci":      ("CI of Run CI Sizes (95%)",  "episode_reward"),
}
METRIC_COLUMN = {k: v[1] for k, v in METRICS.items()}
CI_Z = 1.96
SUPPORTED_ALGS = {"sac", "td3", "meow"}
ALG_COLORS    = {"sac": "orange", "td3": "red",   "meow": "blue"}

def abbreviate_algorithm(path: str) -> str:
    b = os.path.basename(path)
    if "td3_continuous_action" in b:    return "td3"
    if "sac_continuous_action" in b \
    or "autotune" in b:                  return "sac"
    if "meow_continuous_action" in b:    return "meow"
    return os.path.splitext(b)[0].split('_')[0]

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-panel viz for Humanoid-v4 (merged CSV)"
    )
    p.add_argument(
        "--input_csv",
        type=str,
        default="results/Humanoid-v4-merged.csv",
        help="Merged per-episode CSV with episode_q_meandiff"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Where to save the figure"
    )
    return p.parse_args()

def load_and_aggregate(df: pd.DataFrame, metric_key: str):
    # 1) algorithm, alpha, seed
    if "algorithm_abbr" not in df:
        df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)
    if "alpha" not in df and "actor_i_alpha" in df:
        df["alpha"] = pd.to_numeric(df["actor_i_alpha"], errors="coerce")
    if "seed" not in df and "actor_i_seed" in df:
        df["seed"] = df["actor_i_seed"].astype(int)
    # force TD3 baseline at α=1.0
    df.loc[df["algorithm_abbr"] == "td3", "alpha"] = 1.0

    # 2) rename the raw column to metric_key
    raw = METRIC_COLUMN[metric_key]
    if metric_key != "ci_of_ci" and raw in df:
        df = df.rename(columns={raw: metric_key})

    # 3) drop missing
    keep = [metric_key, "alpha", "seed", "algorithm_abbr"]
    df = df.dropna(subset=keep)

    # 4) per-seed means
    if metric_key == "ci_of_ci":
        # CI of run CI sizes
        per = (
            df.groupby(["alpha","seed"])[metric_key]
              .agg(mean="mean", std="std", count="count")
              .reset_index()
        )
    else:
        per = (
            df.groupby(["algorithm_abbr","alpha","seed"])[metric_key]
              .mean()
              .reset_index(name="mean")
        )
    # 5) collapse across seeds
    if metric_key == "ci_of_ci":
        summary = per.groupby("alpha")["mean"] \
                     .agg(mean="mean", std="std", count="count") \
                     .reset_index()
    else:
        summary = (
            per
            .groupby(["algorithm_abbr","alpha"])["mean"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"]  = CI_Z * summary["sem"]
    return summary

def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    nm         = len(METRICS)
    fig_height = 23
    fig_width  = 18 * nm * 1.25
    fig, axes  = plt.subplots(1, nm, figsize=(fig_width, fig_height), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.22, wspace=0.22)

    handles = {}
    for i, (mk, (title, _)) in enumerate(METRICS.items()):
        ax      = axes[i]
        summary = load_and_aggregate(df.copy(), mk)
        if summary is None or summary.empty:
            ax.set_title(f"{title}\n(no data)", fontsize=60)
            continue

        ax.set_title(title, fontsize=60)
        ax.set_xlabel("Alpha", fontsize=54)
        ax.set_xticks(np.arange(0, 0.8, 0.1))
        ax.minorticks_on()
        ax.set_ylabel(f"{title} (log scale)", fontsize=54)
        ax.set_yscale("log")
        ax.grid(True, which="major")
        ax.grid(True, which="minor", linestyle=":", linewidth=1.5)

        # --- special Q‐panel logic: three series + clamp ---
        if mk == "q_infnorm":
            sac_main = summary[
                (summary["algorithm_abbr"] == "sac")
                & (summary["alpha"] != 1.0)
            ]
            sac_base = summary[
                (summary["algorithm_abbr"] == "sac")
                & (summary["alpha"] == 1.0)
            ]
            td3_base = summary[summary["algorithm_abbr"] == "td3"]

            # SAC curve
            x = sac_main["alpha"]; y = sac_main["mean"]; ci = sac_main["ci"]
            ln, = ax.plot(x, y, "-o", color="orange", label="SAC")
            ax.fill_between(x, y-ci, y+ci, color="orange", alpha=0.25)
            handles["SAC"] = ln

            # SAC α=1 baseline
            if not sac_base.empty:
                y0 = sac_base["mean"].iloc[0]
                hl = ax.axhline(
                    y0,
                    linestyle="--",
                    color="orange",
                    linewidth=3,
                    label="SAC α=1 Baseline"
                )
                handles["SAC α=1 Baseline"] = hl

            # TD3 baseline
            if not td3_base.empty:
                y1 = td3_base["mean"].iloc[0]
                hl = ax.axhline(
                    y1,
                    linestyle="--",
                    color="red",
                    linewidth=3,
                    label="TD3 Baseline"
                )
                handles["TD3 Baseline"] = hl

            # clamp top to max SAC point
            ax.set_ylim(top=sac_main["mean"].max())
            ax.set_xticks(
                np.arange(0, sac_main["alpha"].max() + 1e-6, 0.1)
            )
            ax.legend(loc="upper right")
            continue

        # --- generic panels for the other metrics ---
        for alg in SUPPORTED_ALGS:
            part = summary[summary["algorithm_abbr"] == alg]
            if part.empty: continue
            color = ALG_COLORS[alg]
            main  = ~np.isclose(part["alpha"], 1.0, atol=1e-6)
            ref   =  np.isclose(part["alpha"], 1.0, atol=1e-6)

            if main.any():
                x = part.loc[main, "alpha"]
                y = part.loc[main, "mean"]
                ln, = ax.plot(x, y, "-o", color=color, label=alg.upper())
                if mk != "ci_of_ci":
                    ci = part.loc[main, "ci"].fillna(0)
                    ax.fill_between(x, y-ci, y+ci, color=color, alpha=0.25)
                handles[alg.upper()] = ln

            if ref.any():
                y0 = (part.loc[ref, "mean"].iloc[0]
                      if mk != "ci_of_ci"
                      else part.loc[ref, "ci"].iloc[0])
                label = ("SAC α=1 Baseline" if alg=="sac"
                         else "TD3 Baseline"   if alg=="td3"
                         else None)
                if label:
                    hl = ax.axhline(
                        y0, linestyle="--", color=color,
                        linewidth=5, label=label
                    )
                    handles[label] = hl

    # single legend across all panels
    fig.legend(
        handles=list(handles.values()),
        labels=list(handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        bbox_transform=fig.transFigure,
        ncol=len(handles),
        frameon=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "Humanoid-v4_alpha_vs_all_metrics.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.5, dpi=300)
    plt.close(fig)
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
