#!/usr/bin/env python3
"""
Multi-panel visualization over every env and for two groups:
  1) MEOW vs TD3
  2) SAC vs TD3 vs Adaptive

One panel per metric (Reward, KL, Q-infnorm, Jacobian, %Uncert).
Input files are named like HalfCheetah-v4-pairwise_per_episode.csv, 
HalfCheetah-v4-1-pairwise_per_episode.csv, etc.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict

plt.rcParams.update({
    "font.family":      "serif",
    "axes.titlesize":   24,
    "axes.labelsize":   22,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  18,
    "lines.linewidth":  3,
    "lines.markersize": 8,
})

COLOR_MAP = {
    "td3": "#1f77b4",        # blue
    "sac": "#2ca02c",        # green
    "adaptive": "#ff7f0e",   # orange
    "meow": "#d62728",       # red
}

LABEL_MAP = {
    "td3": "TD3 Baseline",
    "sac": "Fixed Entropy SAC",
    "adaptive": "Adaptive α (Ours)",
    "meow": "MEOW"
}

METRICS = {
    "mean_reward":    ("Mean Reward",          "episode_reward"),
    "mean_KL":        ("Mean KL Divergence",   "episode_kl"),
    "q_infnorm":      ("Q-Mean Difference", "episode_q_infnorm"),
    "jacobian_diff":  ("Jacobian Difference",  "episode_jacobian_diff"),
    "percent_uncert": ("Percent Uncertainty",  None),
}
CI_Z = 1.96

def abbreviate_algorithm(path: str) -> str:
    b = os.path.basename(path)
    if "meow_continuous_action" in b:      return "meow"
    if "td3_continuous_action" in b:       return "td3"
    if "sac_continuous_action" in b or "autotune" in b: return "sac"
    if "adaptive_sac" in b:                return "adaptive"
    return "other"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    type=str, default="results")
    p.add_argument("--file_pattern", type=str, default="pairwise_per_episode.csv")
    p.add_argument("--output_dir",   type=str, default="visualizations")
    return p.parse_args()

def load_and_aggregate(df: pd.DataFrame, metric_key: str):
    if "algorithm_abbr" not in df:
        df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)
    if "alpha" not in df and "actor_i_alpha" in df:
        df["alpha"] = pd.to_numeric(df["actor_i_alpha"], errors="coerce").fillna(1.0)
    if "seed" not in df and "actor_i_seed" in df:
        df["seed"] = df["actor_i_seed"].astype(int)
    df.loc[df["algorithm_abbr"]=="td3", "alpha"] = 1.0

    if metric_key == "percent_uncert":
        per = (
            df.groupby(["algorithm_abbr","alpha","seed"])["episode_reward"]
              .agg(std="std", count="count")
              .reset_index()
        )
        per["ci_run"] = CI_Z * per["std"] / np.sqrt(per["count"])
        reward_per = (
            df.groupby(["algorithm_abbr","alpha","seed"])["episode_reward"]
              .mean().reset_index(name="mean_reward")
        )
        merged = per.merge(reward_per, on=["algorithm_abbr","alpha","seed"])
        pct = (
            merged.groupby(["algorithm_abbr","alpha"])
                  .apply(lambda g: (g["ci_run"]/g["mean_reward"]).mean()*100)
                  .reset_index(name="percent_uncert")
        )
        return pct[["algorithm_abbr","alpha","percent_uncert"]]

    _, raw_col = METRICS[metric_key]
    df = df.rename(columns={raw_col: metric_key})
    df = df.dropna(subset=[metric_key,"alpha","seed","algorithm_abbr"])
    per = (
        df.groupby(["algorithm_abbr","alpha","seed"])[metric_key]
          .mean().reset_index(name="mean")
    )
    summary = (
        per.groupby(["algorithm_abbr","alpha"])["mean"]
           .agg(mean="mean", std="std", count="count")
           .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"]  = CI_Z * summary["sem"]
    return summary

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Group all files by env prefix (strip off any -1, -2, etc.)
    file_groups = defaultdict(list)
    pattern = re.compile(rf"^(.*?)-pairwise_per_episode(?:-\d+)?\.csv$")

    for fname in os.listdir(args.input_dir):
        match = pattern.match(fname)
        if match:
            env_key = match.group(1)
            file_groups[env_key].append(os.path.join(args.input_dir, fname))
        
    if not file_groups:
        print(f"No '{args.file_pattern}' files found in {args.input_dir}")
        return

    nm = len(METRICS)
    groups = [
        ("MEOW_vs_TD3",      ["meow","td3"]),
        ("SAC_TD3_Adaptive", ["sac","td3","adaptive"]),
    ]

    for env, file_list in sorted(file_groups.items()):
        print(f"\n→ Environment: {env} (from {len(file_list)} files)")
        df_parts = [pd.read_csv(f) for f in file_list]
        df = pd.concat(df_parts, ignore_index=True)

        df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)
        df["alpha"] = pd.to_numeric(df["actor_i_alpha"], errors="coerce").fillna(1.0)
        df["seed"]  = df["actor_i_seed"].astype(int)
        df.loc[df["algorithm_abbr"]=="td3", "alpha"] = 1.0

        for group_name, algs in groups:
            dfg = df[df["algorithm_abbr"].isin(algs)].copy()
            dfg = dfg[~((dfg["algorithm_abbr"]=="td3") & (dfg["episode_reward"] < 50))]
            dfg = dfg[~((dfg["algorithm_abbr"]=="meow") & (dfg["alpha"]==0.0))]

            if dfg.empty:
                print(f"  (No data for group {group_name} after filtering)")
                continue

            fig, axes = plt.subplots(1, nm, figsize=(6 * nm, 6), sharey=False)
            fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.18, hspace=0.35)
            fig.suptitle(f"{env}", fontsize=28)
            legend_handles = {}

            for i, (mk, (title, _)) in enumerate(METRICS.items()):
                ax = axes[i]
                summ = load_and_aggregate(dfg.copy(), mk)
                ax.set_title(title)
                ax.set_xlabel("Alpha")
                ax.set_xlim(0, 0.7)
                ax.minorticks_off()
                ax.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.4)

                if summ is None or summ.empty:
                    ax.text(0.5, 0.5, "(no data)", ha='center', va='center')
                    continue

                ylabel = "% Uncertainty" if mk == "percent_uncert" else (f"{title} (log)" if mk != "mean_reward" else title)
                ax.set_ylabel(ylabel)
                if mk != "mean_reward" and mk != "percent_uncert":
                    ax.set_yscale("log")

                present = [a for a in algs if a in summ.algorithm_abbr.values]

                for alg in present:
                    part = summ[summ.algorithm_abbr == alg]
                    x = part["alpha"].to_numpy()
                    y = part["percent_uncert"].to_numpy() if mk == "percent_uncert" else part["mean"].to_numpy()
                    color = COLOR_MAP.get(alg, None)
                    label = LABEL_MAP.get(alg, alg)

                    x_1 = part[np.isclose(part["alpha"], 1.0)]
                    x_rest = part[~np.isclose(part["alpha"], 1.0)]

                    if alg == "adaptive":
                        yval = y[0] if len(y) > 0 else None
                        if yval is not None:
                            ln = ax.axhline(yval, linestyle='--', linewidth=2, label=label, color=color)
                            legend_handles[ln.get_label()] = ln
                        continue

                    if not x_1.empty:
                        yval = x_1["percent_uncert"].values[0] if mk == "percent_uncert" else x_1["mean"].values[0]
                        new_label = "Learned Entropy" if alg == "sac" else label + " (α=1)"
                        ln = ax.axhline(yval, linestyle='--', linewidth=2, label=new_label, color=color)
                        legend_handles[ln.get_label()] = ln
                    if not x_rest.empty:
                        x_vals = x_rest["alpha"].to_numpy()
                        y_vals = x_rest["percent_uncert"].to_numpy() if mk == "percent_uncert" else x_rest["mean"].to_numpy()
                        ci = x_rest["ci"].to_numpy() if "ci" in x_rest else None
                        ln, = ax.plot(x_vals, y_vals, '-o', label=label, color=color)
                        if ci is not None:
                            ax.fill_between(x_vals, y_vals - ci, y_vals + ci, alpha=0.25, color=color)
                        legend_handles[ln.get_label()] = ln

            fig.legend(
                list(legend_handles.values()),
                list(legend_handles.keys()),
                loc='lower center',
                bbox_to_anchor=(0.5, -0.08),
                ncol=len(legend_handles),
                frameon=True
            )

            out = os.path.join(args.output_dir, f"{env}_{group_name}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ Saved: {out}")

if __name__ == "__main__":
    main()
