#!/usr/bin/env python3
"""
Multi-panel visualization for each environment: one panel per metric
(+ Between-Run CI of Within-Run CI Size), showing how per-seed confidence
interval widths vary with α and how they themselves vary across seeds.
Y-axes are on a log scale (indicated in each label).
Figures are generated at high resolution (18″ tall × 18″×n_metrics×1.25″ wide).
All font sizes have been increased ~3× for readability.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Increase all font sizes by roughly 3×
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 72,
    "axes.labelsize": 66,
    "xtick.labelsize": 54,
    "ytick.labelsize": 54,
    "legend.fontsize": 60,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

METRICS = {
    "mean_reward":   ("Mean Reward",              "episode_reward"),
    "mean_KL":       ("Mean KL Divergence",       "episode_kl"),
    "q_infnorm":     ("Q Mean Difference",        "episode_q_meandiff"),
    "jacobian_diff": ("Jacobian Difference",      "episode_jacobian_diff"),
    "ci_of_ci":      ("CI of Run CI Sizes (95%)", "episode_reward"),
}
METRIC_COLUMN = {k: v[1] for k, v in METRICS.items()}
CI_Z = 1.96
SUPPORTED_ALGS = {"sac", "td3", "meow"}
ALG_COLORS = {"sac": "orange", "td3": "red", "meow": "blue"}

OVERRIDE_METRIC_FILES = {
    "q_infnorm": "/scratch/gpfs/od2961/similarity-paper/similar-behavior/results/Humanoid-v4-pairwise_q_mean_diff.csv"
}
OVERRIDE_ENV = "Humanoid-v4"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    type=str, default="results")
    p.add_argument("--output_dir",   type=str, default="visualizations")
    p.add_argument("--file_pattern", type=str, default="pairwise_per_episode.csv")
    return p.parse_args()

def abbreviate_algorithm(path: str) -> str:
    b = os.path.basename(path)
    if "td3_continuous_action" in b: return "td3"
    if "sac_continuous_action" in b or "autotune" in b: return "sac"
    if "meow_continuous_action" in b: return "meow"
    return os.path.splitext(b)[0].split('_')[0]

def load_and_aggregate(df, metric_key):
    if metric_key in OVERRIDE_METRIC_FILES:
        try:
            df = pd.read_csv(OVERRIDE_METRIC_FILES[metric_key])
            df["env"] = OVERRIDE_ENV
            df["algorithm_abbr"] = "td3"
            df["alpha"] = 0.0  # Q mean diff is evaluated at alpha = 0
            import re
            def extract_seed(path):
                match = re.search(r'__([0-9]+)__[0-9]+_step', str(path))
                return int(match.group(1)) if match else np.nan
            df["seed"] = df["checkpoint_i"].apply(extract_seed)
        except Exception as e:
            print(f"Failed to load override file for {metric_key}: {e}")
            return None
    else:
        if "algorithm_abbr" not in df.columns and "algorithm" in df.columns:
            df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)
        if "alpha" not in df.columns and "actor_i_alpha" in df.columns:
            df["alpha"] = pd.to_numeric(df["actor_i_alpha"], errors="coerce")
        if "seed" not in df.columns and "actor_i_seed" in df.columns:
            df["seed"] = df["actor_i_seed"]
        df.loc[df["algorithm_abbr"] == "td3", "alpha"] = 1.0

    raw_col = METRIC_COLUMN[metric_key]
    if metric_key != "ci_of_ci" and raw_col in df.columns:
        df = df.rename(columns={raw_col: metric_key})

    df["seed"] = df.get("seed", np.nan)
    df = df[df["algorithm_abbr"].isin(SUPPORTED_ALGS)]

    candidates = []
    if metric_key in df.columns: candidates.append(metric_key)
    if raw_col in df.columns and metric_key == "ci_of_ci": candidates.append(raw_col)
    if not candidates: return None
    drop_col = candidates[0]
    df = df.dropna(subset=[drop_col, "alpha", "seed"])
    if df.empty: return None

    if metric_key == "ci_of_ci":
        per_seed = (
            df.groupby(["env", "algorithm_abbr", "alpha", "seed"])
              .episode_reward.agg(std="std", count="count").reset_index()
        )
        per_seed["sem_run"] = per_seed["std"] / np.sqrt(per_seed["count"])
        per_seed["ci_run"]  = CI_Z * per_seed["sem_run"]
        summary = (
            per_seed.groupby(["env", "algorithm_abbr", "alpha"])["ci_run"]
              .agg(mean="mean", std="std", count="count").reset_index()
        )
    else:
        per_seed = (
            df.groupby(["env", "algorithm_abbr", "alpha", "seed"])[metric_key]
              .mean().reset_index(name="metric_mean")
        )
        summary = (
            per_seed.groupby(["env", "algorithm_abbr", "alpha"])["metric_mean"]
              .agg(mean="mean", std="std", count="count").reset_index()
        )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"]  = CI_Z * summary["sem"]
    return summary

def main():
    args = parse_args()
    files = [f for f in os.listdir(args.input_dir)
             if f.endswith(".csv") and args.file_pattern in f]
    data = []
    for f in files:
        df = pd.read_csv(os.path.join(args.input_dir, f))
        if "env" not in df or df["env"].dropna().empty: continue
        data.append((df["env"].dropna().iloc[0], df))
    if not data:
        print("No valid data found."); return

    nm = len(METRICS)
    fig_height = 23
    fig_width  = 18 * nm * 1.25

    for env, df in data:
        fig, axes = plt.subplots(1, nm, figsize=(fig_width, fig_height), sharey=False)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.22, wspace=0.22)
        fig.suptitle(f"Environment: {env}", fontsize=84, y=0.96)

        handles = {}
        for i, (mk, (title, _)) in enumerate(METRICS.items()):
            ax = axes[i]
            summary = load_and_aggregate(df.copy(), mk)
            if summary is None or summary.empty:
                ax.set_title(f"{title}\n(no data)", fontsize=60)
                continue
            sub = summary[summary["env"] == env]

            ax.set_title(title, fontsize=60)
            ax.set_xlabel("Alpha", fontsize=54)
            ax.set_xticks(np.arange(0, 0.8, 0.1))
            ax.minorticks_on()
            ax.set_ylabel(f"{title} (log scale)", fontsize=54)
            ax.set_yscale("log")
            ax.grid(True, which="major")
            ax.grid(True, which="minor", linestyle=":", linewidth=1.5)

            for alg in SUPPORTED_ALGS:
                part = sub[sub["algorithm_abbr"] == alg]
                if part.empty: continue
                color = ALG_COLORS[alg]
                main = ~np.isclose(part["alpha"], 1.0, atol=1e-6)
                ref  =  np.isclose(part["alpha"], 1.0, atol=1e-6)

                if main.any():
                    x = part.loc[main, "alpha"]
                    y = part.loc[main, "ci"] if mk == "ci_of_ci" else part.loc[main, "mean"]
                    ln, = ax.plot(x, y, "-o", color=color, label=alg.upper())
                    if mk != "ci_of_ci":
                        ci = part.loc[main, "ci"].fillna(0)
                        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.25)
                    handles[alg.upper()] = ln

                if ref.any():
                    y0 = part.loc[ref, "ci"].iloc[0] if mk == "ci_of_ci" else part.loc[ref, "mean"].iloc[0]
                    label = "SAC α=1 Baseline" if alg == "sac" else "TD3 Baseline" if alg == "td3" else None
                    if label:
                        hl = ax.axhline(y0, linestyle="--", color=color, linewidth=5, label=label)
                        handles[label] = hl

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
        out = os.path.join(args.output_dir, f"{env}_alpha_vs_all_metrics.png")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.5, dpi=300)
        plt.close(fig)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
