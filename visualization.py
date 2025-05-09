#!/usr/bin/env python3
"""
Multi-panel visualization across environments with shaded 95% CI (mean ± 1.96*SEM),
TD3 line at alpha = 1.0 and SAC autotune line. Loops automatically over all metrics. Shared legend.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

# metric_key -> (display name, csv column name)
METRICS = {
    "mean_reward":   ("Mean Reward",           "episode_reward"),
    "mean_KL":       ("Mean KL Divergence",    "episode_kl"),
    "q_infnorm":     ("Q Difference",          "episode_q_infnorm"),
    "jacobian_diff": ("Jacobian Difference",   "episode_jacobian_diff"),
}

# column lookup from metric_key
METRIC_COLUMN_MAPPING = {k: v[1] for k, v in METRICS.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--file_pattern", type=str, default="pairwise_per_episode.csv")
    return parser.parse_args()


def abbreviate_algorithm(alg_full):
    base = os.path.basename(alg_full)
    if "td3_continuous_action" in base:
        return "td3"
    if "sac_continuous_action" in base or "autotune" in base:
        return "sac"
    return os.path.splitext(base)[0].split('_')[0]


def load_and_aggregate(df, metric_key):
    metric = metric_key
    df["seed"] = df.get("actor_i_seed", np.nan)
    df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)
    df["alpha"] = pd.to_numeric(df.get("actor_i_alpha", np.nan), errors="coerce")
    df.loc[df["algorithm_abbr"] == "td3", "alpha"] = 1.0

    # include only sac and td3
    allowed = {"sac", "td3"}
    df = df[df["algorithm_abbr"].isin(allowed)]
    df = df.dropna(subset=[metric, "alpha", "seed"])
    if df.empty:
        return None

    per_seed = (
        df.groupby(["env", "algorithm_abbr", "alpha", "seed"])[metric]
          .mean().reset_index()
    )
    summary = (
        per_seed.groupby(["env", "algorithm_abbr", "alpha"])[metric]
               .agg(mean="mean", std="std", count="count")
               .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    return summary


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_pattern = args.file_pattern

    files = [f for f in os.listdir(input_dir)
             if file_pattern in f and f.endswith(".csv")]
    data_list = []
    env_order = []
    for f in files:
        df = pd.read_csv(os.path.join(input_dir, f))
        if "env" not in df.columns or df["env"].isna().all():
            continue
        env_name = df["env"].dropna().iloc[0]
        env_order.append(env_name)
        data_list.append((env_name, df))

    if not data_list:
        print("No valid data loaded.")
        return

    unique_envs = list(dict.fromkeys(env_order))
    num_envs = len(unique_envs)
    algorithm_colors = {"sac": "orange", "td3": "red"}

    for metric_key, (display_name, _) in METRICS.items():
        ylabel = f"{display_name} (log scale)"
        summaries = []
        for env_name, df in data_list:
            col = METRIC_COLUMN_MAPPING[metric_key]
            if col in df.columns and col != metric_key:
                df = df.rename(columns={col: metric_key})
            sm = load_and_aggregate(df.copy(), metric_key)
            if sm is not None:
                summaries.append(sm)

        if not summaries:
            print(f"No data for {metric_key}")
            continue

        full_df = pd.concat(summaries, ignore_index=True)

        fig, axes = plt.subplots(1, num_envs, figsize=(6 * num_envs, 6), sharey=False)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.25, wspace=0.3)
        fig.text(0.045, 0.5, ylabel, va='center', rotation='vertical', fontsize=26)

        legend_handles = {}
        for ax, env in zip(axes, unique_envs):
            df_env = full_df[full_df["env"] == env]
            ax.set_title(env, fontsize=24)
            ax.set_xlabel("Alpha", fontsize=20)
            ax.set_yscale("log")
            ax.grid(True)

            for alg in df_env["algorithm_abbr"].unique():
                df_alg = df_env[df_env["algorithm_abbr"] == alg]
                df_main = df_alg[~np.isclose(df_alg["alpha"], 1.0, atol=1e-6)]
                df_one = df_alg[np.isclose(df_alg["alpha"], 1.0, atol=1e-6)]
                color = algorithm_colors.get(alg, "gray")

                if not df_main.empty:
                    x = df_main["alpha"]
                    y = df_main["mean"]
                    ci = df_main["ci95"].fillna(0)
                    line, = ax.plot(x, y, "-o", color=color, label=alg)
                    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.25)
                    legend_handles[alg] = line
                if not df_one.empty:
                    yval = df_one["mean"].iloc[0]
                    label = "Autotune SAC" if alg == "sac" else "TD3"
                    line = ax.axhline(y=yval, linestyle="--", color=color, linewidth=2.5, label=label)
                    legend_handles[label] = line

        fig.legend(
            [legend_handles[k] for k in sorted(legend_handles)],
            sorted(legend_handles),
            loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_handles), frameon=True
        )

        os.makedirs(output_dir, exist_ok=True)
        fname = f"multi_env_alpha_vs_{metric_key}.png"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {os.path.join(output_dir, fname)}")


if __name__ == "__main__":
    main()
