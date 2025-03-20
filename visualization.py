#!/usr/bin/env python3
"""
Visualizes the evaluation results by plotting a given metric versus alpha,
with separate lines for each algorithm. For every CSV in the results directory,
a separate subplot is created side by side in a single row.

Modifications:
- No shared y-axis (each subplot is independent).
- Only the leftmost subplot displays a y-axis label.
- One overarching legend is placed above all subplots, centered,
  with a box outline around it.
- The y-axis for all plots is set to a log scale.
- In each subplot, if there are TD3 rows (algorithm contains
  "td3_continuous_action.py"), a horizontal dashed line is drawn
  at the average metric value (averaging across all seeds).

Columns expected in each CSV:
    actor_0_alpha, actor_0_seed, actor_1_alpha, actor_1_seed, algorithm,
    n_eval_episodes, mean_reward, mean_KL, q_infnorm, jacobian_diff, env

We unify "q_output_diff" to "q_infnorm" if the former exists.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results from multiple CSV files"
    )
    parser.add_argument(
        "--input_dir", type=str, default="results",
        help="Directory containing input CSV files (default: results)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations",
        help="Directory where the output plot will be saved (default: visualizations)"
    )
    return parser.parse_args()

def abbreviate_algorithm(alg_full):
    """
    Given a full algorithm path, e.g.
    "code/cleanrl/cleanrl/sac_continuous_action.py",
    returns the abbreviated algorithm name, e.g. "sac".
    """
    base = os.path.basename(alg_full)              # e.g. "sac_continuous_action.py"
    name_no_ext = os.path.splitext(base)[0]        # e.g. "sac_continuous_action"
    return name_no_ext.split('_')[0]               # e.g. "sac"

def plot_metric(metric, ylabel, output_file, input_dir):
    # List all CSV files in the input directory.
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    num_files = len(csv_files)
    # Create subplots in a single row without sharing the y-axis.
    fig, axes = plt.subplots(
        nrows=1, ncols=num_files, figsize=(11.69, 8.27 / 3)
    )
    if num_files == 1:
        axes = [axes]

    # Dictionary to collect legend handles (lines) keyed by the abbreviated algorithm name.
    all_handles = {}

    # Iterate over each CSV file to create a subplot.
    for i, csv_file in enumerate(csv_files):
        ax = axes[i]
        df = pd.read_csv(csv_file)

        # --- Unify q_output_diff -> q_infnorm if needed ---
        if "q_output_diff" in df.columns and "q_infnorm" not in df.columns:
            df.rename(columns={"q_output_diff": "q_infnorm"}, inplace=True)
        # --------------------------------------------------

        # Convert the actor_0_alpha column to numeric if it's not already.
        if "actor_0_alpha" in df.columns:
            df["alpha"] = pd.to_numeric(df["actor_0_alpha"], errors="coerce")
        else:
            # fallback in case CSV doesn't have "actor_0_alpha"
            df["alpha"] = np.nan

        # Extract the environment name for the title (if present).
        if "env" in df.columns and not df["env"].empty:
            env_val = df["env"].iloc[0]
        else:
            env_val = os.path.basename(csv_file)

        # Plot each algorithm on this subplot.
        algorithms = df["algorithm"].unique()
        for alg in algorithms:
            df_alg = df[df["algorithm"] == alg]
            if df_alg.empty:
                continue

            alg_abbr = abbreviate_algorithm(alg)
            grouped = df_alg.groupby("alpha")[metric]
            mean_val = grouped.mean()
            if mean_val.empty:
                continue
            std_val = grouped.std()
            n = grouped.count()
            ci = 1.96 * std_val / np.sqrt(n)

            # Plot with error bars.
            err_container = ax.errorbar(
                mean_val.index, mean_val.values,
                yerr=ci,
                fmt="o-",
                capsize=5
            )
            if err_container.lines:
                main_line = err_container.lines[0]
                main_line.set_label(alg_abbr)
                if alg_abbr not in all_handles:
                    all_handles[alg_abbr] = main_line

        # --- TD3 Horizontal Line Overlay ---
        # Filter for rows where algorithm contains "td3_continuous_action.py"
        td3_mask = df["algorithm"].str.contains("td3_continuous_action.py", na=False)
        if td3_mask.sum() > 0:
            # Compute the average metric value across all TD3 rows.
            td3_avg = df[td3_mask][metric].mean()
            td3_line = ax.axhline(
                y=td3_avg, linestyle="--", color="black", linewidth=2, label="TD3 avg"
            )
            if "TD3 avg" not in all_handles:
                all_handles["TD3 avg"] = td3_line

        # Only the leftmost subplot gets the y-axis label.
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Alpha")
        ax.set_title(env_val)
        ax.grid(True)
        # Set y-axis to log scale.
        ax.set_yscale("log")

    # Create one overarching legend, centered above all subplots with a box outline.
    sorted_labels = sorted(all_handles.keys())
    sorted_handles = [all_handles[label] for label in sorted_labels]
    legend = fig.legend(
        sorted_handles,
        sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(sorted_labels),
        frameon=True
    )
    legend.get_frame().set_edgecolor('black')

    plt.subplots_adjust(top=0.85)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # First plot: Mean KL Divergence.
    output_file_kl = os.path.join(output_dir, "alpha_vs_mean_KL_all.png")
    plot_metric("mean_KL", "Mean KL Divergence", output_file_kl, input_dir)

    # Second plot: Q-network difference (unified column: q_infnorm).
    output_file_q = os.path.join(output_dir, "alpha_vs_q_infnorm_all.png")
    plot_metric("q_infnorm", "Q Difference", output_file_q, input_dir)

    # Third plot: Jacobian difference.
    output_file_jacobian = os.path.join(output_dir, "alpha_vs_jacobian_diff_all.png")
    plot_metric("jacobian_diff", "Jacobian Diff", output_file_jacobian, input_dir)

if __name__ == "__main__":
    main()
