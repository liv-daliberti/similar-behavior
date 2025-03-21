#!/usr/bin/env python3
"""
Visualizes the evaluation results by plotting metrics versus alpha,
with separate lines for each algorithm. For every CSV in the results directory,
a separate subplot is created side by side in a single row.

Modifications:
- No shared y-axis (each subplot is independent).
- Only the leftmost subplot displays a y-axis label.
- One overarching legend is placed above all subplots, centered, with a box outline.
- The y-axis for "mean_reward" is set to linear scale; other metrics use a log scale.
- Additionally, a comparison plot of "q_infnorm" and "mean_KL" is generated.
- In each subplot, if there are TD3 rows (algorithm contains "td3_continuous_action.py"),
  a horizontal dashed line is drawn at the average metric value (averaging across all seeds).

Expected CSV columns:
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
    Given a full algorithm path, e.g. "code/cleanrl/cleanrl/sac_continuous_action.py",
    returns the abbreviated algorithm name, e.g. "sac".
    """
    base = os.path.basename(alg_full)              # e.g. "sac_continuous_action.py"
    name_no_ext = os.path.splitext(base)[0]         # e.g. "sac_continuous_action"
    return name_no_ext.split('_')[0]                # e.g. "sac"

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
    # Create subplots in a single row.
    fig, axes = plt.subplots(
        nrows=1, ncols=num_files, figsize=(11.69, 8.27 / 3)
    )
    if num_files == 1:
        axes = [axes]

    all_handles = {}

    for i, csv_file in enumerate(csv_files):
        ax = axes[i]
        df = pd.read_csv(csv_file)

        # Unify q_output_diff -> q_infnorm if needed.
        if "q_output_diff" in df.columns and "q_infnorm" not in df.columns:
            df.rename(columns={"q_output_diff": "q_infnorm"}, inplace=True)

        # Create 'alpha' column.
        if "actor_0_alpha" in df.columns:
            df["alpha"] = pd.to_numeric(df["actor_0_alpha"], errors="coerce")
        else:
            df["alpha"] = np.nan

        # Extract environment name.
        env_val = df["env"].iloc[0] if "env" in df.columns and not df["env"].empty else os.path.basename(csv_file)

        algorithms = df["algorithm"].unique()
        for alg in algorithms:
            df_alg = df[df["algorithm"] == alg]
            if df_alg.empty:
                continue

            alg_abbr = abbreviate_algorithm(alg)
            grouped = df_alg.groupby("alpha")[metric]
            mean_val = grouped.mean()
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

        # TD3 horizontal line overlay.
        td3_mask = df["algorithm"].str.contains("td3_continuous_action.py", na=False)
        if td3_mask.sum() > 0:
            td3_avg = df[td3_mask][metric].mean()
            td3_line = ax.axhline(
                y=td3_avg, linestyle="--", color="black", linewidth=2, label="TD3 avg"
            )
            if "TD3 avg" not in all_handles:
                all_handles["TD3 avg"] = td3_line

        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Alpha")
        ax.set_title(env_val)
        ax.grid(True)
        # Use linear scale for mean_reward; log scale for others.
        if metric == "mean_reward":
            ax.set_yscale("linear")
        else:
            ax.set_yscale("log")

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
def plot_comparison(metric1, metric2, ylabel, output_file, input_dir):
    """
    Plots a side-by-side comparison of two metrics (e.g., q_infnorm and mean_KL)
    versus alpha on the same axes for each CSV file.
    """
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    num_files = len(csv_files)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_files, figsize=(11.69, 8.27 / 3)
    )
    if num_files == 1:
        axes = [axes]
    
    all_handles = {}

    for i, csv_file in enumerate(csv_files):
        ax = axes[i]
        df = pd.read_csv(csv_file)
        # Unify q_output_diff -> q_infnorm if needed.
        if "q_output_diff" in df.columns and "q_infnorm" not in df.columns:
            df.rename(columns={"q_output_diff": "q_infnorm"}, inplace=True)
        if "actor_0_alpha" in df.columns:
            df["alpha"] = pd.to_numeric(df["actor_0_alpha"], errors="coerce")
        else:
            df["alpha"] = np.nan
        env_val = df["env"].iloc[0] if "env" in df.columns and not df["env"].empty else os.path.basename(csv_file)
        
        algorithms = df["algorithm"].unique()
        for alg in algorithms:
            df_alg = df[df["algorithm"] == alg]
            if df_alg.empty:
                continue
            alg_abbr = abbreviate_algorithm(alg)
            # Group by alpha for both metrics.
            grp1 = df_alg.groupby("alpha")[metric1]
            grp2 = df_alg.groupby("alpha")[metric2]
            mean1 = grp1.mean()
            std1 = grp1.std()
            n1 = grp1.count()
            ci1 = 1.96 * std1 / np.sqrt(n1)
            mean2 = grp2.mean()
            std2 = grp2.std()
            n2 = grp2.count()
            ci2 = 1.96 * std2 / np.sqrt(n2)
            # Plot both metrics on the same axes.
            err1 = ax.errorbar(
                mean1.index, mean1.values,
                yerr=ci1, fmt="o-", capsize=5, label=f"{alg_abbr} {metric1}"
            )
            err2 = ax.errorbar(
                mean2.index, mean2.values,
                yerr=ci2, fmt="s--", capsize=5, label=f"{alg_abbr} {metric2}"
            )
            # Flatten the lines from both errorbar containers.
            lines1 = list(err1.lines) if isinstance(err1.lines, (list, tuple)) else [err1.lines]
            lines2 = list(err2.lines) if isinstance(err2.lines, (list, tuple)) else [err2.lines]
            for line in lines1 + lines2:
                if hasattr(line, "get_label"):
                    label = line.get_label()
                    if label not in all_handles:
                        all_handles[label] = line

        # TD3 horizontal line overlay for each metric separately.
        td3_mask = df["algorithm"].str.contains("td3_continuous_action.py", na=False)
        if td3_mask.sum() > 0:
            td3_avg1 = df[td3_mask][metric1].mean()
            td3_avg2 = df[td3_mask][metric2].mean()
            ax.axhline(y=td3_avg1, linestyle="--", color="black", linewidth=1, label="TD3 avg " + metric1)
            ax.axhline(y=td3_avg2, linestyle="--", color="gray", linewidth=1, label="TD3 avg " + metric2)

        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Alpha")
        ax.set_title(env_val)
        ax.grid(True)
        # Set y-axis to log scale.
        ax.set_yscale("log")
    
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
    print(f"Comparison plot saved to: {output_file}")
    plt.show()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Plot 1: Mean Return vs. Alpha.
    output_file_return = os.path.join(output_dir, "alpha_vs_mean_reward.png")
    plot_metric("mean_reward", "Mean Return", output_file_return, input_dir)

    # Plot 2: Mean KL Divergence vs. Alpha.
    output_file_kl = os.path.join(output_dir, "alpha_vs_mean_KL.png")
    plot_metric("mean_KL", "Mean KL Divergence", output_file_kl, input_dir)

    # Plot 3: Q-Network Difference vs. Alpha.
    output_file_q = os.path.join(output_dir, "alpha_vs_q_infnorm.png")
    plot_metric("q_infnorm", "Q Difference", output_file_q, input_dir)

    # Plot 4: Jacobian Difference vs. Alpha.
    output_file_jacobian = os.path.join(output_dir, "alpha_vs_jacobian_diff.png")
    plot_metric("jacobian_diff", "Jacobian Difference", output_file_jacobian, input_dir)

    # Plot 5: Comparison of Q Difference and KL Divergence vs. Alpha.
    output_file_comp = os.path.join(output_dir, "alpha_vs_q_infnorm_mean_KL_comparison.png")
    plot_comparison("q_infnorm", "mean_KL", "Metric Value", output_file_comp, input_dir)

if __name__ == "__main__":
    main()

