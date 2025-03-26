#!/usr/bin/env python3
"""
Visualization script that:
 - Reads all CSV files in a directory, possibly matching a specific pattern like '*pairwise_per_episode.csv'
 - For each CSV, aggregates per-episode data by (algorithm_abbr, alpha), and plots:
   - "mean_reward" (from "episode_reward")
   - "mean_KL" (from "episode_kl")
   - "q_infnorm" (from "episode_q_infnorm")
   - "jacobian_diff" (from "episode_jacobian_diff")
 - Eliminates duplicate lines for the same algorithm by unifying them under one "algorithm_abbr".
 - Draws lines for alpha != 1 and a single dashed line for alpha == 1, with a distinct color for "autotune SAC".
 - If 'td3_continuous_action.py' is present, draws a dashed line for average across all episodes (TD3 IQM).
 - Places the figure title at the top, legend below the title, subplots in the middle, and step-text below each subplot.
 - Rotates all tick labels and scientific-notation offsets by 45°.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# A column mapping for your script to unify e.g. "mean_KL" => "episode_kl"
METRIC_COLUMN_MAPPING = {
    "mean_reward": "episode_reward",
    "mean_KL": "episode_kl",
    "q_infnorm": "episode_q_infnorm",
    "jacobian_diff": "episode_jacobian_diff",
    # If you want to do a "probability_of_improvement" plot, you can add e.g.:
    # "probability_of_improvement": "episode_probability_of_improvement"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize per-episode evaluation results.")
    parser.add_argument(
        "--input_dir", type=str, default="results",
        help="Directory containing input CSV files (default: results)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations",
        help="Directory where the output plots will be saved (default: visualizations)"
    )
    parser.add_argument(
        "--file_pattern", type=str, default="pairwise_per_episode.csv",
        help="Glob pattern to match CSV files. E.g. '*pairwise_per_episode.csv' (default)."
    )
    return parser.parse_args()

def abbreviate_algorithm(alg_full):
    """
    If the base name includes 'liv-autotune', return 'sac-liv-autotune'.
    Otherwise, split on '_' and return first chunk e.g. "sac", "meow", etc.
    """
    base = os.path.basename(alg_full)
    name_no_ext = os.path.splitext(base)[0]
    if "liv-autotune" in name_no_ext:
        return "sac-liv-autotune"
    else:
        return name_no_ext.split('_')[0]

def iqm_func(x):
    """Compute the InterQuartile Mean of array x."""
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    mask = (x >= q1) & (x <= q3)
    return np.mean(x[mask])

def q1_func(x):
    return np.percentile(x, 25)

def q3_func(x):
    return np.percentile(x, 75)

def parse_global_steps_from_checkpoint(path: str) -> str:
    """
    Parse out the integer after 'step' in the checkpoint path
    (e.g. '..._step100000.pth' -> '100000'),
    then convert it to e.g. '0.1M Global Steps'.
    """
    match = re.search(r'step(\d+)', path)
    if match:
        step_int = int(match.group(1))
        return format_steps(step_int)
    return ""

def format_steps(n: int) -> str:
    """Convert a numeric step count into e.g. '0.1M Global Steps'."""
    if n >= 1e6:
        return f"{n/1e6:.1f}M Global Steps"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K Global Steps"
    else:
        return f"{n} Global Steps"

def plot_metric(metric, ylabel, output_file, input_dir, file_pattern="*.csv"):
    """
    Creates a figure with subplots side-by-side, one per CSV file matching file_pattern.
    The layout is:
        Title (top)
        Legend (below title)
        Subplots row
        Step text below each subplot
    All ticks + scientific offsets rotated by 45°, subplots auto-scale y-lims.
    """
    csv_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".csv") and file_pattern in f
    ]
    if not csv_files:
        print(f"No CSV files found matching pattern '{file_pattern}' in {input_dir}.")
        return

    # We'll unify columns e.g. "mean_KL" -> "episode_kl"
    actual_col = METRIC_COLUMN_MAPPING.get(metric, metric)
    num_files = len(csv_files)

    # Make a figure large enough for num_files subplots, plus room for title/legend
    fig = plt.figure(figsize=(4.5 * num_files, 5))
    fig.canvas.draw()
    # We'll store handles in a dictionary to build a legend at the end
    all_handles = {}

    # Base color scheme for normal lines
    algorithm_colors = {
        'sac': 'orange',
        'sac-liv-autotune': 'blue',
        'meow': 'green',
        'td3': 'red',
    }

    # Set the suptitle at the top
    fig.suptitle(f"{ylabel} vs Alpha", fontsize=16, y=1.0)
    # Add an annotation indicating IQM is used for evaluation
    fig.text(0.5, 0.97, "Note: Values are computed using IQM (Interquartile Mean) on episode data", 
             ha="center", fontsize=10)

    # Create each subplot
    for i, csv_file in enumerate(csv_files):
        # add_subplot(rows, cols, index)
        ax = fig.add_subplot(1, num_files, i+1)
        print(f"Reading CSV: {csv_file}")
        df = pd.read_csv(csv_file)

        # Rename the column if needed
        if actual_col in df.columns and actual_col != metric:
            df.rename(columns={actual_col: metric}, inplace=True)

        # Abbreviate the algorithm
        df["algorithm_abbr"] = df["algorithm"].apply(abbreviate_algorithm)

        # Ensure we have alpha
        if "alpha" not in df.columns:
            if "actor_i_alpha" in df.columns:
                df["alpha"] = pd.to_numeric(df["actor_i_alpha"], errors="coerce")
            else:
                df["alpha"] = np.nan

        # If the metric isn't in the data, skip
        if metric not in df.columns:
            print(f"Skipping. Column '{metric}' not found in {csv_file}.")
            continue

        # Attempt to glean environment name
        if "env" in df.columns and not df["env"].isna().all():
            env_val = df["env"].iloc[0]
        else:
            env_val = os.path.basename(csv_file)

        # Filter out rows with NaN in the metric
        df_metric = df.dropna(subset=[metric])

        # Group by (algorithm_abbr, alpha) => compute IQM, Q1, Q3
        grouped = df_metric.groupby(["algorithm_abbr", "alpha"]).agg(
            iqm=(metric, iqm_func),
            q1=(metric, q1_func),
            q3=(metric, q3_func),
            count=(metric, 'count')
        ).reset_index()

        # For each algorithm, plot lines
        alg_abbr_list = grouped["algorithm_abbr"].unique()
        for alg_abbr in alg_abbr_list:
            color = algorithm_colors.get(alg_abbr, 'gray')
            df_alg = grouped[grouped["algorithm_abbr"] == alg_abbr]
            if df_alg.empty:
                continue

            # For alpha != 1: normal line with error bars
            df_alg_non1 = df_alg[df_alg["alpha"] != 1]
            if not df_alg_non1.empty:
                x_vals = df_alg_non1["alpha"]
                y_vals = df_alg_non1["iqm"]
                y_low = df_alg_non1["q1"]
                y_high = df_alg_non1["q3"]
                y_err_lower = y_vals - y_low
                y_err_upper = y_high - y_vals
                err_container = ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=[y_err_lower, y_err_upper],
                    fmt="o-",
                    capsize=5,
                    color=color
                )
                main_line = err_container.lines[0]
                main_line.set_label(alg_abbr)
                if alg_abbr not in all_handles:
                    all_handles[alg_abbr] = main_line

            # For alpha == 1: single dashed line
            df_alg_one = df_alg[df_alg["alpha"] == 1]
            if not df_alg_one.empty:
                weighted_sum = (df_alg_one["iqm"] * df_alg_one["count"]).sum()
                total_count = df_alg_one["count"].sum()
                if total_count > 0:
                    hline_y = weighted_sum / total_count
                else:
                    hline_y = df_alg_one["iqm"].mean()

                # Distinct color for "sac" or "sac-liv-autotune"
                if alg_abbr == "sac":
                    line_label = "autotune SAC"
                    color_line = "purple"
                elif alg_abbr == "sac-liv-autotune":
                    line_label = "liv-autotune SAC"
                    color_line = "blue"
                else:
                    line_label = f"{alg_abbr} @α=1"
                    color_line = color

                dash_line = ax.axhline(
                    y=hline_y,
                    linestyle="--",
                    color=color_line,
                    linewidth=2,
                    label=line_label
                )
                if line_label not in all_handles:
                    all_handles[line_label] = dash_line

        # TD3 overlay: if "td3_continuous_action.py" appears in "algorithm", 
        # we do a dashed black line at the IQM
        td3_mask = df_metric["algorithm"].str.contains("td3_continuous_action.py", na=False)
        if td3_mask.any():
            td3_vals = df_metric.loc[td3_mask, metric]
            if len(td3_vals) > 0:
                td3_iqm = iqm_func(td3_vals)
                td3_line = ax.axhline(
                    y=td3_iqm,
                    linestyle="--",
                    color="black",
                    linewidth=2,
                    label="TD3 IQM"
                )
                if "TD3 IQM" not in all_handles:
                    all_handles["TD3 IQM"] = td3_line

        # Labeling
        ax.set_title(env_val, fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Alpha", fontsize=10)

        # Axis scale
        # Suppose you've decided whether to use log or linear already:
        if ax.get_yscale() == 'log':
            from matplotlib.ticker import LogFormatterSciNotation
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        else:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        fig.canvas.draw()
        
        # Rotate offset text if it exists
        offset_text_y = ax.yaxis.get_offset_text()
        offset_text_y.set_rotation(45)
        offset_text_y.set_ha('right')

        ax.grid(True)

        # Steps text below each subplot
        if "checkpoint_i" in df.columns and not df["checkpoint_i"].isna().all():
            ckpt_str = df["checkpoint_i"].iloc[0]
            step_text = parse_global_steps_from_checkpoint(ckpt_str)
            if step_text:
                ax.text(
                    0.5, -0.20,
                    step_text,
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=9,
                    color='black'
                )

    # -- At this point, subplots are created. Let's do a forced draw
    #    so the offset text objects are generated, then rotate them.
    fig.canvas.draw()

    # Create the legend from the collected handles
    sorted_labels = sorted(all_handles.keys())
    sorted_handles = [all_handles[lbl] for lbl in sorted_labels]

    # Place legend below the suptitle but above subplots
    legend = fig.legend(
        sorted_handles, sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),  # slightly under the title at y=1.0
        ncol=len(sorted_labels),
        frameon=True
    )
    legend.get_frame().set_edgecolor('black')

    # Adjust spacing so there's:
    #  - Enough room for the suptitle at the top
    #  - Legend below that
    #  - Subplots in the middle
    #  - Step text at the bottom
    # Tweak these numbers if you need more/less space
    fig.subplots_adjust(
        top=0.80,      # or however much room you need at top
        bottom=0.15,   # or however much room you need at bottom
        left=0.07,     # left margin
        right=0.97,    # right margin
        wspace=0.15,   # << reduce this to bring subplots closer together
        hspace=0.3     # vertical space (if you had multiple rows)
    )

    # Save + show
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_pattern = args.file_pattern  # e.g. "pairwise_per_episode.csv"

    os.makedirs(output_dir, exist_ok=True)

    # 1) mean_reward
    plot_metric("mean_reward", "Mean Reward",
                os.path.join(output_dir, "alpha_vs_mean_reward.png"),
                input_dir, file_pattern=file_pattern)

    # 2) mean_KL
    plot_metric("mean_KL", "Mean KL Divergence",
                os.path.join(output_dir, "alpha_vs_mean_KL.png"),
                input_dir, file_pattern=file_pattern)

    # 3) q_infnorm
    plot_metric("q_infnorm", "Q Difference",
                os.path.join(output_dir, "alpha_vs_q_infnorm.png"),
                input_dir, file_pattern=file_pattern)

    # 4) jacobian_diff
    plot_metric("jacobian_diff", "Jacobian Difference",
                os.path.join(output_dir, "alpha_vs_jacobian_diff.png"),
                input_dir, file_pattern=file_pattern)

    # 5) Probability of improvement (if your CSVs have "probability_of_improvement")
    plot_metric("probability_of_improvement", "Probability of Improvement",
                os.path.join(output_dir, "alpha_vs_probability_of_improvement.png"),
                input_dir, file_pattern=file_pattern)

if __name__ == "__main__":
    main()
