#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Activate seaborn for better plot aesthetics
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="muted",
    font="Times New Roman",
    font_scale=5.2,
    color_codes=True,
    rc={"figure.figsize": (16, 10), "lines.linewidth": 5},
)


METRIC = "success"

if "reward" in METRIC:
    ALPHA_MEAN = {
        "PPO": 3e-5,
        "PPO-STACK": 3e-5,
        "SAC": 3e-5,
        "SAC-STACK": 3e-5,
        "DreamerV3": 9e-5,
    }
    ALPHA_STD = {
        "PPO": 0.1 * ALPHA_MEAN["PPO"],
        "PPO-STACK": 0.1 * ALPHA_MEAN["PPO-STACK"],
        "SAC": 0.1 * ALPHA_MEAN["SAC"],
        "SAC-STACK": 0.1 * ALPHA_MEAN["SAC-STACK"],
        "DreamerV3": 0.1 * ALPHA_MEAN["DreamerV3"],
    }
elif "success" in METRIC:
    ALPHA_MEAN = {
        "PPO": 2e-5,
        "PPO-STACK": 2e-5,
        "SAC": 1e-5,
        "SAC-STACK": 1e-5,
        "DreamerV3": 4e-5,
    }
    ALPHA_STD = {
        "PPO": ALPHA_MEAN["PPO"],
        "PPO-STACK": ALPHA_MEAN["PPO-STACK"],
        "SAC": ALPHA_MEAN["SAC"],
        "SAC-STACK": ALPHA_MEAN["SAC-STACK"],
        "DreamerV3": ALPHA_MEAN["DreamerV3"],
    }


HOME = os.path.expanduser("~")
FILE_ROOT = os.path.join(HOME, "logdir")
FILE_PATHS = {
    "PPO": [
        os.path.join(FILE_ROOT, "ppo", "peg_in_hole1", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "ppo", "peg_in_hole2", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "ppo", "peg_in_hole3", "monitor.monitor.csv"),
    ],
    "PPO-STACK": [
        os.path.join(FILE_ROOT, "ppo_10obs", "peg_in_hole1", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "ppo_10obs", "peg_in_hole2", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "ppo_10obs", "peg_in_hole3", "monitor.monitor.csv"),
    ],
    "SAC": [
        os.path.join(FILE_ROOT, "sac", "peg_in_hole1", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "sac", "peg_in_hole2", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "sac", "peg_in_hole3", "monitor.monitor.csv"),
    ],
    "SAC-STACK": [
        os.path.join(FILE_ROOT, "sac_10obs", "peg_in_hole1", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "sac_10obs", "peg_in_hole2", "monitor.monitor.csv"),
        os.path.join(FILE_ROOT, "sac_10obs", "peg_in_hole3", "monitor.monitor.csv"),
    ],
    "DreamerV3": [
        os.path.join(FILE_ROOT, "dreamerv3", "peg_in_hole1", "metrics.jsonl"),
        os.path.join(FILE_ROOT, "dreamerv3", "peg_in_hole2", "metrics.jsonl"),
        os.path.join(FILE_ROOT, "dreamerv3", "peg_in_hole3", "metrics.jsonl"),
    ],
}


def preprocess_csv_data(file_paths):
    all_data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df["cumulative_steps"] = df["l"].cumsum()
        df["ep_success"] = df["is_success"].astype(int)
        all_data.append(df[["cumulative_steps", "r", "ep_success"]])
    return pd.concat(all_data, ignore_index=True)


def preprocess_jsonl_data(file_paths):
    all_data = []
    for file_path in file_paths:
        df = pd.read_json(file_path, lines=True)
        df["ep_success"] = (df["episode/score"] == 1.0).astype(int)
        all_data.append(
            df[["step", "episode/score", "ep_success"]].rename(
                columns={"step": "cumulative_steps", "episode/score": "r"}
            )
        )
    return pd.concat(all_data, ignore_index=True)


def aggregate_data(data):
    grouped_data = data.groupby("cumulative_steps").agg(
        {"r": ["mean", "std"], "ep_success": ["mean", "std"]}
    )
    grouped_data.columns = ["reward_mean", "reward_std", "success_mean", "success_std"]
    return grouped_data.reset_index()


def apply_ema(data, alpha):
    ema_data = data.ewm(alpha=alpha).mean()
    return ema_data


def aggregate_and_smooth_data(
    data, alpha_mean=0.001, alpha_std=0.0001, metric="reward_mean", downsampling_rate=10
):
    """
    Aggregate data, apply EMA smoothing, downsample the data points, and clip the maximum reward.
    """
    grouped_data = data.groupby("cumulative_steps").agg(
        {"r": ["mean", "std"], "ep_success": ["mean", "std"]}
    )
    grouped_data.columns = ["reward_mean", "reward_std", "success_mean", "success_std"]

    # Apply EMA for reward
    if "reward" in metric:
        grouped_data["reward_mean"] = apply_ema(
            grouped_data[["reward_mean"]], alpha=alpha_mean
        )["reward_mean"]
        grouped_data["reward_std"] = apply_ema(
            grouped_data[["reward_std"]], alpha=alpha_std
        )["reward_std"]

    # Apply EMA for success rate
    if "success" in metric:
        grouped_data["success_mean"] = apply_ema(
            grouped_data[["success_mean"]], alpha=alpha_mean
        )["success_mean"]
        grouped_data["success_std"] = apply_ema(
            grouped_data[["success_std"]], alpha=alpha_std
        )["success_std"]

    # Downsample the data points
    if downsampling_rate > 1:
        grouped_data = grouped_data.iloc[::downsampling_rate, :]

    return grouped_data.reset_index()


def plot_learning_curves(aggregated_data_dict, metric="reward_mean"):
    plt.figure(figsize=(16, 10))
    for label, data in aggregated_data_dict.items():
        if "reward" in metric:
            plt.plot(data["cumulative_steps"], data["reward_mean"], label=f"{label}")
            plt.fill_between(
                data["cumulative_steps"],
                (data["reward_mean"] - 0.5 * data["reward_std"]),
                (data["reward_mean"] + 0.5 * data["reward_std"]).clip(upper=1.0),
                alpha=0.1,
            )
        elif "success" in metric:
            plt.plot(data["cumulative_steps"], data["success_mean"], label=f"{label}")
            plt.fill_between(
                data["cumulative_steps"],
                (data["success_mean"] - 0.5 * data["success_std"]).clip(lower=0.0),
                (data["success_mean"] + 0.5 * data["success_std"]).clip(upper=1.0),
                alpha=0.1,
            )

    plt.xlabel("Timestep")
    plt.xticks(
        range(0, 100000001, 20000000),
        [
            f"{int(x/1000000)}M" if x > 0.0 else "0"
            for x in range(0, 100000001, 20000000)
        ],
    )
    plt.xlim(0, 100000000)

    if "reward" in metric:
        plt.ylabel("Reward")
    elif "success" in metric:
        plt.ylabel("Success rate")
        plt.yticks(
            np.linspace(0.0, 1.0, int(1.0 / 0.2) + 1),
            [
                f"{int(x*100)}%" if x > 0.0 else ""
                for x in np.linspace(0.0, 1.0, int(1.0 / 0.2) + 1)
            ],
        )
        plt.ylim(0.0, 1.0)

    plt.margins(0, 0)
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=5,
        handletextpad=0.2,
        handlelength=1.5,
    )
    plt.show()


def main():
    aggregated_data_dict = {}
    for agent, paths in FILE_PATHS.items():
        if agent == "DreamerV3":
            processed_data = preprocess_jsonl_data(paths)
        else:
            processed_data = preprocess_csv_data(paths)
        aggregated_data = aggregate_and_smooth_data(
            processed_data,
            alpha_mean=ALPHA_MEAN[agent],
            alpha_std=ALPHA_STD[agent],
            metric=METRIC,
            downsampling_rate=50,  # Downsample the data points by taking every 10th point
        )
        aggregated_data_dict[agent] = aggregated_data

    # Plot for success rate
    plot_learning_curves(aggregated_data_dict, metric=METRIC)


if __name__ == "__main__":
    main()
