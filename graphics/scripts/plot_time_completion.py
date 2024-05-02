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


HOME = os.path.expanduser("~")
FILE_ROOT = os.path.join(HOME, "eval")
FILE_PATHS = {
    "PPO": [
        os.path.join(FILE_ROOT, "ppo", "eval1.csv"),
        os.path.join(FILE_ROOT, "ppo", "eval2.csv"),
        os.path.join(FILE_ROOT, "ppo", "eval3.csv"),
    ],
    "PPO-STACK": [
        os.path.join(FILE_ROOT, "ppo_10obs", "eval1.csv"),
        os.path.join(FILE_ROOT, "ppo_10obs", "eval2.csv"),
        os.path.join(FILE_ROOT, "ppo_10obs", "eval3.csv"),
    ],
    "SAC": [
        os.path.join(FILE_ROOT, "sac", "eval1.csv"),
        os.path.join(FILE_ROOT, "sac", "eval2.csv"),
        os.path.join(FILE_ROOT, "sac", "eval3.csv"),
    ],
    "SAC-STACK": [
        os.path.join(FILE_ROOT, "sac_10obs", "eval1.csv"),
        os.path.join(FILE_ROOT, "sac_10obs", "eval2.csv"),
        os.path.join(FILE_ROOT, "sac_10obs", "eval3.csv"),
    ],
    "DreamerV3": [
        os.path.join(FILE_ROOT, "dreamerv3", "eval1.csv"),
        os.path.join(FILE_ROOT, "dreamerv3", "eval2.csv"),
        os.path.join(FILE_ROOT, "dreamerv3", "eval3.csv"),
    ],
}


def preprocess_csv_data(file_paths):
    all_data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    data["time_until_success_seconds"] = data["steps_until_success"] / 50.0

    # Set unsuccessful episodes beyond 10 seconds
    data.loc[data["is_success"] == False, "time_until_success_seconds"] = 11.0

    return data


def plot_time_until_completion(aggregated_data_dict):
    plt.figure(figsize=(16, 10))
    for label, data in aggregated_data_dict.items():
        valid_data = data[data["time_until_success_seconds"] <= 10.0]
        if label != "PPO":
            plt.axvline(
                valid_data["time_until_success_seconds"].median(),
                color=sns.color_palette("muted")[
                    list(aggregated_data_dict.keys()).index(label)
                ],
                linestyle="dashed",
                linewidth=3,
            )
    for label, data in aggregated_data_dict.items():
        sns.kdeplot(
            data["time_until_success_seconds"],
            label=label,
            clip=(0.0, 10.0),
            bw_adjust=0.5,
            gridsize=2000,
            levels=100,
            thresh=0.0,
            cut=5,
        )

    plt.xlabel("Time until success")
    plt.xticks(
        np.linspace(0.0, 10.0, int(10.0 / 2.0) + 1),
        [
            f"{int(x)} s" if x > 0.0 else "0"
            for x in np.linspace(0.0, 10.0, int(10.0 / 2.0) + 1)
        ],
    )
    plt.xlim(0.0, 10.0)

    plt.ylabel("Density")
    plt.yticks(
        np.linspace(0.0, 0.60000000001, int(0.60000000001 / 0.10) + 1),
        [
            f"{int(x*100)}%" if x > 0.0 else ""
            for x in np.linspace(0.0, 0.60000000001, int(0.60000000001 / 0.10) + 1)
        ],
    )
    plt.ylim(0.0, 0.60000000001)

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
        aggregated_data = preprocess_csv_data(paths)
        aggregated_data_dict[agent] = aggregated_data

    # Calculating the overall statistics
    for agent, data in aggregated_data_dict.items():
        total_success_rate = data["is_success"].mean()

        success_rates = (
            data[data["is_success"] == 1].groupby("env_id").size()
            / data.groupby("env_id").size()
        )
        if len(success_rates) == 0 or success_rates.isna().all():
            continue
        time_until_success = (
            data[data["is_success"] == 1]
            .groupby("env_id")["steps_until_success"]
            .mean()
            / 50
        )

        valid_time_until_success = data[data["time_until_success_seconds"] <= 10.0]
        median_time_until_success = valid_time_until_success[
            "time_until_success_seconds"
        ].median()

        highest_success_rate = success_rates.max()
        envs_with_highest_success_rate = success_rates[
            success_rates == highest_success_rate
        ].index
        best_env_id = time_until_success[envs_with_highest_success_rate].idxmin()

        lowest_success_rate = success_rates.min()
        envs_with_lowest_success_rate = success_rates[
            success_rates == lowest_success_rate
        ].index
        worst_env_id = time_until_success[envs_with_lowest_success_rate].idxmax()

        print(f"Agent: {agent}")
        print(f"Total Success Rate: {total_success_rate:.2%}")
        print(f"Median Time Until Success: {median_time_until_success:.2f} seconds")
        print(
            f"Best Performing Environment: Env {best_env_id} with Success Rate: {success_rates[best_env_id]:.2%} and Time: {time_until_success[best_env_id]:.2f} seconds"
        )
        print(
            f"Worst Performing Environment: Env {worst_env_id} with Success Rate: {success_rates[worst_env_id]:.2%} and Time: {time_until_success[worst_env_id]:.2f} seconds"
        )
        print()

    # Combining all data for overall statistics
    combined_data = pd.concat(aggregated_data_dict.values())
    overall_success_rates = (
        combined_data[combined_data["is_success"] == 1].groupby("env_id").size()
        / combined_data.groupby("env_id").size()
    )
    overall_time_until_success = (
        combined_data[combined_data["is_success"] == 1]
        .groupby("env_id")["steps_until_success"]
        .mean()
        / 50
    )

    highest_success_rate = overall_success_rates.max()
    envs_with_highest_success_rate = overall_success_rates[
        overall_success_rates == highest_success_rate
    ].index
    best_env_id = overall_time_until_success[envs_with_highest_success_rate].idxmin()

    lowest_success_rate = overall_success_rates.min()
    envs_with_lowest_success_rate = overall_success_rates[
        overall_success_rates == lowest_success_rate
    ].index
    worst_env_id = overall_time_until_success[envs_with_lowest_success_rate].idxmax()

    print("Overall Statistics:")
    print(
        f"Best Performing Environment: Env {best_env_id} with Success Rate: {overall_success_rates[best_env_id]:.2%} and Time: {overall_time_until_success[best_env_id]:.2f} seconds"
    )
    print(
        f"Worst Performing Environment: Env {worst_env_id} with Success Rate: {overall_success_rates[worst_env_id]:.2%} and Time: {overall_time_until_success[worst_env_id]:.2f} seconds"
    )
    print()

    plot_time_until_completion(aggregated_data_dict)


if __name__ == "__main__":
    main()
