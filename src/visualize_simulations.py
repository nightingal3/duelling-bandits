import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import List
import numpy as np
import pandas as pd
import pdb

# expected condorcet for 5 arms - 0.4
def proportion_condorcet_winner_hist(
    proportions_uni: List,
    proportions_ts: List,
    out_filename: str,
    expected_condorcet: float = 0.66,
) -> None:
    plt.gcf().clear()
    plt.hist(
        proportions_uni,
        bins=np.arange(0, 1, 0.01),
        alpha=0.4,
        label="Uniform assignment",
    )
    plt.hist(
        proportions_ts,
        bins=np.arange(0, 1, 0.01),
        alpha=0.4,
        label="Double Thompson Sampling\nassignment",
    )
    plt.axvline(
        x=expected_condorcet,
        color="red",
        linestyle="dotted",
        label="Expected assignment",
    )
    plt.xlabel("Proportion of trials\nwith Condorcet winner", fontsize=12)
    plt.ylabel("Number of simulations", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_filename)


def plot_effect_size_over_time(effect_size_df, filename: str = "testing.png") -> None:
    sns.lineplot(data=effect_size_df, x="timestep", y="power", hue="sampling_type")
    plt.axhline(y=0.8, color="r", linestyle="dotted")
    plt.savefig(filename)


def plot_effect_size_bars(filename: str = "testing.png", **effect_size_dfs) -> None:
    final_effect_size = pd.DataFrame()
    for num_timesteps, df in effect_size_dfs.items():
        final_effect_size_ts = df.loc[
            (df["timestep"] == int(num_timesteps))
            & (df["sampling_type"] == "Double Thompson sampling")
        ]
        final_effect_size_uni = df.loc[
            (df["timestep"] == int(num_timesteps))
            & (df["sampling_type"] == "Uniform sampling")
        ]

        final_effect_size = final_effect_size.append(final_effect_size_ts)
        final_effect_size = final_effect_size.append(final_effect_size_uni)

    sns.barplot(x="timestep", y="power", hue="sampling_type", data=final_effect_size)
    plt.axhline(y=0.8, color="r", linestyle="dotted")
    plt.savefig(filename)


def false_pos_rate(found_effects_ts: List, found_effects_uni: List) -> tuple:
    found_effects_ts = np.array(found_effects_ts).flatten()
    found_effects_uni = np.array(found_effects_uni).flatten()

    fp_ts = sum(found_effects_ts) / (found_effects_ts.shape[0])
    fp_uni = sum(found_effects_uni) / (found_effects_uni.shape[0])

    return fp_ts, fp_uni


if __name__ == "__main__":
    EFFECT_SIZE = 0.1
    NUM_ARMS = 3
    SAMP_SIZE_MULT = 5
    data = pickle.load(
        open(f"./data/res_{EFFECT_SIZE}_{NUM_ARMS}_mult{SAMP_SIZE_MULT}.p", "rb")
    )
    data_uni = data["uni"]
    data_ts = data["ts"]
    # print(false_pos_rate(data_ts["found_effect"], data_uni["found_effect"]))
    # assert False
    data_uni["power_over_time"]["sampling_type"] = ["Uniform sampling"] * len(
        data_uni["power_over_time"].index
    )
    data_ts["power_over_time"]["sampling_type"] = ["Double Thompson sampling"] * len(
        data_ts["power_over_time"].index
    )
    combined_power_time = data_uni["power_over_time"].append(data_ts["power_over_time"])
    sample_size_5x = max(combined_power_time["timestep"])
    sample_size = sample_size_5x // 5
    sample_size_05x = sample_size // 2
    # plot_effect_size_over_time(combined_power_time, f"power_over_time_{EFFECT_SIZE}_{NUM_ARMS}_arm.png")
    # proportion_condorcet_winner_hist(data_uni["proportion_condorcet"], data_ts["proportion_condorcet"], f"proportion_condorcet_{EFFECT_SIZE}_{NUM_ARMS}arm.png", expected_condorcet=0.4)
    plot_effect_size_bars(
        f"effect_size_{EFFECT_SIZE}_{NUM_ARMS}_arms.png",
        **{
            str(sample_size_5x): combined_power_time,
            str(sample_size): combined_power_time,
            str(sample_size_05x): combined_power_time,
        },
    )
