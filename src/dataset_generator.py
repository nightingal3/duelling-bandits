from time import sleep
from tqdm import tqdm

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from statsmodels.stats.power import GofChisquarePower
from typing import List
import seaborn as sns
from scipy.stats import chisquare
import pandas as pd

from uniform_sampling_policy import UniformSamplingPolicy
from double_thompson_sampling import DoubleThompsonSamplingPolicy
from gen_preference_matrix import PreferenceMatrix
from dataset import Dataset

import pdb

from ltr_matrices import get_rankers


def estimate_sample_size(
    num_actions: int,
    effect_size: float = 0.1,
    effect_size_threshold: float = 0.8,
    significance_threshold: float = 0.05,
) -> float:
    power_analysis = GofChisquarePower()  # is this the right power analysis?
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        power=effect_size_threshold,
        alpha=significance_threshold,
        n_bins=2,
    )
    total_sample_size = sample_size * len(
        list(combinations(range(num_actions), 2))
    )  # assume same effect size between arms for now
    return total_sample_size


def estimate_sample_size_in_total(
    pm: PreferenceMatrix,
    num_actions: int,
    effect_size_threshold: float = 0.8,
    significance_threshold: float = 0.05,
) -> float:
    total_sample_size = 0
    power_analysis = GofChisquarePower()  # is this the right power analysis?
    rewards = {}
    for i in range(0, num_actions):
        for j in range(i + 1, num_actions):
            effect_size = 2 * abs(pm.data[i][j] - 0.5)
            sample_size = power_analysis.solve_power(
                effect_size=effect_size,
                power=effect_size_threshold,
                alpha=significance_threshold,
                n_bins=2,
            )
            total_sample_size += sample_size
    return total_sample_size


def generate_rewards_for_arm(
    effect_size: float, sample_size: int, sample_size_multiple: float = 1
) -> List:
    if effect_size == 0:
        threshold = 0.5
    if effect_size == 0.1:
        threshold = 0.55
    elif effect_size == 0.3:
        threshold = 0.65
    elif effect_size == 0.5:
        threshold = 0.75
    else:
        threshold = 0.5 + effect_size / 2

    rewards = [
        1 if random.uniform(0, 1) < threshold else 0
        for _ in range(round(sample_size * sample_size_multiple))
    ]
    rewards_gen = (x for x in rewards)
    return rewards_gen


def pref_matrix_from_effect_size(
    effect_size: float, num_actions: int
) -> PreferenceMatrix:
    data = np.zeros((num_actions, num_actions))
    for i in range(num_actions):
        for j in range(i):
            if effect_size == 0.1:
                threshold = 0.55
            elif effect_size == 0.3:
                threshold = 0.65
            elif effect_size == 0.5:
                threshold = 0.75
            data[i][j] = threshold
            data[j][i] = 1 - threshold

    pm = PreferenceMatrix(num_actions=num_actions, data=data)
    return pm


def pref_matrix_no_effect(num_actions: int) -> PreferenceMatrix:
    data = np.full((num_actions, num_actions), 0.5)
    np.fill_diagonal(data, 0)
    pm = PreferenceMatrix(num_actions=num_actions, data=data)
    #pdb.set_trace()
    return pm


def condorcet_winner_gen(effect_size: float) -> None:
    raise NotImplementedError


def run_simulation(
    num_actions: int, effect_size: float = 0.3, sample_size_multiple: float = 1
) -> tuple:
    if effect_size is None:
        pm = get_rankers(num_actions)
    elif effect_size > 0:
        pm = pref_matrix_from_effect_size(
            effect_size=effect_size, num_actions=num_actions
        )

        # Comment the following code, if you don't want the winning frequency between the condorcer winnter and other arms to be random.
        pm.set_matrix_random_with_condorcet_winner(effect_size)
    else:
        pm = pref_matrix_no_effect(num_actions)

    combs = list(combinations(range(num_actions), 2))
    condorcet_winner = pm.condorcet_winner()
    policy = UniformSamplingPolicy(pm)
    policy_ts = DoubleThompsonSamplingPolicy(pm)
    if effect_size is None:
        sample_size = estimate_sample_size_in_total(
            pm = pm, num_actions=num_actions
        )

        # convention: 1 means the victory of the arm with the lowest index
        rewards = {
            combination: generate_rewards_for_arm(2 * abs(pm.data[combination[0]][combination[1]] - 0.5), sample_size)
            for combination in combs
        }        
    elif effect_size > 0:
        sample_size = estimate_sample_size(
            num_actions=num_actions, effect_size=effect_size
        )
        # convention: 1 means the victory of the arm with the lowest index
        rewards = {
            combination: generate_rewards_for_arm(effect_size, sample_size)
            for combination in combs
        }
    else:
        sample_size = estimate_sample_size(num_actions=num_actions, effect_size=0.3)

        # convention: 1 means the victory of the arm with the lowest index
        rewards = {
            combination: generate_rewards_for_arm(effect_size, sample_size)
            for combination in combs
        }



    num_duels = {combination: 0 for combination in combs}
    num_duels_ts = {combination: 0 for combination in combs}
    effect_size_over_time = []
    effect_size_over_time_uni = []
    effect_size_df = (
        []
    )  # columns=["timestep", "sampling_type", "action1", "action2", "power"])
    actions_ts = []
    actions_uni = []


    for i in range(int(sample_size * sample_size_multiple)):
        action1, action2 = policy.choose_actions()
        actions_uni.append((action1, action2))
        ts_action1, ts_action2 = policy_ts.choose_actions()
        actions_ts.append((ts_action1, ts_action2))
        duel = tuple(sorted([action1, action2]))
        ts_duel = tuple(sorted([ts_action1, ts_action2]))

        all_powers = policy_ts.get_all_power(effect_size)
        for power, p_action1, p_action2 in all_powers:
            chosen = p_action1 == ts_action1 and p_action2 == ts_action2
            effect_size_df.append(
                [
                    i + 1,
                    "thompson_sampling",
                    p_action1,
                    p_action2,
                    chosen,
                    power,
                    policy_ts.rewards_over_time[-1],
                ]
            )

        all_powers_uni = policy.get_all_power(effect_size)
        for power, p_action1_uni, p_action2_uni in all_powers_uni:
            chosen = p_action1_uni == action1 and p_action2_uni == action2
            effect_size_df.append(
                [
                    i + 1,
                    "uniform",
                    p_action1_uni,
                    p_action2_uni,
                    chosen,
                    power,
                    policy.rewards_over_time[-1],
                ]
            )

    effect_size_df = pd.DataFrame(
        effect_size_df,
        columns=[
            "timestep",
            "sampling_type",
            "action1",
            "action2",
            "chosen",
            "power",
            "borda_reward",
        ],
    )

    return effect_size_df, policy, policy_ts, actions_uni, actions_ts, condorcet_winner, sample_size


def run_multi_simulations(
    num_experiments: int,
    num_actions: int,
    effect_size: int = 0.3,
    sample_size_multiple: float = 1,
) -> None:
    possible_duels = list(combinations(range(num_actions), 2))

    duel_log = {
        "uni": {
            "found_effect": [],
            "t_stat": [],
            "p_val": [],
            "proportion_condorcet": [],
            "final_power": [],
            "power_over_time": pd.DataFrame(),
        },
        "ts": {
            "found_effect": [],
            "t_stat": [],
            "p_val": [],
            "proportion_condorcet": [],
            "final_power": [],
            "power_over_time": pd.DataFrame(),
        },
    }

    combinations_result = {
        "ts": {
            "simulation": [],
            "first_arm": [],
            "second_arm": [],
            "found_effect": [],
            "t_stats": [],
            "p_values": [],
            "final_powers": []
        },
        "uni": {
            "simulation": [],
            "first_arm": [],
            "second_arm": [],
            "found_effect": [],
            "t_stats": [],
            "p_values": [],
            "final_powers": []
        }
    }

    ts_result = {
        "simulation": [],
        "proportion_condorcet": [],
        "strong_regret": [],
        "weak_regret": []
    }

    uni_result = {
        "simulation": [],
        "proportion_condorcet": [],
        "strong_regret": [],
        "weak_regret": []
    }

    env = {
        "simulation": [],
        "effect_size": [],
        "num_actions": []
    }


    # Init a time table for this round of simulations.

    pbar = tqdm(total = num_experiments, position = 0, leave=True)
    for i in range(num_experiments):
        pbar.update(1)
        combinations_uni = {
            duel: {
                "found_effect": 0,
                "t_stat": [],
                "p_val": [],
                "proportion_condorcet": -1,
                "final_power": [],
                "power_over_time": [],
            }
            for duel in possible_duels
        }
        combinations_ts = {
            duel: {
                "found_effect": 0,
                "t_stat": [],
                "p_val": [],
                "proportion_condorcet": -1,
                "final_power": [],
                "power_over_time": []
            }
            for duel in possible_duels
        }

        experiment_results, policy, policy_ts, actions_uni, actions_ts, condorcet_winner, sample_size = run_simulation(
            num_actions, effect_size, sample_size_multiple
        )
        num_timesteps = len(actions_uni)

        env['simulation'].append(i)
        env['effect_size'] = effect_size
        env['num_actions'] = num_actions
        env['sample_size'] = sample_size

        for comb in possible_duels:
            action1, action2 = sorted(comb)

            wins1_ts = policy_ts.bandits[action1][action2].wins
            losses1_ts = policy_ts.bandits[action1][action2].losses
            wins1_uni = policy.wins[action1][action2]
            losses1_uni = policy.wins[action2][action1]

            if wins1_ts == 0 and losses1_ts == 0:
                # The action1 and action2 are never chosen at the same time.
                chisquare_ts = [float("NaN"), float("NaN")]
            else:
                chisquare_ts = chisquare([wins1_ts, losses1_ts])


            if chisquare_ts[1] < 0.05:
                combinations_ts[comb]["found_effect"] = 1
            combinations_ts[comb]["t_stat"] = chisquare_ts[0]
            combinations_ts[comb]["p_val"] = chisquare_ts[1]



            if wins1_uni == 0 and losses1_uni == 0:
                # The action1 and action2 are never chosen at the same time.
                chisquare_uni = [float("NaN"), float("NaN")]
            else:
                chisquare_uni = chisquare([wins1_uni, losses1_uni])
            if chisquare_uni[1] < 0.05:
                combinations_uni[comb]["found_effect"] = 1
            combinations_uni[comb]["t_stat"] = chisquare_uni[0]
            combinations_uni[comb]["p_val"] = chisquare_uni[1]

        proportion_condorcet_uni = len(
            [
                action
                for action in actions_uni
                if action[0] == condorcet_winner or action[1] == condorcet_winner
            ]
        ) / len(actions_uni)
        proportion_condorcet_ts = len(
            [
                action
                for action in actions_ts
                if action[0] == condorcet_winner or action[1] == condorcet_winner
            ]
        ) / len(actions_ts)

        final_effect_size_ts = experiment_results.loc[
            (experiment_results["timestep"] == int(num_timesteps) - 1)
            & (experiment_results["sampling_type"] == "thompson_sampling")
        ]["power"].values
        final_effect_size_uni = experiment_results.loc[
            (experiment_results["timestep"] == int(num_timesteps) - 1)
            & (experiment_results["sampling_type"] == "uniform")
        ]["power"].values

        power_over_time_uni = experiment_results.loc[
            experiment_results["sampling_type"] == "uniform"
        ][["timestep", "power"]].groupby("timestep").mean()

        power_over_time_ts = experiment_results.loc[
            experiment_results["sampling_type"] == "thompson_sampling"
        ][["timestep", "power"]].groupby("timestep").mean()

        duel_log["uni"]["found_effect"].append(
            [combinations_uni[comb]["found_effect"] for comb in combinations_uni]
        )
        duel_log["ts"]["found_effect"].append(
            [combinations_ts[comb]["found_effect"] for comb in combinations_ts]
        )
        duel_log["uni"]["t_stat"].append(
            [combinations_uni[comb]["t_stat"] for comb in combinations_uni]
        )
        duel_log["ts"]["t_stat"].append(
            [combinations_ts[comb]["t_stat"] for comb in combinations_ts]
        )
        duel_log["uni"]["p_val"].append(
            [combinations_uni[comb]["p_val"] for comb in combinations_uni]
        )
        duel_log["ts"]["p_val"].append(
            [combinations_ts[comb]["p_val"] for comb in combinations_ts]
        )
        duel_log["uni"]["proportion_condorcet"].append(proportion_condorcet_uni)
        duel_log["ts"]["proportion_condorcet"].append(proportion_condorcet_ts)
        duel_log["uni"]["final_power"].append(list(final_effect_size_uni))
        duel_log["ts"]["final_power"].append(list(final_effect_size_ts))


        power_over_time_uni['simulation'] = i #Add simulation id to power_over_time dataframe.
        power_over_time_ts['simulation'] = i #Add simulation id to power_over_time dataframe.

        duel_log["uni"]["power_over_time"] = duel_log["uni"]["power_over_time"].append(
            power_over_time_uni
        )
        duel_log["ts"]["power_over_time"] = duel_log["ts"]["power_over_time"].append(
            power_over_time_ts
        )

        # Write to the pytables.
        for comb in combinations_uni:
            found_effect = combinations_uni[comb]["found_effect"]
            t_stats = combinations_uni[comb]["t_stat"]
            p_vals = combinations_uni[comb]["p_val"]
            # Update the Combination Result for Uniform.
            combinations_result["uni"]["simulation"].append(i)
            combinations_result["uni"]["first_arm"].append(comb[0])
            combinations_result["uni"]["second_arm"].append(comb[1])
            combinations_result["uni"]["found_effect"].append(found_effect)
            combinations_result["uni"]["t_stats"].append(t_stats)
            combinations_result["uni"]["p_values"].append(p_vals)
            combinations_result["uni"]["final_powers"].append(list(final_effect_size_uni)[list(combinations_uni).index(comb)])

        uni_result["simulation"].append(i)
        uni_result["proportion_condorcet"].append(proportion_condorcet_uni)
        uni_result["strong_regret"].append(policy.strong_regret)
        uni_result["weak_regret"].append(policy.weak_regret)

        # dataset.update_uniform_table(i, proportion_condorcet_uni)
        for comb in combinations_ts:
            found_effect = combinations_ts[comb]["found_effect"]
            t_stats = combinations_ts[comb]["t_stat"]
            p_vals = combinations_ts[comb]["p_val"]
            # Update the Combination Table.
            combinations_result["ts"]["simulation"].append(i)
            combinations_result["ts"]["first_arm"].append(comb[0])
            combinations_result["ts"]["second_arm"].append(comb[1])
            combinations_result["ts"]["found_effect"].append(found_effect)
            combinations_result["ts"]["t_stats"].append(t_stats)
            combinations_result["ts"]["p_values"].append(p_vals)
            combinations_result["ts"]["final_powers"].append(list(final_effect_size_ts)[list(combinations_ts).index(comb)])

        # Update the Double Thompson Sampling Table.
        ts_result["simulation"].append(i)
        ts_result["proportion_condorcet"].append(proportion_condorcet_ts)
        ts_result["strong_regret"].append(policy_ts.strong_regret)
        ts_result["weak_regret"].append(policy_ts.weak_regret)


    dataset = Dataset("data/output_" + str(effect_size) + "_" + str(num_actions) + "_arms.h5", effect_size, num_actions, combinations_result, duel_log, ts_result, uni_result, env)

    print("DONE! :)")


def plot_effect_size_over_time(effect_size_df, filename: str = "testing.png") -> None:
    sns.lineplot(data=effect_size_df, x="timestep", y="power", hue="sampling_type")
    plt.axhline(y=0.8, color="r", linestyle="dotted")
    plt.savefig(filename)


def plot_effect_size_bars(filename: str = "testing.png", **effect_size_dfs) -> None:
    final_effect_size = pd.DataFrame()
    for num_timesteps, df in effect_size_dfs.items():
        final_effect_size_ts = df.loc[
            (df["timestep"] == int(num_timesteps) - 1)
            & (df["sampling_type"] == "thompson_sampling")
        ]
        final_effect_size_uni = df.loc[
            (df["timestep"] == int(num_timesteps) - 1)
            & (df["sampling_type"] == "uniform")
        ]

        final_effect_size = final_effect_size.append(final_effect_size_ts)
        final_effect_size = final_effect_size.append(final_effect_size_uni)

    sns.barplot(x="timestep", y="power", hue="sampling_type", data=final_effect_size)
    plt.savefig(filename)


def plot_borda_reward_bars(
    num_duels: int, filename: str = "testing.png", **dfs
) -> None:
    sum_borda_score = []
    for num_timesteps, df in dfs.items():
        borda_score_ts = (
            df.loc[df["sampling_type"] == "thompson_sampling"]["borda_reward"].sum()
        ) / (int(num_timesteps) * num_duels)
        borda_score_uni = df.loc[df["sampling_type"] == "uniform"][
            "borda_reward"
        ].sum() / (int(num_timesteps) * num_duels)

        sum_borda_score.append(
            [int(num_timesteps), "thompson_sampling", borda_score_ts]
        )
        sum_borda_score.append([int(num_timesteps), "uniform", borda_score_uni])

    sum_borda_score = pd.DataFrame(
        sum_borda_score, columns=["timestep", "sampling_type", "average reward"]
    )
    sns.barplot(
        x="timestep", y="average reward", hue="sampling_type", data=sum_borda_score
    )
    plt.savefig(filename)


def plot_borda_reward_over_time(
    reward_df: pd.DataFrame, filename: str = "testing.png"
) -> None:
    sns.lineplot(data=reward_df, x="timestep", y="borda_reward", hue="sampling_type")
    plt.savefig(filename)


if __name__ == "__main__":
    DATASET = str(sys.argv[1])
    if(DATASET == "RANDOM"):
        # The effect size is pre-set.
        EFFECT_SIZE = float(sys.argv[2])
        NUM_ARMS = int(sys.argv[3])
        NUM_SIMULATIONS = int(sys.argv[4])
    else:
        EFFECT_SIZE = None # There is no effect size that is pre-set
        NUM_ARMS = int(sys.argv[2])
        NUM_SIMULATIONS = int(sys.argv[3])
    SAMPLE_SIZE_MULT = 1

    print(
        f"EFFECT_SIZE:{EFFECT_SIZE}, NUM ARMS:{NUM_ARMS}, NUM SIMULATIONS:{NUM_SIMULATIONS}, SAMPLE SIZE MULTIPLE:{SAMPLE_SIZE_MULT}"
    )
    num_duels = len(list(combinations(range(NUM_ARMS), 2))) # number of possible cominations of duels between two arms, order does't matter. The smaller index is always in 0th index.

    # sample_size = estimate_sample_size(NUM_ARMS, effect_size=EFFECT_SIZE)
    # print(sample_size)
    run_multi_simulations(
        NUM_SIMULATIONS,
        NUM_ARMS,
        EFFECT_SIZE,
        sample_size_multiple=SAMPLE_SIZE_MULT,
    )
    
