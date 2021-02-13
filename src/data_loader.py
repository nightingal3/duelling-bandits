import sys
import pandas as pd
from dataset import Dataset
from tables import *
class DataLoder():
  def __init__(self, filename):
      self.tables = pd.HDFStore(filename)
      self.filename = filename

  def to_file(self):
    f = open(self.filename.replace(".h5", ".txt"), "w")
    ts_combinations = self.tables['ts_combinations']
    uni_combinations = self.tables['uni_combinations']
    ts_result = self.tables['ts_result']
    uni_result = self.tables['uni_result']
    env = self.tables['env']

    for index, sim in env.iterrows():
        simulation_id = int(sim['simulation'])
        effect_size = sim['effect_size']
        sample_size = int(sim['sample_size'])

        uni_combinations_for_this_sim = uni_combinations[uni_combinations['simulation'] == simulation_id]
        ts_combinations_for_this_sim = ts_combinations[ts_combinations['simulation'] == simulation_id]

        uni_result_for_this_sim = uni_result[uni_result['simulation'] == simulation_id].iloc[0]
        ts_result_for_this_sim = ts_result[ts_result['simulation'] == simulation_id].iloc[0]

        f.write(f"=== SIMULATION NUMBER {simulation_id} === \n")
        f.write(f"Effect size: {effect_size}, Sample size: {sample_size}\n")

        f.write("Uniform:\n")
        for index, combination in uni_combinations_for_this_sim.iterrows():
          f.write(f"\tCombination ({combination['first_arm']}, {combination['second_arm']}): Found effect? {combination['found_effect']}\tTest stats: {combination['t_stats']}\tp-vals: {combination['p_values']}\n")

        f.write(f"Proportion Condorcet: {uni_result_for_this_sim['proportion_condorcet']}\n")
        f.write(f"Strong Regret: {uni_result_for_this_sim['strong_regret']}\n")
        f.write(f"Weak Regret: {uni_result_for_this_sim['weak_regret']}\n\n")


        f.write("D-TS:\n")
        for index, combination in ts_combinations_for_this_sim.iterrows():
          f.write(f"\tCombination ({combination['first_arm']}, {combination['second_arm']}): Found effect? {combination['found_effect']}\tTest stats: {combination['t_stats']}\tp-vals: {combination['p_values']}\n")

        f.write(f"Proportion Condorcet: {ts_result_for_this_sim['proportion_condorcet']}\n")
        f.write(f"Strong Regret: {ts_result_for_this_sim['strong_regret']}\n")
        f.write(f"Weak Regret: {ts_result_for_this_sim['weak_regret']}\n\n")

if __name__ == "__main__":
  dataset = DataLoder(sys.argv[1])
  dataset.to_file()

