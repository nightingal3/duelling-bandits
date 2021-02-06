import sys
from tables import *
class DataLoder():
  def __init__(self, h5_file):
    self.h5_file = open_file(h5_file, mode="r")

  def to_file(self, filename):
    f = open(filename, "w")
    effect_size_table = self.h5_file.root.effect_size # Every simulation has exact one effect size.
    for i in range(0, len(effect_size_table)):
      simulation_id = effect_size_table[i]["simulation_id"]
      effect_size = str(effect_size_table[i]["value"])
      sample_size = str(self.get_sample_size(simulation_id))

      combinations_uni = self.get_combinations(simulation_id, 0)
      combinations_ts = self.get_combinations(simulation_id, 1)

      uni = self.get_uniform(simulation_id)
      ts = self.get_thompson_sampling(simulation_id)

      f.write(f"=== SIMULATION NUMBER {simulation_id} === \n")
      f.write(f"Effect size: {effect_size}, Sample size: {sample_size}\n")
      f.write("Uniform:\n")
      for combination_uni in combinations_uni:
        f.write(f"\tCombination ({combination_uni['first_arm']}, {combination_uni['second_arm']}): Found effect? {combination_uni['find_effect']}\tTest stats: {combination_uni['test_stats']}\tp-vals: {combination_uni['p_values']}\n")
      f.write(f"Proportion Condorcet: {uni['proportion_condorcet']}\n")




      f.write("D-TS:\n")
      for combination_ts in combinations_ts:
        f.write(f"\tCombination ({combination_ts['first_arm']}, {combination_ts['second_arm']}): Found effect? {combination_ts['find_effect']}\tTest stats: {combination_ts['test_stats']}\tp-vals: {combination_ts['p_values']}\n")

      f.write(f"Proportion Condorcet: {ts['proportion_condorcet']}\n")
      f.write(f"Strong Regret: {ts['strong_regret']}\n")
      f.write(f"Weak Regret: {ts['weak_regret']}\n\n")
    f.close()

  def get_sample_size(self, simulation_id):
    table = self.h5_file.root.sample_size
    result = [{"sample_size": row['value']} for row in table.iterrows() if row['simulation_id'] == simulation_id]
    return result[0]["sample_size"]

  def get_combinations(self, simulation_id, is_DTS):
    table = self.h5_file.root.combination
    result = [{"first_arm": row['first_arm'], "second_arm": row['second_arm'], "find_effect": row['find_effect'], "test_stats": row['test_stats'], "p_values": row['p_values']} for row in table.iterrows() if row['simulation_id'] == simulation_id and row['is_DTS'] == is_DTS]
    return result

  def get_uniform(self, simulation_id):
    table = self.h5_file.root.uniform
    result = [
      {"proportion_condorcet": row['proportion_condorcet']}
       for row in table.iterrows() if row['simulation_id'] == simulation_id
    ]
    return result[0]

  def get_thompson_sampling(self, simulation_id):
    table = self.h5_file.root.double_thompson_sampling
    result = [
      {"proportion_condorcet": row['proportion_condorcet'], "strong_regret": row['strong_regret'], "weak_regret": row['weak_regret']}
       for row in table.iterrows() if row['simulation_id'] == simulation_id
    ]
    return result[0]  


if __name__ == "__main__":
  data_loader = DataLoder(sys.argv[1])

  data_loader.to_file(sys.argv[1].replace(".h5", ".txt"))