from tables import *
from datetime import datetime


# Define each rows. 
# 1. For EffectSize.
# 2. For SampleSize.
# 3. For Uniform.
# 4. For Double Thompson Sampling.
# 5. For Combiantion.
# Each table is connected by simmulation_id.

class EffectSize(IsDescription):
    simulation_id = Int32Col()
    value = Float32Col()


class SampleSize(IsDescription):
    simulation_id = Int32Col()
    value = Int32Col()


class Uniform(IsDescription):
    simulation_id = Int32Col()
    proportion_condorcet = Float32Col()


class DoubleThompsonSampling(IsDescription):
    simulation_id = Int32Col()
    proportion_condorcet = Float32Col()
    strong_regret = Float32Col()
    weak_regret = Float32Col()


class Combination(IsDescription):
    simulation_id = Int32Col()
    first_arm = Int32Col()
    second_arm = Int32Col()
    find_effect = Int32Col()
    is_DTS = Int32Col() # 1 means it's DTS, 0 means it's uniform
    test_stats = Float32Col()
    p_values = Float32Col()


class Dataset(): 
    def __init__(self):
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        self.h5file = open_file("data/dataset_" + timestampStr + ".h5", mode = "w", title = "Dueling Bandit Simulations")
        self.h5file.create_table("/", 'effect_size', EffectSize)
        self.h5file.create_table("/", 'sample_size', SampleSize)
        self.h5file.create_table("/", 'uniform', Uniform)
        self.h5file.create_table("/", 'double_thompson_sampling', DoubleThompsonSampling)
        self.h5file.create_table("/", 'combination', Combination)

    def update_effect_size_table(self, simulation_id, value):
        effect_size = self.h5file.root.effect_size.row
        effect_size['simulation_id'] = simulation_id
        effect_size['value'] = value
        effect_size.append()
        self.h5file.root.effect_size.flush()

    def update_sample_size_table(self, simulation_id, value):
        sample_size = self.h5file.root.sample_size.row
        sample_size['simulation_id'] = simulation_id
        sample_size['value'] = value
        sample_size.append()
        self.h5file.root.sample_size.flush()

    def update_uniform_table(self, simulation_id, proportion_condorcet):
        uniform = self.h5file.root.uniform.row
        uniform['simulation_id'] = simulation_id
        uniform['proportion_condorcet'] = proportion_condorcet
        uniform.append()
        self.h5file.root.uniform.flush()

    def update_double_thompson_sampling_table(self, simulation_id, proportion_condorcet, strong_regret, weak_regret):
        double_thompson_sampling = self.h5file.root.double_thompson_sampling.row
        double_thompson_sampling['simulation_id'] = simulation_id
        double_thompson_sampling['proportion_condorcet'] = proportion_condorcet
        double_thompson_sampling['strong_regret'] = strong_regret
        double_thompson_sampling['weak_regret'] = weak_regret
        double_thompson_sampling.append()
        self.h5file.root.double_thompson_sampling.flush()

    def update_combination_table(self, simulation_id, first_arm, second_arm, find_effect, is_DTS, test_stats, p_values):
        combination = self.h5file.root.combination.row
        combination['simulation_id'] = simulation_id
        combination['first_arm'] = first_arm
        combination['second_arm'] = second_arm
        combination['find_effect'] = find_effect
        combination['is_DTS'] = is_DTS
        combination['test_stats'] = test_stats
        combination['p_values'] = p_values
        combination.append()
        self.h5file.root.combination.flush()

    
if __name__ == "__main__":
    # dataset = Dataset()
    # dataset.update_effect_size_table(1, 0.5)
    # dataset.update_effect_size_table(2, 0.7)
    # dataset.update_sample_size_table(2, 7)
    # dataset.update_uniform_table(2, 0.77)

    # dataset.update_double_thompson_sampling_table(2, 0.77, 0.22, 0.123)

    # dataset.update_combination_table(2, 1, 2, 1, 1, 2, 300)

    # dataset.update_combination_table(2, 1, 2, 1, 0, 2, 500)

    # table = dataset.h5file.root.combination

    # for x in table.iterrows():
    #     print(x['p_values'])

    
    h5file = open_file("data/dataset_27-Jan-2021 (03:13:02.292022).h5", mode="r")

    table = h5file.root.combination

    combinations = [x for x in table.iterrows() if x['simulation_id'] == 1 and x['first_arm'] == 0 and x['second_arm'] == 1]

    for combination in combinations:
        print("Simulataion: " + str(combination['simulation_id']) + " Is DTS?: " + str(combination['is_DTS']) + " Frist arm: " + str(combination['first_arm']) + " Second arm: " + str(combination['second_arm']) + " Found Effect?: " + str(combination['find_effect']) + " Test Stats: " + str(combination['test_stats']) + " P Values: " + str(combination['p_values']))


