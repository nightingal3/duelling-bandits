import numpy as np
from gen_preference_matrix import PreferenceMatrix

# directly from Appendix D.1.3 of Wu and Liu 2016

arr_condorcet = np.array([
[0.500, 0.525, 0.613, 0.757, 0.765],
[0.465, 0.500, 0.580, 0.727, 0.738],
[0.387, 0.420, 0.500, 0.659, 0.669],
[0.243, 0.273, 0.341, 0.500 0.510],
[0.235, 0.262, 0.331, 0.490, 0.500]])

condorcet_ltr = PreferenceMatrix(num_actions=5, data=arr_condorcet)


arr_non_condorcet = np.array([
[0.500, 0.484, 0.519, 0.529, 0.518],
[0.516, 0.500, 0.481, 0.530, 0.539],
[0.481, 0.519, 0.500, 0.504, 0.512],
[0.471, 0.470, 0.496, 0.500, 0.503],
[0.482, 0.461, 0.488, 0.497, 0.500]])

non_condorcet_ltr = PreferenceMatrix(num_actions=5, data=arr_non_condorcet)

# data provided by Zoghi, Karnin, Whiteson and Rijke
data_path = "data/MSLR_Perfect_PMat.npz"

def get_rankers(N: int = 5) -> PreferenceMatrix:
    full_matrix = np.load(data_path)["PMat"]
    inds = np.random.choice(full_matrix.shape[0], N)
    submatrix = full_matrix[randInds][:, randInds]

    return PreferenceMatrix(num_actions=N, data=submatrix)

