from dataclasses import dataclass
import numpy as np
from typing import List
import pdb


def init_preference_matrix(num_actions: int, num_criteria: int) -> np.ndarray:
    return np.zeros((num_actions, num_actions))


@dataclass
class PreferenceMatrix:
    num_actions: int
    num_criteria: int = 1
    data: np.ndarray = None

    def __post_init__(self):
        if self.data is None:
            self.data = init_preference_matrix(self.num_actions, self.num_criteria)
            self.curr_condorcet_winner = None
            self.num_observations = np.array([0] * self.num_actions)
            self.shape = (self.num_actions, self.num_actions)
        else:
            self.set_matrix_explicit(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

    def reset_state(self) -> None:
        self.data = np.zeros((self.num_actions, self.num_actions))
        self.curr_condorcet_winner = None
        self.num_observations = np.array([0] * self.num_actions)

    def set_matrix_explicit(self, matrix) -> None:
        if not isinstance(matrix, np.ndarray) or matrix.shape != (
            self.num_actions,
            self.num_actions,
        ):
            raise Exception(
                "Matrix must be an ndarray with shape (num_actions, num_actions)"
            )
        self.data = matrix
        self.shape = matrix.shape
        self.curr_condorcet_winner = None
        self.num_observations = np.array(
            [
                np.sum(self.data[i, :]) + np.sum(self.data[i, :]) - self.data[i, i]
                for i in range(self.num_actions)
            ]
        )

    def set_matrix_random(self) -> None:
        pm_list = [[]] * self.num_actions

        for i in range(0, self.num_actions):
            pm_list[i] = [0] * self.num_actions
            for j in range(0, self.num_actions):
                if i == j:
                    pm_list[i][j] = 0 # do not allow self-duel.
                elif i > j:
                    pm_list[i][j] = 1 - pm_list[j][i]
                else:
                    pm_list[i][j] = np.random.rand()

        matrix = np.array(pm_list)
        self.data = matrix
        self.shape = matrix.shape
        self.curr_condorcet_winner = None
        self.num_observations = np.array(
            [
                np.sum(self.data[i, :]) + np.sum(self.data[i, :]) - self.data[i, i]
                for i in range(self.num_actions)
            ]
        )

    def set_matrix_random_with_condorcet_winner(self, effect_size) -> None:
        # (0, 0) will be the condorcet winner, for each P(0, i), P(0, i) > 0.5.
        pm_list = [[]] * self.num_actions
        for i in range(0, self.num_actions):
            pm_list[i] = [0] * self.num_actions

        pm_list[0][0] = 0.5
        for j in range(1, self.num_actions):
            random_number = 0.5
            while random_number == 0.5:
                random_number = np.random.rand()
            random_number = random_number if random_number > 0.5 else 1 - random_number
            pm_list[0][j] = random_number
            pm_list[j][0] = 1 - random_number


        winner_frequency = 0.5 + (effect_size / 2)
        loser_frequency = 1 - winner_frequency
        # for other arms, we let the effect size consistent.
        for i in range(1, self.num_actions):
            for j in range(i, self.num_actions):
                if i == j:
                    pm_list[i][j] = 0.5
                    continue
                winner = np.random.choice([0,1])
                pm_list[i][j] = winner_frequency if winner == 0 else loser_frequency
                pm_list[j][i] = winner_frequency if winner == 1 else loser_frequency

        matrix = np.array(pm_list)
        self.data = matrix
        self.shape = matrix.shape
        self.curr_condorcet_winner = None
        self.num_observations = np.array(
            [
                np.sum(self.data[i, :]) + np.sum(self.data[i, :]) - self.data[i, i]
                for i in range(self.num_actions)
            ]
        )

        



    def record_win(self, winner: int, loser: int) -> None:
        if winner != self.curr_condorcet_winner:
            self.curr_condorcet_winner = None
        self.data[winner, loser] += 1
        self.num_observations[winner] += 1
        self.num_observations[loser] += 1

    def condorcet_winner(self) -> int:
        if self.curr_condorcet_winner == None:
            for i in range(self.num_actions):
                for j in range(self.num_actions):
                    if i == j:
                        continue
                    if self.data[j, i] >= self.data[i, j]:
                        break
                    if j == self.num_actions - 1 or (
                        j == self.num_actions - 2 and i == self.num_actions - 1
                    ):
                        self.curr_condorcet_winner = i
                        return i
            return -1

        else:
            return self.curr_condorcet_winner

    def borda_winner(self) -> np.ndarray:
        return np.argwhere(
            np.sum(self.data, axis=1) == np.amax(np.sum(self.data, axis=1))
        )

    def borda_score(self, action: int) -> float:
        return np.sum(self.data, axis=1)[action]


if __name__ == "__main__":
    matrix = PreferenceMatrix(num_actions=2, num_criteria=1)
    print(matrix.num_observations)
