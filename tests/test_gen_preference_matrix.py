import numpy as np
import pytest

from src.gen_preference_matrix import PreferenceMatrix

class TestPreferenceMatrix:
    @classmethod
    def setup_class(self):
        self.pm = PreferenceMatrix(3)
    
    def test_reset_state(self):
        self.pm.reset_state()
        assert not np.any(self.pm.data)
        assert not np.any(self.pm.num_observations)
        assert self.pm.curr_condorcet_winner is None

    def test_set_matrix_wrong_size(self):
        wrong_matrix = np.zeros((4, 5))
        with pytest.raises(Exception):
            self.pm.set_matrix_explicit(wrong_matrix)
    
    def test_set_matrix_wrong_type(self):
        wrong_matrix = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        with pytest.raises(Exception):
            self.pm.set_matrix_explicit(wrong_matrix)

    def test_record_win(self):
        winner = 0
        loser = 1
        self.pm.record_win(winner, loser)
        assert self.pm.data[0, 1] == 1
        assert self.pm.num_observations[0] == 1
        assert self.pm.num_observations[1] == 1
    
    def test_get_condorcet_winner_empty(self):
        self.pm.reset_state()
        assert self.pm.condorcet_winner() == -1
    
    def test_get_condorcet_winner_not_exists(self):
        self.pm.reset_state()
        no_winner = np.array([[0, 2, 1],
        [1, 0, 2],
        [2, 1, 0]])
        self.pm.set_matrix_explicit(no_winner)
        assert self.pm.condorcet_winner() == -1
        
    def test_get_condorcet_winner_exists(self):
        self.pm.reset_state()
        winner = np.array([[0, 186, 405],
        [305, 0, 272],
        [78, 105, 0]]) #example is from wikipedia (Condorcet criterion)

        self.pm.set_matrix_explicit(winner)
        print(self.pm.num_observations)
        assert self.pm.condorcet_winner() == 1



    
    




