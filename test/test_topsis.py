import numpy as np
import pytest
import sys
sys.path.append("../src")
from decision_making import TOPSIS


dec_mat_2 = [
    [15, 6, 25000, 7],
    [12, 7, 35000, 7],
    [10, 9, 55000, 8]
]
dec_norm_w = [
    [0.20779069, 0.02328452, 0.21483446, 0.0274986],
    [0.16623255, 0.02716527, 0.30076825, 0.0274986],
    [0.13852713, 0.03492677, 0.47263582, 0.03142697]
]
alternatives = ["Alt 1", "Alt 2", "Alt 3"]
criteria = ["criterion 1", "criterion 2", "criterion 3", "criterion 4"]
weights = [0.3, 0.05, 0.6, 0.05]
cost_ben = ["c", "b", "c", "b"]

ideal_pos = [0.13852713, 0.03492677, 0.21483446, 0.03142697]
ideal_neg = [0.20779069, 0.02328452, 0.47263582, 0.0274986]
dist_pos = [0.07034498, 0.09070766, 0.25780135]
dist_neg = [0.25780135, 0.17686323, 0.07034498]
clos_coefficient = [0.78562925, 0.66099579, 0.21437075]


def is_equal(a, b):
    return all(a[i][j] == b[i][j] for i in range(len(a)) for j in range(len(a[0])))


def test_input_from_csv_file_normal_flow():
    tp = TOPSIS("dec_mat_2.csv", cost_ben, weights=weights, crit_col_names=criteria, alt_col_name="alternative")
    assert np.allclose(tp.matrix_d, dec_norm_w)


def test_input_from_csv_file_alternative():
    tp = TOPSIS("dec_mat_2.csv", cost_ben,weights=weights, crit_col_names=criteria, alt_col_name="alternative")
    assert is_equal(list(tp.alternatives), alternatives)


def test_input_from_csv_file_criteria():
    tp = TOPSIS("dec_mat_2.csv", cost_ben, crit_col_names=criteria, alt_col_name="alternative")
    assert is_equal(list(tp.criteria), criteria)


def test_cost_ben_str():
    tp = TOPSIS("dec_mat_2.csv", "cost", crit_col_names=criteria, alt_col_name="alternative")
    assert is_equal(list(tp.cost_ben), ["cost"]*4)


def test_input_from_list():
    tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
    assert np.allclose(tp.matrix_d, dec_norm_w)


def test_input_from_numpy():
    tp = TOPSIS(np.array(dec_mat_2), cost_ben, weights=weights)
    assert np.allclose(tp.matrix_d, dec_norm_w)


def test_input_invalid():
    with pytest.raises(ValueError):
        tp = TOPSIS(10, cost_ben)


def test_num_crit_alter():
    tp = TOPSIS(dec_mat_2, cost_ben)
    assert tp.n_alt == 3
    assert tp.n_crit == 4


def test_weights_none():
    tp = TOPSIS(dec_mat_2, cost_ben)
    assert tp.n_crit == len(tp.weights)
    assert np.isclose(tp.weights.sum(), 1.0)


def test_weights_input():
    tp = TOPSIS(dec_mat_2, cost_ben, weights=[1, 2, 3, 4])
    assert np.isclose(tp.weights.sum(), 1.0)
    with pytest.raises(ValueError):
        _ = TOPSIS(dec_mat_2, cost_ben, weights=[1, 2, 3])
    with pytest.raises(ValueError):
        _ = TOPSIS(dec_mat_2, cost_ben, weights=10)


def test_ideal_solutions():
    tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
    tp.get_ideal_solutions()
    assert np.allclose(tp.ideal_neg, ideal_neg)
    assert np.allclose(ideal_pos, ideal_pos)


def test_ideal_distances():
    tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
    tp.get_distance_to_ideal()
    assert np.allclose(tp.dist_neg, dist_neg)
    assert np.allclose(tp.dist_pos, dist_pos)


def test_clos_coefficient():
    tp = TOPSIS(dec_mat_2, cost_ben, weights=weights)
    tp.get_closeness_coefficient()
    assert np.allclose(tp.clos_coefficient, clos_coefficient)