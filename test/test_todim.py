import numpy as np
import pytest
import sys
sys.path.append("../src")
from decision_making import TODIM

dec_mat_1 = [
    [8.627, 5.223],
    [9.838, 4.023],
    [10.374, 3.495],
    [8.200, 5.659],
    [5.854, 7.989],
    [8.108, 5.790],
    [6.845, 7.083],
    [5.738, 8.238],
    [5.858, 8.189],
    [6.269, 7.808]
]
alternatives_1 = ["Alg 1", "Alg 2", "Alg 3", "Alg 4", "Alg 5", "Alg 6", "Alg 7", "Alg 8", "Alg 9", "Alg 10"]
criteria_1 = ["criterion 1", "criterion 2"]
delta = [
    [0., 0.45169534],
    [0.47991535, 0.],
    [0.57675307, 0.31988437],
    [0.27509041, 0.52885644],
    [0.68734689, 0.82247955],
    [0.32078588, 0.55385392],
    [0.57240075, 0.7290908 ],
    [0.72872817, 0.85729724],
    [0.7290156, 0.85745864],
    [0.68570151, 0.82085436]
]

closs_coeff = [
     [0.        ],
     [0.02486829],
     [0.39209586],
     [0.31041422],
     [0.93245574],
     [0.3727109 ],
     [0.74886502],
     [0.99960448],
     [1.        ],
     [0.92957362],
]


def is_equal(a, b):
    return all(a[i][j] == b[i][j] for i in range(len(a)) for j in range(len(a[0])))


def test_input_from_csv_file_normal_flow():
    td = TODIM("dec_mat_1.csv", crit_col_names=["criterion 1", "criterion 2"], alt_col_name="alternative")
    assert (td.matrix_d.all() == np.asarray(dec_mat_1).all())


def test_input_from_csv_file_alternative():
    td = TODIM("dec_mat_1.csv", crit_col_names=["criterion 1", "criterion 2"], alt_col_name="alternative")
    assert is_equal(list(td.alternatives), alternatives_1)


def test_input_from_csv_file_criteria():
    td = TODIM("dec_mat_1.csv", crit_col_names=["criterion 1", "criterion 2"], alt_col_name="alternative")
    assert is_equal(list(td.criteria), criteria_1)


def test_input_from_list():
    td = TODIM(dec_mat_1)
    assert is_equal(list(td.matrix_d), dec_mat_1)


def test_input_from_numpy():
    td = TODIM(np.array(dec_mat_1))
    assert is_equal(list(td.matrix_d), dec_mat_1)


def test_input_invalid():
    with pytest.raises(ValueError):
        td = TODIM(10)


def test_num_crit_alter():
    td = TODIM(dec_mat_1)
    assert td.n_alt == 10
    assert td.n_crit == 2


def test_weights_none():
    td = TODIM(dec_mat_1)
    assert td.n_crit == len(td.weights)
    assert np.isclose(td.weights.sum(), 1.0)


def test_weights_input():
    td2 = TODIM(dec_mat_1, weights=[1, 2])
    assert np.isclose(td2.weights.sum(), 1.0)
    with pytest.raises(ValueError):
        td3 = TODIM(dec_mat_1, weights=[1, 2, 3])
    with pytest.raises(ValueError):
        td4 = TODIM(dec_mat_1, weights=10)


def test_delta():
    td = TODIM(dec_mat_1, weights=[0.5107, 0.4893], theta=2.5)
    td.get_delta()
    assert td.delta.all() == np.asarray(delta).all()


def test_closs_coefficient():
    td = TODIM(dec_mat_1, weights=[0.5107, 0.4893], theta=2.5)
    td.get_closeness_coefficient()
    assert td.clos_coefficient.all() == np.asarray(closs_coeff).all()

