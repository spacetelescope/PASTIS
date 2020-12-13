import numpy as np
from pastis import util


def test_get_segment_list():
    # Check that the first and last segment of the segment list is correct.

    seglist_luvoir = util.get_segment_list('LUVOIR')
    assert seglist_luvoir[0] == 1
    assert seglist_luvoir[-1] == 120

    seglist_hicat = util.get_segment_list('HiCAT')
    assert seglist_hicat[0] == 0
    assert seglist_hicat[-1] == 36

    seglist_jwst = util.get_segment_list('JWST')
    assert seglist_jwst[0] == 0
    assert seglist_jwst[-1] == 17


def test_segment_pairs_all():
    # Check that the permutation list has the correct length.

    nseg = 120
    permutation_list = list(util.segment_pairs_all(nseg))
    assert len(permutation_list) == nseg * nseg


def test_segment_pairs_non_repeating():
    # Check that the permutation list has the correct length.

    nseg = 120
    permutation_list = list(util.segment_pairs_non_repeating(nseg))
    number_of_measurements = util.pastis_matrix_measurements(nseg)

    assert len(permutation_list) == number_of_measurements


def test_symmetrize():
    # Check that a matrix gets symmetrized correctly.

    matrix = np.array([[1,0,0], [4,5,0], [7,8,9]])
    symmetric_matrix = util.symmetrize(matrix)

    assert symmetric_matrix[1,2] == symmetric_matrix[2,1]
    assert np.all(symmetric_matrix) == np.all(symmetric_matrix.T)


def test_rms():
    # Check that the rms calculation is correct.

    values = np.array([0.24, 0.76, 3.5])
    rms_manual = np.sqrt(np.mean(np.square(values)) - np.square(np.mean(values)))
    assert util.rms(values) == rms_manual


def test_create_random_rms_values():
    # Check that the random WFE map is correctly scaled to the target global WFE.

    nseg = 120
    total_rms = 1
    random_array = util.create_random_rms_values(nseg, total_rms)

    assert util.rms(random_array) == total_rms
