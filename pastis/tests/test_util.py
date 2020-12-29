import astropy.units as u
import numpy as np
from pastis import util


def test_get_segment_list():
    # Check that the first and last segment of the segment list is correct.

    seglist_luvoir = util.get_segment_list('LUVOIR')
    assert seglist_luvoir[0] == 1, 'First LUVOIR segment index is not 1, but should be.'
    assert seglist_luvoir[-1] == 120, 'Last LUVOIR segment index is not 120, but should be.'

    seglist_hicat = util.get_segment_list('HiCAT')
    assert seglist_hicat[0] == 0, 'First HiCAT segment index is not 0, but should be.'
    assert seglist_hicat[-1] == 36, 'Last HiCAT segment index is not 36, but should be.'

    seglist_jwst = util.get_segment_list('JWST')
    assert seglist_jwst[0] == 0, 'First JWST segment index is not 0, but should be.'
    assert seglist_jwst[-1] == 17, 'Last JWST segment index is not 17, but should be.'


def test_segment_pairs_all():
    # Check that the permutation list has the correct length.

    nseg = 120
    permutation_list = list(util.segment_pairs_all(nseg))
    assert len(permutation_list) == nseg * nseg, 'Total number of repeating permutations is not correct.'


def test_segment_pairs_non_repeating():
    # Check that the permutation list has the correct length.

    nseg = 120
    permutation_list = list(util.segment_pairs_non_repeating(nseg))
    number_of_measurements = util.pastis_matrix_measurements(nseg)

    assert len(permutation_list) == number_of_measurements, 'Total number of non-repeating permutations is not correct.'


def test_symmetrize():
    # Check that a matrix gets symmetrized correctly.

    matrix = np.array([[1,0,0], [4,5,0], [7,8,9]])
    symmetric_matrix = util.symmetrize(matrix)

    assert symmetric_matrix[1,2] == symmetric_matrix[2,1], 'Matrix not correctly symmetrized.'
    assert (symmetric_matrix == symmetric_matrix.T).all(), 'Matrix not correctly symmetrized.'


def test_rms():
    # Check that the rms calculation is correct.

    values = np.array([0.24, 0.76, 3.5])
    rms_manual = np.sqrt(np.mean(np.square(values)) - np.square(np.mean(values)))
    assert util.rms(values) == rms_manual, 'Calculated rms value does not check out.'


def test_create_random_rms_values():
    # Check that the random WFE map is correctly scaled to the target global WFE.

    nseg = 120
    target_rms = 1 * u.nm
    random_array = util.create_random_rms_values(nseg, target_rms)

    resulting_rms = util.rms(random_array)
    assert resulting_rms.unit == target_rms.unit, 'The resulting total rms has wrong units.'
    assert np.isclose(resulting_rms, target_rms, 1e-13), 'Calculated total rms does not agree with target rms value.'
