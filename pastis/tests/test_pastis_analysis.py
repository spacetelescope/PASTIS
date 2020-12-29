import os
from astropy.io import fits
import astropy.units as u
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.matrix_building_numerical import calculate_unaberrated_contrast_and_normalization
from pastis import pastis_analysis
from pastis import util


# Read the LUVOIR-A small APLC PASTIS matrix
# TODO: This needs to be a matrix I will save with Git LFS
matrix_path = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-07-17T00-57-55_luvoir-small', 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits')
#matrix_path = os.path.join('data', 'pastis_matrices', 'LUVOIR_small_matrix_piston-only.fits')

LUVOIR_MATRIX_SMALL = fits.getdata(matrix_path)
NSEG = LUVOIR_MATRIX_SMALL.shape[0]

CORO_FLOOR, NORM, LUVOIR_SIM = calculate_unaberrated_contrast_and_normalization('LUVOIR', 'small')

C_TARGET = 1e-10


def test_modes_from_matrix():
    # Calculate modes and eigenvalues from matrix
    pmodes, svals, vh = np.linalg.svd(LUVOIR_MATRIX_SMALL, full_matrices=True)

    # Check that all eigenvalues are positive
    assert (svals > 0).all(), 'SVD yields negative eigenvalues'

    # Test eigenvalue problem for a selection of modes, M u = lam u
    test_modes = [3, 45, 98]
    for mode_nb in test_modes:
        left = np.dot(LUVOIR_MATRIX_SMALL, svals[mode_nb])
        right = np.dot(LUVOIR_MATRIX_SMALL, pmodes[:, mode_nb])
        assert np.allclose(left, right)


def test_uniform_mode_weights():
    # Calculate modes and eigenvalues from matrix
    pmodes, svals, vh = np.linalg.svd(LUVOIR_MATRIX_SMALL, full_matrices=True)

    # Calculate the mode weights for uniform contrast allocation
    sigmas = pastis_analysis.calculate_sigma(C_TARGET, NSEG, svals, CORO_FLOOR)

    # Calculate cumulative contrast from the sigmas, with PASTIS propagation
    cumulative_pastis = pastis_analysis.cumulative_contrast_matrix(pmodes, sigmas, LUVOIR_MATRIX_SMALL, CORO_FLOOR)

    # Calculate cumulative contrast from the sigmas, with E2E simulator
    cumulative_e2e = pastis_analysis.cumulative_contrast_e2e('LUVOIR', pmodes, sigmas, LUVOIR_SIM, LUVOIR_SIM.dh_mask.shaped, NORM)

    # Check that final calculated contrast is equal to target contrast
    assert np.isclose(cumulative_pastis[-1], C_TARGET, rtol=1e-3, atol=1e-15), "Cumulative contrast with PASTIS doesn't end up at target contrast."
    assert np.isclose(cumulative_e2e[-1], C_TARGET, rtol=1e-2, atol=1e-15), "Cumulative contrast with E2E doesn't end up at target contrast."

    # Define a couple of testing points (mode indices)
    test_modes_first = [4, 65, 111]
    test_modes_second = [59, 109, 35]
    test_intervals = [1, 30, 20]

    # Test for equal contrast difference in equal intervals
    for first, second, interval in zip(test_modes_first, test_modes_second, test_intervals):
        assert np.isclose(cumulative_pastis[first] - cumulative_pastis[first - interval],
                          cumulative_pastis[second] - cumulative_pastis[second - interval], rtol=1e-5, atol=1e-27)
        assert np.isclose(cumulative_e2e[first] - cumulative_e2e[first - interval],
                          cumulative_e2e[second] - cumulative_e2e[second - interval], rtol=1e-1, atol=1e-15)


def test_analytical_mean_and_variance():
    # Calculate modes and eigenvalues from matrix
    pmodes, svals, vh = np.linalg.svd(LUVOIR_MATRIX_SMALL, full_matrices=True)

    # Calculate independent segment weights
    mus = pastis_analysis.calculate_segment_constraints(pmodes, LUVOIR_MATRIX_SMALL, C_TARGET, CORO_FLOOR) * u.nm

    # Calculate independent segment based covariance matrix
    Ca = np.diag(np.square(mus.value))

    # Perform basis transformation to PASTIS mode basis covariance matrix
    Cb = np.dot(np.transpose(pmodes), np.dot(Ca, pmodes))
    # Construct PASTIS matric in mode space, D
    D_matrix = np.diag(svals)

    # Calculate contrast mean and variance in both bases
    segs_mean_stat_c = util.calc_statistical_mean_contrast(LUVOIR_MATRIX_SMALL, Ca, CORO_FLOOR)
    segs_var_c = util.calc_variance_of_mean_contrast(LUVOIR_MATRIX_SMALL, Ca)
    modes_mean_stat_c = util.calc_statistical_mean_contrast(D_matrix, Cb, CORO_FLOOR)
    modes_var_c = util.calc_variance_of_mean_contrast(D_matrix, Cb)

    # Compare results in both bases to each other
    assert np.isclose(segs_mean_stat_c, modes_mean_stat_c, rtol=1e-15, atol=1e-30), 'Mean in two bases does not agree'
    assert np.isclose(segs_var_c, modes_var_c, rtol=1e-15, atol=1e-40), 'Variance in both bases does not agree'
