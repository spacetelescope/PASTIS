import os
from astropy.io import fits
import astropy.units as u
import numpy as np

import pastis.matrix_building_numerical as matrix_calc
from pastis import util


# Read the LUVOIR-A small APLC PASTIS matrix
# From data dir: 2021-01-09T01-01-53_luvoir-small
# Created on commit: 0842c7
test_data_dir = os.path.join(util.find_package_location(), 'tests')
matrix_path = os.path.join(test_data_dir, 'data', 'pastis_matrices', 'LUVOIR_small_matrix_piston-only.fits')
LUVOIR_MATRIX_SMALL = fits.getdata(matrix_path)
NSEG = LUVOIR_MATRIX_SMALL.shape[0]

# Read the LUVOIR-A small APLC contrast matrix
# From data dir: 2021-01-09T01-01-53_luvoir-small
# Created on commit: 0842c7
# Coronagraph floor has not been subtracted from this contrast matrix
contrast_matrix_path = os.path.join(test_data_dir, 'data', 'pastis_matrices',
                                    'contrast_matrix_LUVOIR_small_piston-only.fits')
CONTRAST_MATRIX = fits.getdata(contrast_matrix_path)


def test_semi_analytic_matrix_from_contrast_matrix():
    """ Test that the analytical calculation of the semi-analytical PASTIS matrix calculation is correct. """

    # Create seglist and drop in calibration aberration the matrices in tests/data have been created with
    seglist = util.get_segment_list('LUVOIR')
    wfe_aber = 1e-9    # m

    ### Test the case in which the coronagraph floor is constant across all matrix measurements

    # Hard-code the contrast floor the matrix was generated with
    coro_floor = 4.315823935036038e-11
    # Calculate the PASTIS matrix under assumption of a CONSTANT coronagraph floor
    pastis_matrix_constant = matrix_calc.pastis_from_contrast_matrix(CONTRAST_MATRIX, seglist, wfe_aber, coro_floor)
    # Compare to PASTIS matrix in the tests folder
    assert np.allclose(pastis_matrix_constant, LUVOIR_MATRIX_SMALL, rtol=1e-8, atol=1e-24), 'Calculated LUVOIR small PASTIS matrix is wrong.'

    ### Test the case in which the coronagraph floor is drifting across matrix measurements

    # Construct random coro floor matrix
    coro_floor_state = np.random.RandomState()
    coro_floor_matrix_full = coro_floor_state.normal(loc=0, scale=1, size=(NSEG, NSEG)) * coro_floor
    random_coro_floor_matrix = np.triu(coro_floor_matrix_full)

    # Subtract the original contrast floor from the contrast matrix
    constant_coro_floor_matrix = np.ones((NSEG, NSEG)) * coro_floor
    contrast_matrix_subtracted = CONTRAST_MATRIX - np.triu(constant_coro_floor_matrix)

    # Add the random c0 matrix to subtracted contrast matrix
    contrast_matrix_random_c0 = contrast_matrix_subtracted + random_coro_floor_matrix

    # Calculate the PASTIS matrix under assumption of a DRIFTING coronagraph floor
    pastis_matrix_drift = matrix_calc.pastis_from_contrast_matrix(contrast_matrix_random_c0, seglist, wfe_aber, random_coro_floor_matrix)
    # Compare to PASTIS matrix in the tests folder
    assert np.allclose(pastis_matrix_drift, LUVOIR_MATRIX_SMALL, rtol=1e-8, atol=1e-24), 'Calculated LUVOIR small PASTIS matrix is wrong.'


def test_pastis_forward_model():
    """ Test that the PASTIS matrix propagates aberrations correctly
    This is essentially a test for the hockey stick curve, inside its valid range. """

    # Define a couple of total pupil rms values of WFE to test
    rms_values = [1, 10, 15] * u.nm    # nm WFE rms over total  pupil
    relative_tolerances = [1e-3, 1e-2, 1e-1]
    absolute_tolerances = [1e-15, 1e-9, 1e-8]

    # Calculate coronagraph floor, direct PSF peak normalization factor, and return E2E sim instance
    contrast_floor, norm, luvoir_sim = matrix_calc.calculate_unaberrated_contrast_and_normalization('LUVOIR', 'small')

    for rms, rel_tol, abs_tol in zip(rms_values, relative_tolerances, absolute_tolerances):
        # Create random aberration coefficients on segments, scaled to total rms
        aber = util.create_random_rms_values(NSEG, rms)

        # Contrast from PASTIS propagation
        contrasts_matrix = (util.pastis_contrast(aber, LUVOIR_MATRIX_SMALL) + contrast_floor)

        # Contrast from E2E propagator
        for nb_seg in range(NSEG):
            luvoir_sim.set_segment(nb_seg + 1, aber[nb_seg].to(u.m).value / 2, 0, 0)
        psf_luvoir = luvoir_sim.calc_psf()
        psf_luvoir /= norm
        contrasts_e2e = (util.dh_mean(psf_luvoir, luvoir_sim.dh_mask))

        assert np.isclose(contrasts_matrix, contrasts_e2e, rtol=rel_tol, atol=abs_tol), f'Calculated contrasts from PASTIS and E2E are not the same for rms={rms} and rtol={rel_tol}.'


def test_luvoir_matrix_regression():
    """ Check multiprocessed matrix calculation against previously calculated matrix """

    # Calculate new LUVOIR small PASTIS matrix
    new_matrix_path = matrix_calc.num_matrix_multiprocess(instrument='LUVOIR', design='small', savepsfs=False, saveopds=False)
    new_matrix = fits.getdata(os.path.join(new_matrix_path, 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits'))

    # Check that the calculated PASTIS matrix is symmetric
    assert (new_matrix == new_matrix.T).all(), 'Calculated LUVOIR small PASTIS matrix is not symmetric'

    # Check that new matrix is equal to previously computed matrix that is known to be correct, down to numerical noise
    # on the order of 1e-23
    assert np.allclose(new_matrix, LUVOIR_MATRIX_SMALL, rtol=1e-8, atol=1e-24), 'Calculated LUVOIR small PASTIS matrix is wrong.'
