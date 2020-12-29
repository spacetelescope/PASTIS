import os
from astropy.io import fits
import astropy.units as u
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.matrix_building_numerical import calculate_unaberrated_contrast_and_normalization, num_matrix_multiprocess
from pastis import util


# Read the LUVOIR-A small APLC PASTIS matrix
# TODO: This needs to be a matrix I will save with Git LFS
matrix_path = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-07-17T00-57-55_luvoir-small', 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits')
#matrix_path = os.path.join('data', 'pastis_matrices', 'LUVOIR_small_matrix_piston-only.fits')

LUVOIR_MATRIX_SMALL = fits.getdata(matrix_path)
NSEG = LUVOIR_MATRIX_SMALL.shape[0]


def test_luvoir_matrix_regression():
    # Check multiprocessed matrix calculation against previously calculated matrix
    new_matrix_path = num_matrix_multiprocess(instrument='LUVOIR', design='small', savepsfs=False, saveopds=False)
    new_matrix = fits.getdata(os.path.join(new_matrix_path, 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits'))

    # Check that the calculated PASTIS matrix is symmetric
    assert (new_matrix == new_matrix.T).all(), 'Calculated LUVOIR small PASTIS matrix is not symmetric'

    # Check that new matrix is equal to previously computed matrix that is known to be correct
    assert (new_matrix == LUVOIR_MATRIX_SMALL).all(), 'Calculated LUVOIR small PASTIS matrix is wrong.'


def test_pastis_forward_model():
    # Test that the PASTIS matrix propagates aberrations correctly
    # This is essentially a test for the hockey stick curve, inside its valid range
    rms_values = [1, 10, 15] * u.nm    # nm WFE rms over total  pupil
    relative_tolerances = [1e-3, 1e-2, 1e-1]
    absolute_tolerances = [1e-15, 1e-9, 1e-8]

    # Calculate coronagraph floor, direct PSF peak normalization factor, and return E2E sim instance
    contrast_floor, norm, luvoir_sim = calculate_unaberrated_contrast_and_normalization('LUVOIR', 'small')

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
