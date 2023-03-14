import os
from astropy.io import fits
import astropy.units as u
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_building_numerical import pastis_from_contrast_matrix
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex, pastis_matrix_from_efields
from pastis.simulators.scda_telescopes import HexRingAPLC
from pastis import util


# Read the LUVOIR-A small APLC PASTIS matrix created from intensities
# From data dir: 2021-03-21T19-15-50_luvoir-small
# Created on commit: 33edfa9265a4a07844927402cbde97761c413fcd
test_data_dir = os.path.join(util.find_package_location(), 'tests')
matrix_path = os.path.join(test_data_dir, 'data', 'pastis_matrices', 'LUVOIR_small_matrix_piston-only.fits')
LUVOIR_INTENSITY_MATRIX_SMALL = fits.getdata(matrix_path)
NSEG_LUVOIR = LUVOIR_INTENSITY_MATRIX_SMALL.shape[0]

# Read the LUVOIR-A small APLC contrast matrix
# From data dir: 2021-03-21T19-15-50_luvoir-small
# Created on commit: 33edfa9265a4a07844927402cbde97761c413fcd
# Coronagraph floor has not been subtracted from this contrast matrix
contrast_matrix_path = os.path.join(test_data_dir, 'data', 'pastis_matrices',
                                    'contrast_matrix_LUVOIR_small_piston-only.fits')
CONTRAST_MATRIX = fits.getdata(contrast_matrix_path)

# Read the 2-Hex simulator E-fields
# From data dir: 2023-03-14T17-10-12_hexringtelescope
# Created on commit: 1771fc536b120fb862ff700917069e36135b20a5
efields_path = os.path.join(test_data_dir, 'data', 'pastis_matrices')
HEX2_E_FIELDS_REAL = fits.getdata(os.path.join(efields_path, 'Hex2_efield_coron_real.fits'))
HEX2_E_FIELDS_IMAG = fits.getdata(os.path.join(efields_path, 'Hex2_efield_coron_imag.fits'))
NUM_RINGS = 2

# Read the 2-Hex matrix created from E-fields
# From data dir: 2023-03-14T17-10-12_hexringtelescope
# Created on commit: 1771fc536b120fb862ff700917069e36135b20a5
test_data_dir = os.path.join(util.find_package_location(), 'tests')
matrix_path2 = os.path.join(test_data_dir, 'data', 'pastis_matrices', 'Hex2_matrix_piston-only.fits')
HEX2_INTENSITY_MATRIX = fits.getdata(matrix_path2)
NSEG_HEX2 = 3 * NUM_RINGS * (NUM_RINGS + 1) + 1

# Set up 2-Hex simulator instance and its basic properties
optics_input = os.path.join(util.find_repo_location(), 'data', 'SCDA')
sampling = CONFIG_PASTIS.getfloat('HexRingTelescope', 'sampling')
hex2 = HexRingAPLC(optics_input, NUM_RINGS, sampling)

unaberrated_coro_psf, direct = hex2.calc_psf(ref=True, norm_one_photon=True)
NORM = np.max(direct)
EFIELD_REF, _inter = hex2.calc_psf(return_intermediate='efield', norm_one_photon=True)


def test_semi_analytic_matrix_from_contrast_matrix():
    """Test that the analytical calculation of the semi-analytical PASTIS matrix calculation from intensity images is correct."""

    # Create seglist and drop in calibration aberration the matrices in tests/data have been created with
    seglist = util.get_segment_list('LUVOIR')
    wfe_aber = 1e-9    # m

    ### Test the case in which the coronagraph floor is constant across all matrix measurements

    # Hard-code the contrast floor the matrix was generated with (LUVOIR small)
    coro_floor = 4.2376360700565846e-11
    # Calculate the PASTIS matrix under assumption of a CONSTANT coronagraph floor
    pastis_matrix_constant = pastis_from_contrast_matrix(CONTRAST_MATRIX, seglist, wfe_aber, coro_floor)
    # Compare to PASTIS matrix in the tests folder
    assert np.allclose(pastis_matrix_constant, LUVOIR_INTENSITY_MATRIX_SMALL, rtol=1e-8, atol=1e-24), 'Calculated LUVOIR small PASTIS matrix is wrong.'

    ### Test the case in which the coronagraph floor is drifting across matrix measurements

    # Construct random coro floor matrix
    coro_floor_state = np.random.RandomState()
    coro_floor_matrix_full = coro_floor_state.normal(loc=0, scale=1, size=(NSEG_LUVOIR, NSEG_LUVOIR)) * coro_floor
    random_coro_floor_matrix = np.triu(coro_floor_matrix_full)

    # Subtract the original contrast floor from the contrast matrix
    constant_coro_floor_matrix = np.ones((NSEG_LUVOIR, NSEG_LUVOIR)) * coro_floor
    contrast_matrix_subtracted = CONTRAST_MATRIX - np.triu(constant_coro_floor_matrix)

    # Add the random c0 matrix to subtracted contrast matrix
    contrast_matrix_random_c0 = contrast_matrix_subtracted + random_coro_floor_matrix

    # Calculate the PASTIS matrix under assumption of a DRIFTING coronagraph floor
    pastis_matrix_drift = pastis_from_contrast_matrix(contrast_matrix_random_c0, seglist, wfe_aber, random_coro_floor_matrix)
    # Compare to PASTIS matrix in the tests folder
    assert np.allclose(pastis_matrix_drift, LUVOIR_INTENSITY_MATRIX_SMALL, rtol=1e-8, atol=1e-24), 'Calculated LUVOIR small PASTIS matrix is wrong.'


def test_semi_analytic_matrix_from_efields():
    """Test that the analytical calculation of the semi-analytical PASTIS matrix calculation from electric fields is correct."""
    wfe_aber = 1e-9    # m

    # Recombine complex E-fields
    E_FIELDS = np.array([real.ravel() + 1j * imag.ravel() for real, imag in zip(HEX2_E_FIELDS_REAL, HEX2_E_FIELDS_IMAG)])

    # Calculate the PASTIS matrix under assumption of a CONSTANT coronagraph floor
    pastis_matrix_constant = pastis_matrix_from_efields(E_FIELDS, EFIELD_REF.electric_field, NORM, hex2.dh_mask, wfe_aber)
    # Compare to PASTIS matrix in the tests folder
    assert np.allclose(pastis_matrix_constant, HEX2_INTENSITY_MATRIX, rtol=1e-8, atol=1e-24), 'Calculated 2-Hex PASTIS matrix is wrong.'


def test_pastis_forward_model():
    """Test that the PASTIS matrix propagates aberrations correctly
    This is essentially a test for the hockey stick curve, inside its valid range."""

    # Define a couple of total pupil rms values of WFE to test
    rms_values = [1, 10, 15] * u.nm    # nm WFE rms over total  pupil
    relative_tolerances = [1e-3, 1e-2, 1e-1]
    absolute_tolerances = [1e-15, 1e-9, 1e-8]

    # Known from aberration-free image (2-Hex simulator)
    contrast_floor = 4.1713373582172286e-11

    for rms, rel_tol, abs_tol in zip(rms_values, relative_tolerances, absolute_tolerances):
        # Create random aberration coefficients on segments, scaled to total rms
        aber = util.create_random_rms_values(NSEG_HEX2, rms)

        # Contrast from PASTIS propagation
        contrasts_matrix = (util.pastis_contrast(aber, HEX2_INTENSITY_MATRIX) + contrast_floor)

        # Contrast from E2E propagator
        for nb_seg in range(NSEG_HEX2):
            hex2.set_segment(nb_seg, aber[nb_seg].to(u.m).value / 2, 0, 0)
        psf_2hex = hex2.calc_psf(norm_one_photon=True)
        psf_2hex /= NORM
        contrasts_e2e = util.dh_mean(psf_2hex, hex2.dh_mask)

        assert np.isclose(contrasts_matrix, contrasts_e2e, rtol=rel_tol, atol=abs_tol), f'Calculated contrasts from PASTIS and E2E are not the same for rms={rms} and rtol={rel_tol}.'


def test_2hex_efields_matrix_regression(tmpdir):
    """Check matrix calculation against previously calculated matrix"""

    # Calculate new 2-Hex PASTIS matrix from E-fields
    WHICH_DM = 'seg_mirror'
    DM_SPEC = 1

    new_matrix_calc = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                      calc_science=True, calc_wfs=False,
                                      initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
                                      norm_one_photon=True)
    new_matrix_calc.calc()
    new_matrix = new_matrix_calc.matrix_pastis

    # Check that the calculated PASTIS matrix is symmetric
    assert (new_matrix == new_matrix.T).all(), 'Calculated 2-Hex PASTIS matrix is not symmetric'

    # Check that new matrix is equal to previously computed matrix that is known to be correct
    assert np.allclose(new_matrix, HEX2_INTENSITY_MATRIX, rtol=1e-24, atol=1e-24), 'Calculated 2-Hex PASTIS matrix is wrong.'
