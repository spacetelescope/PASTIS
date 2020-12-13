import os
from astropy.io import fits
import numpy as np
from pastis.config import CONFIG_PASTIS


# Read the LUVOIR-A small APLC PASTIS matrix
# This needs to be replaced by a matrix calculated during testing
matrix_path = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-07-17T00-57-55_luvoir-small', 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits')

LUVOIR_MATRIX_SMALL = fits.getdata(matrix_path)
NSEG = LUVOIR_MATRIX_SMALL.shape[0]


def test_symmetry():
    # Check that PASTIS matrix is symmetric
    assert np.all(LUVOIR_MATRIX_SMALL) == np.all(LUVOIR_MATRIX_SMALL.T)


def test_pastis_forward_model():
    # Test that the PASTIS matrix propagates aberrations correctly
    # This is essentially a test for the hockey stick curve, inside its valid range
    pass


def test_matrix_regression():
    # Check that new matrix is equal to previously computed matrix that is known to be correct
    pass
