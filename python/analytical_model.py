import os
from astropy.io import fits

from python.config import CONFIG_INI


if __name__ == "__main__":

    #-# Define parameters
    dataDir = os.path.join('..', 'data', 'py_data')
    wvln = CONFIG_INI.getint('filter', 'lambda')
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')

    #-# Generate a dark hole
    dh_area = circle_mask(im, im.shape[1]/2., im.shape[2]/2., inner_wa)

    #-# Mean subtraction for piston

    #-# Generic segment shape

    #-# Import baseline information

    Baseline_vec = fits.getdata(os.path.join(dataDir, 'Baseline_vec.fits'))
    Projection_Matrix = fits.getdata(os.path.join(dataDir, 'Projection_Matrix.fits'))
    vec_list = fits.getdata(os.path.join(dataDir, 'vec_list.fits'))
    NR_pairs_list_int = fits.getdata(os.path.join(dataDir, 'NR_pairs_list_int.fits'))

    # Figure out how many NRPs we're dealing with
    NR_pairs_nb = NR_pairs_list_int.shape[0]

    #-# Calibration about to happen yes or no

    #-# Generic coefficients

    #-# Constant sum and cosine sum

    #-# Local Zernike

    #-# Final image