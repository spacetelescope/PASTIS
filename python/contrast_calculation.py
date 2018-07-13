"""This program compares contrast calculation from different methods."""

import os
import time
import numpy as np
from astropy.io import fits

from python.config import CONFIG_INI
import python.util_pastis as util
import python.analytical_model as am
import python.webbpsf_imaging as webbim


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around ? minutes

    # Parameters
    dataDir = CONFIG_INI.get('local', 'local_data_path')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    filter = CONFIG_INI.get('filter', 'name')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')         # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    # Import PASTIS matrix
    filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    matrix_pastis = fits.getdata(os.path.join(dataDir, 'results', filename + '.fits'))

    # Create random aberration coefficients
    if zern_number == 1:   # piston
        Aber = np.random.random([nb_seg]) * 10   # piston values

    # Mean subtraction for piston
    if zern_number == 1:
        Aber -= np.mean(Aber)

    # Make equivalent aberration array that goes into the WebbPSF function
    Aber_WSS = np.zeros([nb_seg, zern_max])
    Aber_WSS[:,0] = Aber   # index "0" works because we're using piston currently

    ### WEBBPSF
    print('Generating WebbPSF contrast')
    start_webb = time.time()
    # Set up NIRCam and coronagraph
    psf_webbpsf = webbim.nircam_coro(filter, fpm, lyot_stop, Aber_WSS)
    # Get the mean conrast from the WebbPSF coronagraph
    contrast_webbpsf = np.mean(psf_webbpsf[np.where(psf_webbpsf != 0)])
    end_webb = time.time()

    ### IMAGE PASTIS
    print('Generating contrast from image-PASTIS')
    start_impastis = time.time()
    # Create calibrated image from analytical model
    psf_am = am.analytical_model(zern_number, Aber, cali=True)
    # Get the mean contrast from image PASTIS
    contrast_am = np.mean(psf_am[np.where(psf_am != 0)])
    end_impastis = time.time()

    # Load in baseline contast
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_base = float(np.loadtxt(os.path.join(dataDir, 'calibration', contrastname+'.txt')))

    ### MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(Aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ratio = contrast_am / contrast_matrix

    print('\n--- CONTRASTS: ---')
    print('Mean contrast from WebbPSF:', contrast_webbpsf)
    print('Mean contrast with image PASTIS:', contrast_am)
    print('Contrast from matrix PASTIS:', contrast_matrix)
    print('Ratio image PASTIS / matrix PASTIS:', ratio)

    print('\n--- RUNTIMES: ---')
    print('WebbPSF: ', end_webb-start_webb, 'sec =', (end_webb-start_webb)/60, 'min')
    print('Image PASTIS: ', end_impastis-start_impastis, 'sec =', (end_impastis-start_impastis)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    print('Runtime for contrast_calculation.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
