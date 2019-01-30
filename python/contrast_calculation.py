"""This program compares contrast calculation from different methods:
WebbPSF coronagraph
Image-based PASTIS
Matrix-based PASTIS"""

import os
import time
import numpy as np
from astropy.io import fits
import webbpsf

from python.config import CONFIG_INI
import python.util_pastis as util
import python.image_pastis as impastis
import python.webbpsf_imaging as webbim


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Do you want to use the analytical or the numerical matrix
    matrix_mode = 'analytical'    # 'analytical' or 'numerical'

    # Parameters
    dataDir = CONFIG_INI.get('local', 'local_data_path')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    filter = CONFIG_INI.get('filter', 'name')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')         # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    #real_samp = sampling * tel_size_px / im_size
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    # Import PASTIS matrix
    if matrix_mode == 'analytical':
        filename = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_numerical', filename + '.fits'))
    elif matrix_mode == 'numerical':
        filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_analytical', filename + '.fits'))

    # Create random aberration coefficients
    if zern_number == 1:   # piston
        Aber = np.random.random([nb_seg]) * 100   # piston values
        print('PISTON ABERRATIONS:', Aber)

    # Mean subtraction for piston   - we already have this in image_pastis.py
    #if zern_number == 1:
    #    Aber -= np.mean(Aber)

    # Make equivalent aberration array that goes into the WebbPSF function
    Aber_WSS = np.zeros([nb_seg, zern_max])
    Aber_WSS[:,0] = Aber / 1e9   # index "0" works because we're using piston currently; convert to meters

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    print('Generating baseline PSF from WebbPSF - no coronagraph, no aberrations')
    psf_perfect = webbim.nircam_nocoro(filter, np.zeros_like(Aber_WSS))
    normp = np.max(psf_perfect)
    psf_perfect = psf_perfect / normp

    ### WEBBPSF
    print('Generating WebbPSF coro contrast')
    start_webb = time.time()
    # Set up NIRCam and coronagraph, get PSF
    psf_webbpsf = webbim.nircam_coro(filter, fpm, lyot_stop, Aber_WSS)
    psf_webbpsf = psf_webbpsf / normp
    # Create dark hole
    dh_area = util.create_dark_hole(psf_webbpsf, inner_wa, outer_wa, sampling)
    # Get the mean conrast from the WebbPSF coronagraph
    webb_dh_psf = psf_webbpsf * dh_area
    contrast_webbpsf = np.mean(webb_dh_psf[np.where(webb_dh_psf != 0)])
    end_webb = time.time()

    # Load in baseline contrast
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_base = float(np.loadtxt(os.path.join(dataDir, 'calibration', contrastname+'.txt')))

    ### IMAGE PASTIS
    print('Generating contrast from image-PASTIS')
    start_impastis = time.time()
    # Create calibrated image from analytical model
    psf_am, full_psf = impastis.analytical_model(zern_number, Aber, cali=True)
    # Get the mean contrast from image PASTIS
    contrast_am = np.mean(psf_am[np.where(psf_am != 0)]) + contrast_base
    end_impastis = time.time()

    ### MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(Aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ratio = contrast_am / contrast_matrix

    # Outputs
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
