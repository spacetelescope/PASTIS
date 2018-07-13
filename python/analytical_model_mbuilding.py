"""This program constructs the matrix M for PASTIS and compares the contrast of PASTIS with image generation with
the contrast from matrix PASTIS."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from python.config import CONFIG_INI
import python.util_pastis as util
import python.analytical_model as am
import python.webbpsf_imaging as webbim


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 11 minutes

    # Parameters
    dataDir = CONFIG_INI.get('local', 'local_data_path')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    largeur = tel_size_px * sampling
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get('filter', 'name')
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    real_samp = sampling * tel_size_px / im_size
    wvln = CONFIG_INI.getint('filter', 'lambda')
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    #-# Generating the PASTIS matrix
    matrix_pastis = np.zeros([nb_seg, nb_seg])   # Generate empty matrix

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('STEP:', str(i+1) + '-' + str(j+1), '/', str(nb_seg) + '-' + str(nb_seg))

            # Putting 1 nm only on segments i and j
            tempA = np.zeros([nb_seg])
            tempA[i] = nm_aber
            tempA[j] = nm_aber

            temp_im_am = am.analytical_model(zern_number, tempA, cali=True)
            matrix_pastis[i,j] = np.mean(temp_im_am[np.where(temp_im_am != 0)])

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_pastis)

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_pastis[i,j] = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.

    # Save matrix to file
        filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        util.write_fits(matrix_pastis, os.path.join(dataDir, 'results', filename + '.fits'), header=None, metadata=None)



    #-# Compare PASTIS contrast from PSF generation and PASTIS contrast from matrix PASTIS
    # Create random aberration coefficients
    Aber = np.random.random([nb_seg]) * 10   # piston values

    # mean subtraction for piston
    if zern_number == 1:
        Aber -= np.mean(Aber)

    ### WEBBPSF
    # Set up NIRCam and coronagraph
    psf_webbpsf = webbim.nircam_coro(filter, fpm, lyot_stop)
    # Get the mean conrast from the WebbPSF coronagraph
    contrast_webbpsf = np.mean(psf_webbpsf[np.where(psf_webbpsf != 0)])

    ### IMAGE PASTIS
    # Create calibrated image from analytical model
    dh_psf = am.analytical_model(zern_number, Aber, cali=True)
    # Get the mean contrast from image PASTIS
    contrast_am = np.mean(dh_psf[np.where(dh_psf != 0)])

    # Load in baseline contast
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_base = float(np.loadtxt(os.path.join(dataDir, 'calibration', contrastname+'.txt')))

    ### MATRIX PASTIS
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(Aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
    ratio = contrast_am / contrast_matrix

    print('Mean contrast from WebbPSF:', contrast_webbpsf)
    print('Mean contrast with image PASTIS:', contrast_am)
    print('Contrast from matrix PASTIS:', contrast_matrix)
    print('Ratio image PASTIS / matrix PASTIS:', ratio)

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for analytical_model_mbuild.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')


