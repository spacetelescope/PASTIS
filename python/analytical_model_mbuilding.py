"""This program constructs the matrix M for PASTIS."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from python.config import CONFIG_INI
import python.util_pastis as util
import python.analytical_model as am


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()

    # Parameters
    outDir = os.path.join('..', 'data', 'py_data')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    largeur = tel_size_px * sampling
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

            # Putting 1 nm only on segments i and j - why 1nm again, not random? Or is it actually always 1 nm?
            tempA = np.zeros([nb_seg, nb_seg])
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
        util.write_fits(matrix_pastis, os.path.join(outDir, filename + '.fits'), header=None, metadata=None)

    #-# Compare PASTIS contrast from PSF generation and PASTIS contrast from matrix PASTIS
    # Create random aberration coefficients
    Aber = np.random.random([nb_seg]) * 10   # piston values

    # mean subtraction for piston
    if zern_number == 1:
        Aber -= np.mean(Aber)

    # Create calibrated image from analytical model
    dh_psf = am.analytical_model(zern_number, Aber, cali=True)

    # Get the mean contrast from AM
    contrast_am = np.mean(dh_psf[np.where(dh_psf != 0)])
    print('Dark hole mean from analytical model:', contrast_am)

    # Calculate final contrast
    result = np.matmul(np.matmul(Aber, matrix_pastis), Aber)   # generating final matriix PASTIS result
    contrast_final = contrast_am / result
    print('Mean contrast with AM:', contrast_am)
    print('Mean contrast with matrices:', contrast_final)

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for analytical_model_mbuild.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')


