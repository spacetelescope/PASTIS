"""This program constructs the matrix M for PASTIS and aves it."""

import os
import time
import numpy as np

from python.config import CONFIG_INI
import python.util_pastis as util
import python.image_pastis as impastis


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
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')         # focal plane mask
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

            temp_im_am, full_psf = impastis.analytical_model(zern_number, tempA, cali=True)
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

    print('Matrix saved to:', os.path.join(dataDir, 'results', filename + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for analytical_model_mbuild.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
