"""This program constructs the matrix M for PASTIS and saves it."""

import os
import time
import numpy as np

from python.config import CONFIG_INI
import python.util_pastis as util
import python.image_pastis as impastis


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 10 minutes

    # Parameters
    resDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'results')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    #-# Generating the PASTIS matrix
    matrix_pastis = np.zeros([nb_seg, nb_seg])   # Generate empty matrix

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('STEP:', str(i+1) + '-' + str(j+1), '/', str(nb_seg) + '-' + str(nb_seg))

            # Putting aberration only on segments i and j
            tempA = np.zeros([nb_seg])
            tempA[i] = nm_aber
            tempA[j] = nm_aber

            temp_im_am, full_psf = impastis.analytical_model(zern_number, tempA, cali=True)
            contrast = np.mean(temp_im_am[np.where(temp_im_am != 0)])
            matrix_pastis[i,j] = contrast
            print('contrast =', contrast)

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_pastis)

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                print('Off-axis for i' + str(i+1) + '-j' + str(j+1) + ': ' + str(matrix_off_val))

    # Save matrix to file
    filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(matrix_pastis, os.path.join(resDir, filename + '.fits'), header=None, metadata=None)

    print('Matrix saved to:', os.path.join(resDir, filename + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
