"""This program constructs the matrix M for PASTIS and saves it."""

import os
import time
import numpy as np

from python.config import CONFIG_INI
import python.util_pastis as util
import python.image_pastis as impastis


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 minutes

    # Parameters
    datadir = CONFIG_INI.get('local', 'local_data_path')
    resDir = os.path.join(datadir, 'matrix_analytical')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')       # Noll convention!
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    # Load baseline contrast
    blcontr = np.loadtxt(os.path.join(datadir, 'calibration', 'base-contrast_piston_Noll1.txt'))

    #-# Generating the PASTIS matrix
    matrix_direct = np.zeros([nb_seg, nb_seg])   # Generate empty matrix for contrast values from loop.
    all_ims = []
    all_dhs = []
    all_contrasts = []

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('STEP:', str(i+1) + '-' + str(j+1), '/', str(nb_seg) + '-' + str(nb_seg))

            # Putting aberration only on segments i and j
            tempA = np.zeros([nb_seg])
            tempA[i] = nm_aber
            tempA[j] = nm_aber

            # Create PASTIS image and save full image as well as DH image
            temp_im_am, full_psf = impastis.analytical_model(zern_number, tempA, cali=True)

            filename_psf = 'psf_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            util.write_fits(full_psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'), header=None, metadata=None)
            all_ims.append(full_psf)

            filename_dh = 'dh_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            util.write_fits(temp_im_am, os.path.join(resDir, 'darkholes', filename_dh + '.fits'), header=None, metadata=None)
            all_dhs.append(temp_im_am)

            contrast = np.mean(temp_im_am[np.where(temp_im_am != 0)])
            matrix_direct[i,j] = contrast
            print('contrast =', contrast)
            all_contrasts.append(contrast)

    all_ims = np.array(all_ims)
    all_dhs = np.array(all_dhs)
    all_contrasts = np.array(all_contrasts)

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_direct)     # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(matrix_direct)    # This will be the final PASTIS matrix.

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i, j] - matrix_two_N[i, i] - matrix_two_N[j, j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                print('Off-axis for i' + str(i+1) + '-j' + str(j+1) + ': ' + str(matrix_off_val))

    # Save matrix to file
    filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(matrix_pastis, os.path.join(resDir, filename + '.fits'), header=None, metadata=None)

    print('Matrix saved to:', os.path.join(resDir, filename + '.fits'))

    # Save the PSF and DH image cubes as well
    util.write_fits(all_ims, os.path.join(resDir, 'psfs', 'psf_cube' + '.fits'), header=None, metadata=None)
    util.write_fits(all_dhs, os.path.join(resDir, 'darkholes', 'dh_cube' + '.fits'), header=None, metadata=None)
    np.savetxt(os.path.join(resDir, 'contrasts.txt'), all_contrasts, fmt='%2.2f')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
