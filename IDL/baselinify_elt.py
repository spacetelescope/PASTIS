"""
Construction of the matrices to go from Redundant Pairs (RP) to Non-Redundant Pairs (NRP) for the phase retrieval
code PASTIS (Leboulleux, 2017).

Author: P. Zeidler
Adapted by: Iva Laginja
"""

import numpy as np
from astropy.io import ascii, fits
from multiprocessing import Pool, cpu_count
from time import time


def matrix_calc(i):
    """
    What is this function about?
    :param i:
    :return:
    """
    return_vec = np.array([None, None])

    if i == 0:
        return_vec = np.array([0, i])

    if i != 0:
        for k in range(i):
            if np.abs(two_norm_reshaped[i] - two_norm_reshaped[k]) >= 4:
                return_vec = np.array([1, np.nan])
            if np.abs(two_norm_reshaped[i] - two_norm_reshaped[k]) < 4:
                if np.abs(np.cross(np.array([vec_list_x[i], vec_list_y[i]]),
                                   np.array([vec_list_x[k], vec_list_y[k]]))) >= 1000:
                    return_vec = np.array([1, np.nan])
                else:
                    return np.array([0, i])

    return return_vec


if __name__ == "__main__":

    # Record when code is starting
    start_time = time()

    # Old data?
    # data = ascii.read('SegmentVertices.txt')
    # x, y = data['col1'][0:150],data['col2'][0:150]
    # x, y = data['col1'],data['col2']

    # Load the maps of centers of different segments
    x_file = 'x_center_luvoir.fits'
    y_file = 'y_center_luvoir.fits'

    x = fits.open(x_file)[0].data
    y = fits.open(y_file)[0].data

    nb_seg = len(x)
    seg_position = np.zeros((nb_seg, 2))

    seg_position[:, 0] = x * 100.
    seg_position[:, 1] = y * 100.

    vec_list = np.zeros((nb_seg, nb_seg, 2))

    for i in range(nb_seg):
        for j in range(nb_seg):
            vec_list[j, i, :] = seg_position[i, :] - seg_position[j, :]

    two_norm = np.linalg.norm(vec_list, axis=2)

    two_norm_reshaped = two_norm.reshape(len(two_norm) ** 2)
    vec_list_x = vec_list[:, :, 0].reshape(len(two_norm) ** 2)
    vec_list_y = vec_list[:, :, 1].reshape(len(two_norm) ** 2)
    vec_list_reshaped = vec_list.reshape((len(two_norm) ** 2, -1))

    counter = np.arange(len(two_norm_reshaped))
    pool = Pool(cpu_count())  # If you need to limit the number of CPUs used: substitute cpu_count() with number of desired CPUs
    #pool = Pool(1)           # If you need to limit the number of CPUs used: substitute cpu_count() with number of desired CPUs

    zero_indicator = pool.map(matrix_calc, counter)

    pool.close()
    pool.join()

    zero_indicator = np.array(zero_indicator)
    vec_list_reshaped_bck = vec_list_reshaped

    vec_list_reshaped[zero_indicator[:, 1][~np.isnan(zero_indicator[:, 1])].astype(int)] = (0., 0.)

    vec_list_new = vec_list_reshaped.reshape(np.shape(vec_list))

    # Record end time of code run
    end_time = time()

    # Save result vector as fits file
    hdu = fits.PrimaryHDU(vec_list_new)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('cube.fits', overwrite=True)

    # Some outputs
    print('number of non_zero elements by          counter: ' + str(np.nansum(zero_indicator[:, 0])))
    print('number of non_zero elements by (0,0) - elements: ' + str(
        len(np.where((vec_list_reshaped[:, 1] != 0) | (vec_list_reshaped[:, 0] != 0))[0])))
    print('size of final array: ' + str(np.shape(vec_list_new)))

    print('Execution time: ', end_time - start_time)
