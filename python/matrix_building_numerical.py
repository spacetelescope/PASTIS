"""This program constructs the matrix M for PASTIS *NUMERICALLY FROM WEBBPSF* and saves it."""

import os
import time
import numpy as np
import webbpsf

from python.config import CONFIG_INI
import python.util_pastis as util

# Set WebbPSF environment variable
os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 10 minutes

    # Parameters
    resDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'results')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get('filter', 'name')
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling
    wss_zern_nb = util.noll_to_wss(zern_number)                     # Convert from Noll to WSS framework

    # Create the dark hole mask.
    pup_im = np.zeros([im_size, im_size])    # this is just used for DH mask generation
    dh_area = util.create_dark_hole(pup_im, inner_wa, outer_wa, sampling)

    # Set up NIRCam object from WebbPSF
    nc_coro = webbpsf.NIRCam()
    nc_coro.filter = filter
    nc_coro.image_mask = fpm
    nc_coro.pupil_mask = lyot_stop

    # Null the OTE OPDs for the PSFs, maybe we will add internal WFE later.
    nc_coro, ote_coro = webbpsf.enable_adjustable_ote(nc_coro)      # create OTE for coronagraph

    #-# Generating the PASTIS matrix
    matrix_pastis = np.zeros([nb_seg, nb_seg])   # Generate empty matrix

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('STEP:', str(i+1) + '-' + str(j+1), '/', str(nb_seg) + '-' + str(nb_seg))

            # Get names of segments, they're being addressed by their names in the ote functions.
            seg_i = wss_segs[i].split('-')[0]
            seg_j = wss_segs[j].split('-')[0]

            Aber_WSS = np.zeros([nb_seg, zern_max])         # The Zernikes here will be filled in the WSS order!!!
                                                            # Because it goes into _apply_hexikes_to_seg().
            Aber_WSS[i, wss_zern_nb - 1] = nm_aber / 1e9    # Aberration on the segment we're currently working on;
                                                            # convert to meters; -1 on the Zernike because Python starts
                                                            # numbering at 0.
            Aber_WSS[j, wss_zern_nb - 1] = nm_aber / 1e9    # same for other segment

            # Putting aberrations on segments i and j
            ote_coro.reset()    # Making sure there are no previous movements on the segments.
            ote_coro.zero()     # set OTE for coronagraph to zero
            ote_coro._apply_hexikes_to_seg(seg_i, Aber_WSS[i, :])    # set segment i
            ote_coro._apply_hexikes_to_seg(seg_j, Aber_WSS[j, :])    # set segment j

            # If you want to display it:
            # ote_coro.display_opd()
            # plt.show()

            print('Calculating WebbPSF image')
            image = nc_coro.calc_psf(fov_pixels=int(im_size), oversample=1, nlambda=1)
            psf = image[1].data

            print('Calculating mean contrast in dark hole')
            dh_intensity = psf * dh_area
            contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
            print('contrast:', contrast)

            # Fill according entry in the matrix
            matrix_pastis[i,j] = contrast

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_pastis)

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_pastis[i,j] = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.

    # Save matrix to file
    filename = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(matrix_pastis, os.path.join(resDir, filename + '.fits'), header=None, metadata=None)

    print('Matrix saved to:', os.path.join(resDir, filename + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
