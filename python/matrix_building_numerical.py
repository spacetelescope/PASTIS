"""This program constructs the matrix M for PASTIS *NUMERICALLY FROM WEBBPSF* and saves it."""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import webbpsf

from python.config import CONFIG_INI
import python.util_pastis as util
import python.webbpsf_imaging as webbim

# Set WebbPSF environment variable
os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()

    # Parameters
    resDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'matrix_numerical')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    im_size_e2e = CONFIG_INI.getint('numerical', 'im_size_px_webbpsf')
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
    pup_im = np.zeros([im_size_e2e, im_size_e2e])    # this is just used for DH mask generation
    dh_area = util.create_dark_hole(pup_im, inner_wa, outer_wa, sampling)

    # Create a direct WebbPSF image for normalization factor
    fake_aber = np.zeros([nb_seg, zern_max])
    psf_perfect = webbim.nircam_nocoro(filter, fake_aber)
    normp = np.max(psf_perfect)
    psf_perfect = psf_perfect / normp

    # Set up NIRCam coro object from WebbPSF
    nc_coro = webbpsf.NIRCam()
    nc_coro.filter = filter
    nc_coro.image_mask = fpm
    nc_coro.pupil_mask = lyot_stop

    # Null the OTE OPDs for the PSFs, maybe we will add internal WFE later.
    nc_coro, ote_coro = webbpsf.enable_adjustable_ote(nc_coro)      # create OTE for coronagraph
    nc_coro.include_si_wfe = False                                  # set SI internal WFE to zero

    #-# Generating the PASTIS matrix and a list for all contrasts
    matrix_pastis = np.zeros([nb_seg, nb_seg])   # Generate empty matrix
    all_psfs = []
    all_dhs = []
    all_contrasts = []

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('\nSTEP:', str(i+1) + '-' + str(j+1), '/', str(nb_seg) + '-' + str(nb_seg))

            # Get names of segments, they're being addressed by their names in the ote functions.
            seg_i = wss_segs[i].split('-')[0]
            seg_j = wss_segs[j].split('-')[0]

            # Put the aberration on the correct segments
            Aber_WSS = np.zeros([nb_seg, zern_max])         # The Zernikes here will be filled in the WSS order!!!
                                                            # Because it goes into _apply_hexikes_to_seg().
            Aber_WSS[i, wss_zern_nb - 1] = nm_aber / 1e9    # Aberration on the segment we're currently working on;
                                                            # convert to meters; -1 on the Zernike because Python starts
                                                            # numbering at 0.
            Aber_WSS[j, wss_zern_nb - 1] = nm_aber / 1e9    # same for other segment

            # Putting aberrations on segments i and j
            ote_coro.reset()    # Making sure there are no previous movements on the segments.
            ote_coro.zero()     # set OTE for coronagraph to zero

            # Apply both aberrations to OTE. If i=j, apply only once!
            ote_coro._apply_hexikes_to_seg(seg_i, Aber_WSS[i, :])    # set segment i  (segment numbering starts at 1)
            if i != j:
                ote_coro._apply_hexikes_to_seg(seg_j, Aber_WSS[j, :])    # set segment j

            # If you want to display it:
            # ote_coro.display_opd()
            # plt.show()

            # Save ODP images for testing
            opd_name = 'opd_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            plt.clf()
            ote_coro.display_opd()
            plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

            print('Calculating WebbPSF image')
            image = nc_coro.calc_psf(fov_pixels=int(im_size_e2e), oversample=1, nlambda=1)
            psf = image[0].data / normp

            # Save WebbPSF image to disk
            filename_psf = 'psf_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            util.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'), header=None, metadata=None)
            all_psfs.append(psf)

            print('Calculating mean contrast in dark hole')
            dh_intensity = psf * dh_area
            contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
            print('contrast:', contrast)

            # Save DH image to disk and put current contrast in list
            filename_dh = 'dh_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            util.write_fits(dh_intensity, os.path.join(resDir, 'darkholes', filename_dh + '.fits'), header=None, metadata=None)
            all_dhs.append(dh_intensity)
            all_contrasts.append(contrast)

            # Fill according entry in the matrix
            matrix_pastis[i,j] = contrast

    # Transform saved lists to arrays
    all_psfs = np.array(all_psfs)
    all_dhs = np.array(all_dhs)
    all_contrasts = np.array(all_contrasts)

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_pastis)

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                print('Off-axis for i' + str(i+1) + '-j' + str(j+1) + ': ' + str(matrix_off_val))

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'), header=None, metadata=None)
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Save extra data to disk
    util.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube' + '.fits'), header=None, metadata=None)
    util.write_fits(all_dhs, os.path.join(resDir, 'darkholes', 'dh_cube' + '.fits'), header=None, metadata=None)
    np.savetxt(os.path.join(resDir, 'contrasts.txt'), all_contrasts, fmt='%2.2f')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')

    # -- Runtime notes: --
    #
    # im_size = 128
    # oversampling = 1
    # nb_seg = 18
    # runtime = 20 min
