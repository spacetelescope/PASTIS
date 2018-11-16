"""
Translation of atlast_calibration.pro, which makes the calibration files for PASTIS.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import webbpsf

from python.config import CONFIG_INI
import python.util_pastis as util
import python.image_pastis as impastis


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 5 minutes

    # Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
    # webbpsf.utils.get_webbpsf_data_path()
    # --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
    os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')

    # Parameters
    outDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'calibration')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get('filter', 'name')
    wvln = CONFIG_INI.getfloat('filter', 'lambda')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    size_seg = CONFIG_INI.getint('numerical', 'size_seg')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    #real_samp = sampling * tel_size_px / im_size

    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')    # [nm] amplitude of aberration
    zern_number = CONFIG_INI.getint('calibration', 'zernike')               # Which (Noll) Zernike we are calibrating for
    wss_zern_nb = util.noll_to_wss(zern_number)                             # Convert from Noll to WSS framework

    # If subfolder "calibration" doesn't exist yet, create it.
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # If subfolder "calibration" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(outDir, 'images')):
        os.mkdir(os.path.join(outDir, 'images'))

    # Create Zernike mode object for easier handling
    zern_mode = util.ZernikeMode(zern_number)

    # Create NIRCam objects, one for perfect PSF and one with coronagraph
    print('Setting up the E2E simulation.')
    nc = webbpsf.NIRCam()
    # Set filter
    nc.filter = filter

    # Same for coronagraphic case
    nc_coro = webbpsf.NIRCam()
    nc_coro.filter = filter

    # Add coronagraphic elements to nc_coro
    nc_coro.image_mask = fpm
    nc_coro.pupil_mask = lyot_stop

    # Null the OTE OPDs for the PSFs, maybe we will add internal WFE later.
    nc, ote = webbpsf.enable_adjustable_ote(nc)                     # create OTE for default PSF
    nc_coro, ote_coro = webbpsf.enable_adjustable_ote(nc_coro)      # create OTE for coronagraph
    ote.zero()          # set OTE for default PSF to zero
    ote_coro.zero()     # set OTE for coronagraph to zero

    # Generate the E2E PSFs with and without coronagraph
    print('Calculating perfect PSF without coronograph...')
    psf_start_time = time.time()
    psf_default_hdu = nc.calc_psf(fov_pixels=int(im_size), oversample=1, nlambda=1)
    psf_end_time = time.time()
    print('Calculating this PSF with WebbPSF took', psf_end_time-psf_start_time, 'sec =', (psf_end_time-psf_start_time)/60, 'min')
    print('Calculating perfect PSF with coronagraph...\n')
    psf_coro_hdu = nc_coro.calc_psf(fov_pixels=int(im_size), oversample=1, nlambda=1)

    # Extract the PSFs to image arrays - the [1] extension gives me detector resolution
    psf_default = psf_default_hdu[1].data
    psf_coro = psf_coro_hdu[1].data

    # Save the PSFs for testing
    util.write_fits(psf_default, os.path.join(outDir, 'psf_default.fits'), header=None, metadata=None)
    util.write_fits(psf_coro, os.path.join(outDir, 'psf_coro.fits'), header=None, metadata=None)

    # Get maximum of PSF for the normalization and normalize PSFs we have so far
    normp = np.max(psf_default)
    psf_default = psf_default / normp
    psf_coro = psf_coro / normp

    # Create the dark hole
    dh_area = util.create_dark_hole(psf_coro, inner_wa, outer_wa, sampling)

    # Calculate the baseline contrast *with* the coronagraph and *without* aberrations and save the value to file
    contrast_im = psf_coro * dh_area
    contrast_base = np.mean(contrast_im[np.where(contrast_im != 0)])
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_fake_array = np.array(contrast_base).reshape(1,)   # Convert int to array of shape (1,), otherwise np.savetxt() doesn't work
    np.savetxt(os.path.join(outDir, contrastname+'.txt'), contrast_fake_array)

    # Create the arrays to hold the contrast values from the iterations
    contrastAPLC_vec_int = np.zeros([nb_seg])
    contrastAM_vec_int = np.zeros([nb_seg])

    # Loop over each individual segment, putting always the same aberration on
    for i in range(nb_seg):

        iter_start = time.time()

        # Create the name of the segment the loop is currently at
        seg = wss_segs[i].split('-')[0]

        print('')
        print('Working on segment ' + str(i+1) + '/' + str(nb_seg) + ': ' + seg)
        # We have to make sure here that we aberrate the segments in their order of numbering as it was set
        # in the script that generates the aperture (here: function_baselinify.py)!
        # Currently there is a bug in WebbPSF though that numbers the segments wrong when used in the exit pupil
        # orientation, hence I added this quickfix until it is fixed inside WebbPSF:

        ### FIX FOR MISSING LEFT_RIGHT FLIP IN WEBBPSF'S EXIT PUPIL ### - remove when it gets fixed in WebbPSF
        # inner circle of segments
        if seg == 'A6':
            seg = 'A2'
        elif seg == 'A2':
            seg = 'A6'
        if seg == 'A5':
            seg = 'A3'
        elif seg == 'A3':
            seg = 'A5'

        # outer circle of segments
        if seg == 'C6':
            seg = 'C1'
        elif seg == 'C1':
            seg = 'C6'
        if seg == 'B6':
            seg = 'B2'
        elif seg == 'B2':
            seg = 'B6'
        if seg == 'C5':
            seg = 'C2'
        elif seg == 'C2':
            seg = 'C5'
        if seg == 'B5':
            seg = 'B3'
        elif seg == 'B3':
            seg = 'B5'
        if seg == 'C4':
            seg = 'C3'
        elif seg == 'C3':
            seg = 'C4'
        ### FIX END ###

        Aber_WSS = np.zeros([nb_seg, zern_max])           # The Zernikes here will be filled in the WSS order!!!
                                                          # Because it goes into _apply_hexikes_to_seg().
        Aber_Noll = np.copy(Aber_WSS)                     # This is the Noll version for later.
        Aber_WSS[i, wss_zern_nb-1] = nm_aber / 1e9        # Aberration on the segment we're currenlty working on;
                                                          # convert to meters; -1 on the Zernike because Python starts
                                                          # numbering at 0.
        Aber_Noll[i, zern_number-1] = nm_aber             # Noll version - in nm!

        #-# Crate OPD with aberrated segment(s)
        print('Applying aberration to OTE.')
        ote_coro.reset()   # Making sure there are no previous movements on the segments.
        ote_coro.zero()    # For now, ignore internal WFE.
        ote_coro._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

        # If you want to display it:
        #ote_coro.display_opd()
        #plt.show()

        #-# Generate the coronagraphic PSF
        print('Calculating coronagraphic PSF.')
        psf_endsim = nc_coro.calc_psf(fov_pixels=int(im_size), oversample=1, nlambda=1)
        psf_end = psf_endsim[1].data

        #-# Normalize coro PSF
        psf_end = psf_end / normp   # NORM

        #-# Get end-to-end image in DH, calculate the contrast (mean) and put it in array
        im_end = psf_end * dh_area
        contrastAPLC_vec_int[i] = np.mean(im_end[np.where(im_end != 0)])

        #-# Create image from analytical model, calculate contrast (mean, in DH) and put in array
        dh_im_am, full_im_am = impastis.analytical_model(zern_number, Aber_Noll[:,zern_number-1], cali=False)
        contrastAM_vec_int[i] = np.mean(dh_im_am[np.where(dh_im_am != 0)])

        print('Contrast WebbPSF:', contrastAPLC_vec_int[i])
        print('Contrast image-PASTIS, uncalibrated:', contrastAM_vec_int[i])

        # # Save images for testing
        # im_am_name = 'image_pastis_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        # util.write_fits(full_im_am, os.path.join(outDir, 'images', im_am_name + '.fits'))
        # #dh_im_am - for image with DH
        # im_end_name = 'image_webbpsf_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        # util.write_fits(psf_end, os.path.join(outDir, 'images', im_end_name + '.fits'))
        # #im_end - for image with DH
        # # Save OTE OPD
        # opd_name = 'opd_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        # plt.clf()
        # ote_coro.display_opd()
        # plt.savefig(os.path.join(outDir, 'images', opd_name + '.pdf'))

        iter_end = time.time()
        print('Iteration', i+1, 'runtime:', iter_end-iter_start, 'sec =', (iter_end-iter_start)/60, 'min')

    print('\n--- All PSFs calculated. ---\n')
    # Calculate calibration vector
    calibration = np.zeros_like(contrastAPLC_vec_int)
    calibration = (contrastAPLC_vec_int - contrast_base) / contrastAM_vec_int
    #calibration = contrastAPLC_vec_int / contrastAM_vec_int   # without taking C_0 into account

    #-# Save calibration vector
    filename = 'calibration_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(calibration, os.path.join(outDir, filename+'.fits'), header=None, metadata=None)

    # Save contrast vectors for WebbPSF and image-PASTIS so that we can look at the values if needed
    name_webbpsf = 'calibration_contrast_WEBBPSF_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    name_impastis = 'calibration_contrast_IMAGE-PASTIS_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(contrastAPLC_vec_int, os.path.join(outDir, name_webbpsf+'.fits'), header=None, metadata=None)
    util.write_fits(contrastAM_vec_int, os.path.join(outDir, name_impastis + '.fits'), header=None, metadata=None)

    # Generate some plots
    plt.clf()
    plt.plot(contrastAPLC_vec_int, label='WebbPSF')
    plt.plot(contrastAM_vec_int, label='imagePASTIS')
    plt.title('Aberration per segment: ' + str(nm_aber) + ' nm')
    plt.xlabel('Segment number')
    plt.ylabel('Contrast')
    plt.legend()
    plt.savefig(os.path.join(outDir, filename+'.pdf'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for calibration.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')

    # Extra comments from Lucie:
    ### Your calibration factor for each segment will be the ratio between the contrast from end-to-end simulation
    ### and PASTIS.

    ### If there were an apodizer, leave it in when calculating psf_default.
    # Leave Lyot stop in for psf_default?? -> try it, check max of value, because that's our normalization factor