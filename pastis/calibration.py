"""
Making the calibration files for PASTIS.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import webbpsf

from config import CONFIG_INI
import util_pastis as util
import image_pastis as impastis


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()   # runtime currently is around 10 minutes

    # Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
    # webbpsf.utils.get_webbpsf_data_path()
    # --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
    os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')

    # Parameters
    outDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'active', 'calibration')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get('filter', 'name')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    im_size_e2e = CONFIG_INI.getint('numerical', 'im_size_px_webbpsf')
    size_seg = CONFIG_INI.getint('numerical', 'size_seg')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')

    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration') * u.nm       # [nm] amplitude of aberration
    zern_number = CONFIG_INI.getint('calibration', 'zernike')               # Which (Noll) Zernike we are calibrating for
    wss_zern_nb = util.noll_to_wss(zern_number)                             # Convert from Noll to WSS framework

    # If subfolder "calibration" doesn't exist yet, create it.
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # If subfolder "images" in "calibration" doesn't exist yet, create it.
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
    ote.zero()                        # set OTE for default PSF to zero
    ote_coro.zero()                   # set OTE for coronagraph to zero
    nc.include_si_wfe = False         # set SI internal WFE to zero
    nc_coro.include_si_wfe = False    # set SI internal WFE to zero

    # Generate the E2E PSFs with and without coronagraph
    print('Calculating perfect PSF without coronograph...')
    psf_start_time = time.time()
    psf_default_hdu = nc.calc_psf(fov_pixels=int(im_size_e2e), oversample=1, nlambda=1)
    psf_end_time = time.time()
    print('Calculating this PSF with WebbPSF took', psf_end_time-psf_start_time, 'sec =', (psf_end_time-psf_start_time)/60, 'min')
    print('Calculating perfect PSF with coronagraph...\n')
    psf_coro_hdu = nc_coro.calc_psf(fov_pixels=int(im_size_e2e), oversample=1, nlambda=1)

    # Extract the PSFs to image arrays - the [1] extension gives me detector resolution
    psf_default = psf_default_hdu[0].data
    psf_coro = psf_coro_hdu[0].data

    # Get maximum of PSF for the normalization and normalize PSFs we have so far
    normp = np.max(psf_default)
    psf_default = psf_default / normp
    psf_coro = psf_coro / normp

    # Save the PSFs for testing
    util.write_fits(psf_default, os.path.join(outDir, 'psf_default.fits'), header=None, metadata=None)
    util.write_fits(psf_coro, os.path.join(outDir, 'psf_coro.fits'), header=None, metadata=None)

    # Create the dark hole
    dh_area = util.create_dark_hole(psf_coro, inner_wa, outer_wa, sampling)
    util.write_fits(dh_area, os.path.join(outDir, 'dh_area.fits'), header=None, metadata=None)

    # Calculate the baseline contrast *with* the coronagraph and *without* aberrations and save the value to file
    contrast_im = psf_coro * dh_area
    contrast_base = np.mean(contrast_im[np.where(contrast_im != 0)])
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)   #TODO: Why does the filename include a Zernike if this is supposed to be the perfect PSF without aberrations?
    contrast_fake_array = np.array(contrast_base).reshape(1,)   # Convert into array of shape (1,), otherwise np.savetxt() doesn't work
    np.savetxt(os.path.join(outDir, contrastname+'.txt'), contrast_fake_array)

    # Create the arrays to hold the contrast values from the iterations
    contrast_e2e = np.zeros([nb_seg])
    contrast_pastis = np.zeros([nb_seg])

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
        # orientation, so don't be confused when the segments are numbered wrong in the exit pupil!

        # Create arrays to hold Zernike aberration coefficients
        Aber_WSS = np.zeros([nb_seg, zern_max])           # The Zernikes here will be filled in the WSS order!!!
                                                          # Because it goes into _apply_hexikes_to_seg().
        Aber_Noll = np.copy(Aber_WSS)                     # This is the Noll version for input into PASTIS.

        # Feed the aberration nm_aber into the array position
        # that corresponds to the correct Zernike, but only on segment i
        Aber_WSS[i, wss_zern_nb-1] = nm_aber.to(u.m).value  # Aberration on the segment we're currently working on;
                                                            # convert to meters; -1 on the Zernike because Python starts
                                                            # numbering at 0.
        Aber_Noll[i, zern_number-1] = nm_aber.value         # Noll version - in nm

        # Make sure the aberration coefficients have correct units
        Aber_Noll *= u.nm
        # Aber_WSS does NOT get multiplied by u.m, because the poppy function it goes to is actually a private function
        # that is not decorated with the astropy decorator for checking units and does not use astropy.units. Which is
        # why we made sure it gets filled with values in units of meters already a couple of lines above this.

        #-# Crate OPD with aberrated segment(s)
        print('Applying aberration to OTE.')
        print('nm_aber: {}'.format(nm_aber))
        ote_coro.reset()   # Making sure there are no previous movements on the segments.
        ote_coro.zero()    # For now, ignore internal WFE.
        ote_coro._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

        # If you want to display it:
        #ote_coro.display_opd()
        #plt.show()

        #-# Generate the coronagraphic PSF
        print('Calculating coronagraphic PSF.')
        psf_endsim = nc_coro.calc_psf(fov_pixels=int(im_size_e2e), oversample=1, nlambda=1)
        psf_end = psf_endsim[0].data

        #-# Normalize coro PSF
        psf_end = psf_end / normp

        #-# Get end-to-end image in DH, calculate the contrast (mean) and put it in array
        im_end = psf_end * dh_area
        contrast_e2e[i] = np.mean(im_end[np.where(im_end != 0)])

        #-# Create image from PASTIS (analytical model), calculate contrast (mean, in DH) and put in array
        dh_im_am, full_im_am = impastis.analytical_model(zern_number, Aber_Noll[:, zern_number-1], cali=False)
        contrast_pastis[i] = np.mean(dh_im_am[np.where(dh_im_am != 0)])

        print('Contrast WebbPSF:', contrast_e2e[i])
        print('Contrast image-PASTIS, uncalibrated:', contrast_pastis[i])

        # Save images for testing
        im_am_name = 'image_pastis_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        util.write_fits(full_im_am, os.path.join(outDir, 'images', im_am_name + '.fits'))
        #dh_im_am - for image with DH
        im_end_name = 'image_webbpsf_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        util.write_fits(psf_end, os.path.join(outDir, 'images', im_end_name + '.fits'))
        #im_end - for image with DH
        # Save OTE OPD
        opd_name = 'opd_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_seg' + str(i+1)
        plt.clf()
        ote_coro.display_opd()
        plt.savefig(os.path.join(outDir, 'images', opd_name + '.pdf'))

        iter_end = time.time()
        print('Iteration', i+1, 'runtime:', iter_end-iter_start, 'sec =', (iter_end-iter_start)/60, 'min')

    print('\n--- All PSFs calculated. ---\n')
    # Calculate calibration vector
    calibration = np.zeros_like(contrast_e2e)

    calibration = np.sqrt((contrast_e2e - contrast_base) / contrast_pastis)

    #calibration = contrast_e2e / contrast_pastis   # without taking C_0 into account. should never to this, but it's
                                                    # ok for testing if calibration aberration is too low

    # Not taking C_0 into account is necessary to avoid negative values of the E2E contrast in cases where
    # we're calibrating at a level that is too low and numerically, the calibration contrast of one aberrated
    # segment might end up being lower than the baseline contrast, which could lead to negative values in the sqrt.

    #-# Save calibration vector
    filename = 'calibration_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(calibration, os.path.join(outDir, filename+'.fits'), header=None, metadata=None)

    # Save contrast vectors for WebbPSF and image-PASTIS so that we can look at the values if needed
    name_webbpsf = 'contrast_WEBBPSF_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    name_impastis = 'contrast_IMAGE-PASTIS_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(contrast_e2e, os.path.join(outDir, name_webbpsf+'.fits'), header=None, metadata=None)
    util.write_fits(contrast_pastis, os.path.join(outDir, name_impastis + '.fits'), header=None, metadata=None)

    # Generate some plots
    """
    plt.clf()
    plt.plot(contrast_e2e, label='WebbPSF')
    plt.plot(contrast_pastis, label='imagePASTIS')
    plt.title('Aberration per segment: {}'.format(nm_aber))
    plt.xlabel('Segment number')
    plt.ylabel('Contrast')
    plt.legend()
    plt.savefig(os.path.join(outDir, filename+'.pdf'))

    plt.clf()
    plt.plot(contrast_e2e, label='WebbPSF')
    plt.title('Aberration per segment: {}'.format(nm_aber))
    plt.xlabel('Segment number')
    plt.ylabel('Contrast')
    plt.legend()
    plt.savefig(os.path.join(outDir, 'only_webbpsf'+'.pdf'))

    plt.clf()
    plt.plot(contrast_pastis, label='imagePASTIS')
    plt.title('Aberration per segment: {}'.format(nm_aber))
    plt.xlabel('Segment number')
    plt.ylabel('Contrast')
    plt.legend()
    plt.savefig(os.path.join(outDir, 'only_pastis'+'.pdf'))
    """

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for calibration.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(outDir))

    # Extra comments from Lucie:
    ### Your calibration factor for each segment will be the ratio between the contrast from end-to-end simulation
    ### and PASTIS image contrast.

    ### If there were an apodizer, leave it in when calculating psf_default ("no coronagraph").
    # Leave Lyot stop in for psf_default?? -> try it, check max of value, because that's our normalization factor.
