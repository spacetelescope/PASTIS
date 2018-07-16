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
    start_time = time.time()   # runtime currently is around 29 minutes

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
    real_samp = sampling * tel_size_px / im_size

    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')    # [nm] amplitude of aberration
    zern_number = CONFIG_INI.getint('calibration', 'zernike')               # Which (Noll) Zernike we are calibrating for
    wss_zern_nb = util.noll_to_wss(zern_number)                             # Convert from Noll to WSS framework

    # If subfolder "calibration" doesn't exist yet, create it.
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # Create Zernike mode object for easier handling
    zern_mode = util.ZernikeMode(zern_number)

    # Create NIRCam objects, one for perfect PSF and one with coronagraph
    nc = webbpsf.NIRCam()
    # Set filter
    nc.filter = filter

    # Same for coronagraphic case
    nc_coro = webbpsf.NIRCam()
    nc_coro.filter = filter

    # Add coronagraphic elements to nc_coro
    nc_coro.image_mask = fpm
    nc_coro.pupil_mask = lyot_stop

    # Generate the PSFs
    print('Calculating perfect PSF without coronograph...')
    psf_start_time = time.time()
    psf_default_hdu = nc.calc_psf(fov_pixels=int(im_size), nlambda=1) # monochromatic=wvln/1e9)
    psf_end_time = time.time()
    print('Calculating the PSF with WebbPSF took', psf_end_time-psf_start_time, 'sec =', (psf_end_time-psf_start_time)/60, 'min')
    print('Calculating perfect PSF with coronagraph...\n')
    psf_coro_hdu = nc_coro.calc_psf(fov_pixels=int(im_size), nlambda=1) # monochromatic=wvln/1e9)

    # Extract the PSFs to image arrays - the [1] extension gives me detector resolution
    psf_default = psf_default_hdu[1].data
    psf_coro = psf_coro_hdu[1].data

    # Get maximum of PSF for the normalization and normalize PSFs we have so far
    normp = np.max(psf_default)
    psf_default = psf_default / normp
    psf_coro = psf_coro / normp

    # Create the dark hole and cut out the inner part of it
    dh_area = util.create_dark_hole(psf_coro, inner_wa, outer_wa, real_samp)
    dh_area_zoom = util.zoom(dh_area, int(dh_area.shape[0] / 2.), int(dh_area.shape[1] / 2.), 25)

    # Calculate the baseline contrast *with* the coronagraph and *without* aberrations and save value to file
    contrast_base = np.mean(psf_coro[dh_area_zoom])
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_fake_array = np.array(contrast_base).reshape(1,)   # Convert int to array of shape (1,), otherwise np.savetxt() doesn't work
    np.savetxt(os.path.join(outDir, contrastname+'.txt'), contrast_fake_array)

    # Create the arrays to hold the contrast values from the iterations
    contrastAPLC_vec_int = np.zeros([nb_seg])
    contrastAM_vec_int = np.zeros([nb_seg])

    # Create OTE for coro
    nc_coro, ote_coro = webbpsf.enable_adjustable_ote(nc_coro)

    # Loop over each individual segment, putting always the same aberration on
    for i in range(nb_seg):

        # Create the name of the segment the loop is currently at
        seg = wss_segs[i].split('-')[0]

        print('Working on segment ' + str(i+1) + '/' + str(nb_seg) + ': ' + seg)
        # We have to make sure here that we aberrate the segments in their order of numbering as it was set
        # in the script that generates the aperture (here: function_baselinify.py)!

        Aber_WSS = np.zeros([nb_seg, zern_max])           # The Zernikes here will be filled in the WSS order!!!
                                                          # Because it goes into _apply_hexikes_to_seg().
        Aber_Noll = np.copy(Aber_WSS)                     # This is the Noll version for later.
        Aber_WSS[i, wss_zern_nb-1] = nm_aber / 1e9        # Aberration on the segment we're currenlty working on;
                                                          # convert to meters; -1 on the Zernike because Python starts
                                                          # numbering at 0.
        Aber_Noll[i, zern_number-1] = nm_aber             # Noll version - in nm!

        #-# Crate OPD with aberrated segment(s)
        print('Applying aberration to OTE.')
        ote_coro._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

        # If you want to display it:
        #ote_coro.display_opd()
        #plt.show()

        #-# Generate the coronagraphic PSF
        print('Calculating coronagraphic PSF.')
        psf_endsim = nc_coro.calc_psf(fov_pixels=int(im_size), nlambda=1) # monochromatic=wvln/1e9)
        psf_end = psf_endsim[1].data

        #-# Normalize coro PSF
        psf_end = psf_end / normp

        #-# Crop coro PSF and DH to same small size (like in analytical_model.py)
        psf_end_zoom = util.zoom(psf_end, int(psf_end.shape[0] / 2.), int(psf_end.shape[1] / 2.), 25)

        #-# Get end-to-end image in DH, calculate the contrast (mean) and put it in array
        im_end = psf_end_zoom * dh_area_zoom
        contrastAPLC_vec_int[i] = np.mean(im_end[np.where(im_end != 0)])

        #-# Create image from analytical model, (normalize,) calculate contrast (mean) and put in array
        im_am = impastis.analytical_model(zern_number, Aber_Noll[:,zern_number-1], cali=False)
        contrastAM_vec_int[i] = np.mean(im_am[np.where(im_am != 0)])

    print('\n--- All PSFs calculated. ---\n')
    # Calculate calibration vector
    calibration = np.zeros_like(contrastAPLC_vec_int)
    calibration = (contrastAPLC_vec_int - contrast_base) / contrastAM_vec_int

    #-# Save calibration vector
    filename = 'calibration_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(calibration, os.path.join(outDir, filename+'.fits'), header=None, metadata=None)

    # Generate some plots
    plt.plot(contrastAPLC_vec_int, label='WebbPSF')
    plt.plot(contrastAM_vec_int, label='imagePASTIS')
    plt.title('Aberration per segment: ' + str(nm_aber) + ' nm')
    plt.xlabel('Segment number')
    plt.ylabel('Contrast')
    plt.legend()
    plt.savefig(os.path.join(outDir, filename+'.pdf'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for atlast_calibration.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')

    # Extra comments from Lucie:
    ### Your calibration factor for each segment will be the ratio between the contrast from end-to-end simulation
    ### and PASTIS.

    ### If there were an apodizer, leave it in when calculating psf_default.
    # Leave Lyot stop in for psf_default?? -> try it, check max of value, because that's our normalization factor