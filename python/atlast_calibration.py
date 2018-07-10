"""
Translation of atlast_calibration.pro, which makes the calibration files for PASTIS.
"""
import os
import numpy as np
import webbpsf

from python.config import CONFIG_INI
import python.util_pastis as util


if __name__ == '__main__':

    # Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
    # webbpsf.utils.get_webbpsf_data_path()
    # --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
    os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')

    # Parameters
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get('filter', 'name')
    wvln = CONFIG_INI.getfloat('filter', 'lambda')
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    size_seg = CONFIG_INI.getint('numerical', 'size_seg')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration_nm')    # [nm] amplitude of aberration
    zern_number = CONFIG_INI.getint('calibration', 'zernike')               # Which (Noll) Zernike we are calibrating for
    wss_zern_nb = util.noll_to_wss(zern_number)                             # Convert from Noll to WSS framework

    # Create NIRCam objects, for perfect PSF and one with coronagraph
    nc = webbpsf.NIRCam()
    # Don't include NIRCam specific WFE, because it is currently not available in WebbPSF
    nc.insert_si_wfe = False
    # Set filter
    nc.filter = filter

    # Same for coronagraphic case
    nc_coro = webbpsf.NIRCam()
    nc_coro.insert_si_wfe = False
    nc_coro.filter = filter

    # Add coronagraphic elements to nc_coro
    nc_coro.image_mask = fpm
    nc_coro.pupil_mask = lyot_stop

    # Generate the PSFs
    psf_default_hdu = nc.calc_psf(fov_pixels=int(im_size))
    psf_coro_hdu = nc_coro.calc_psf(fov_pixels=int(im_size))

    # Extract the PSFs to image arrays - the [1] extension gives me detector resolution
    psf_default = psf_default_hdu[1].data
    psf_coro = psf_coro_hdu[1].data

    print(psf_coro.shape)

    # Get maximum of PSF for the normalization later
    normp = np.max(psf_default)

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

        Aber = np.zeros([nb_seg, zern_max])
        Aber[i, wss_zern_nb-1] = nm_aber * 0.001      # aberration on the segment we're currenlty working on; 0.001 converts to microns

        #-# Crate OPD with aberrated segment(s)
        ote_coro._apply_hexikes_to_seg(seg, Aber[i,:])

        # If you want to display it:
        #ote_coro.display_opd()
        #plt.show()

        #-# Generate the coronagraphic PSF
        psf_endsim = nc_coro.calc_psf()
        psf_end = psf_endsim[1].data

        #-# Normalize coro PSF
        psf_end = psf_end / normp

        #-# Crop coro PSF and DH to same small size (like in analytical_model.py)

        #-# get end-to-end image in DH, calculate the contrast (mean) and put it in array

        #-# Create image from analytical model, calculate contrast (mean) and put in array

        #-# Save calibration vector

        
        # Extra comment form Lucie:
        ### Your calibration factor for each segment will be the ratio between the contrast from end-to-end simulation
        ### and PASTIS.
