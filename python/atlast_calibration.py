"""
Translation of atlast_calibration.pro, which makes the calibration files for PASTIS.
"""
import os
import numpy as np
import webbpsf
import poppy.zernike as zern

from python.config import CONFIG_INI


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
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    nm_aber = 1.    # [nm] amplitude of aberration
    zern_number = 2

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
    psf_default = nc.calc_psf(fov_pixels=int(im_size/4))
    psf_coro = nc_coro.calc_psf(fov_pixels=int(im_size/4))

    # Extract the PSFs to image arrays
    psf_default = psf_default[0].data
    psf_coro = psf_coro[0].data

    print(psf_coro.shape)

    # Get maximum of PSF for the normalization later
    normp = np.max(psf_default)

    # Create the arrays to hold the contrast values from the iterations
    contrastAPLC_vec_int = np.zeros([nb_seg+1])
    contrastAM_vec_int = np.zeros([nb_seg+1])

    # Loop over each individual segment, putting always the same aberration on
    for i in range(nb_seg+1):
        print('Working on segment ' + str(i) + '/' + str(nb_seg+1))

        Aber = np.zeros([nb_seg+1])
        Aber[i] = nm_aber         # aberration on the segment we're currenlty working on
        Aber[nb_seg/2+1] = 0.     # 0 nm aberration on the central segment

        #-# Crate OPD with only one aberrated segment

        #-# Generate the coronagraphic PSF

        #-# Normalize coro PSF

        #-# Crop coro PSF and DH to same small size (like in analytical_model.py)

        #-# get end-to-end image in DH, calculate the contrast (mean) and put it in array

        #-# Create image from analyticla model, calculate contrast (mean) and put in array

        #-# Save calibration vector

        
        # Extra comment form Lucie:
        ### Your calibration factor for each segment will be the ratio between the contrast from en-to-end simulation
        ### and PASTIS.
