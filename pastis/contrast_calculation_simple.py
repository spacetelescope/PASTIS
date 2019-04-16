"""This program compares contrast calculation from different methods:
WebbPSF coronagraph
Image-based PASTIS
Matrix-based PASTIS"""

import os
import time
import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from config import CONFIG_INI
import util_pastis as util
import image_pastis as impastis
import webbpsf_imaging as webbim


@u.quantity_input(rms=u.nm)
def pastis_vs_e2e(dir, matrix_mode="analytical", rms=1.*u.nm, im_pastis=False, plotting=False):
    """
    Calculate the contrast for an RMS WFE with image PASTIS, matrix PASTIS
    :param dir: data directory to use for matrix and calibration coefficients from
    :param matrix_mode: use 'analytical or 'numerical' matrix
    :param rms: RMS wavefront error in pupil to calculate contrast for; in NANOMETERS
    :param im_pastis: default False, whether to also calculate contrast from image PASTIS
    :param plotting: default False, whether to make a figure of E2E and PASTIS DH PSFs; works only if im_pastis=True;
                     for debugging mostly
    :return:
    """

    print("THIS ONLY WORKS FOR PISTON FOR NOW")

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Parameters
    dataDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), dir)
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
    wvln = CONFIG_INI.getfloat('filter', 'lambda') * u.nm
    filter = CONFIG_INI.get('filter', 'name')
    fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')         # focal plane mask
    lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')   # Lyot stop
    inner_wa = CONFIG_INI.getint('coronagraph', 'IWA')
    outer_wa = CONFIG_INI.getint('coronagraph', 'OWA')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    #real_samp = sampling * tel_size_px / im_size
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    # Import PASTIS matrix
    if matrix_mode == 'numerical':
        filename = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_numerical', filename + '.fits'))
    elif matrix_mode == 'analytical':
        filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_analytical', filename + '.fits'))

    # Create random aberration coefficients
    if zern_number == 1:   # piston
        aber = np.random.random([nb_seg])   # piston values in input units
        #print('PISTON ABERRATIONS:', aber)

        # Remove global piston
        aber -= np.mean(aber)

    else:
        raise("Other Zernikes than piston not implemented yet.")

    # Normalize to the RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    print("Calculated RMS:", calc_rms)

    # Modulo wavelength to get rid of phase wrapping.
    # The modulo operator on negative nuber is weird in Python,
    # this is a quick fix to account for that. It's ugly and
    # can definitely be done better.
    for i, k in enumerate(aber):
        if k < 0:
            aber[i] = -(np.abs(aber[i]) % wvln)
        else:
            aber[i] = aber[i] % wvln

    # Make equivalent aberration array that goes into the WebbPSF function
    Aber_WSS = np.zeros([nb_seg, zern_max])
    Aber_WSS[:,0] = aber.to(u.m).value   # index "0" works because we're using piston currently; convert to meters

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    print('Generating baseline PSF from WebbPSF - no coronagraph, no aberrations')
    psf_perfect = webbim.nircam_nocoro(filter, np.zeros_like(Aber_WSS))
    normp = np.max(psf_perfect)
    psf_perfect = psf_perfect / normp

    ### WEBBPSF
    print('Generating WebbPSF coro contrast')
    start_webb = time.time()
    # Set up NIRCam and coronagraph, get PSF
    psf_webbpsf = webbim.nircam_coro(filter, fpm, lyot_stop, Aber_WSS)
    psf_webbpsf = psf_webbpsf / normp
    # Create dark hole
    dh_area = util.create_dark_hole(psf_webbpsf, inner_wa, outer_wa, sampling)
    # Get the mean contrast from the WebbPSF coronagraph
    webb_dh_psf = psf_webbpsf * dh_area
    contrast_webbpsf = np.mean(webb_dh_psf[np.where(webb_dh_psf != 0)])
    end_webb = time.time()

    # Load in baseline contrast
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    contrast_base = float(np.loadtxt(os.path.join(dataDir, 'calibration', contrastname+'.txt')))

    ### IMAGE PASTIS
    contrast_am = None
    if im_pastis:
        print('Generating contrast from image-PASTIS')
        start_impastis = time.time()
        # Create calibrated image from analytical model
        psf_am, full_psf = impastis.analytical_model(zern_number, aber, cali=True)
        # Get the mean contrast from image PASTIS
        contrast_am = np.mean(psf_am[np.where(psf_am != 0)]) + contrast_base
        end_impastis = time.time()

    ### MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ratio = None
    if im_pastis:
        ratio = contrast_am / contrast_matrix

    # Outputs
    print('\n--- CONTRASTS: ---')
    print('Mean contrast from WebbPSF:', contrast_webbpsf)
    print('Mean contrast with image PASTIS:', contrast_am)
    print('Contrast from matrix PASTIS:', contrast_matrix)
    print('Ratio image PASTIS / matrix PASTIS:', ratio)

    print('\n--- RUNTIMES: ---')
    print('WebbPSF: ', end_webb-start_webb, 'sec =', (end_webb-start_webb)/60, 'min')
    if im_pastis:
        print('Image PASTIS: ', end_impastis-start_impastis, 'sec =', (end_impastis-start_impastis)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for contrast_calculation_simple.py: {} sec = {} min'.format(runtime, runtime/60))

    # Plot the PSFs, mostly used for debugging
    if im_pastis:
        if plotting:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title("E2E")
            plt.imshow(webb_dh_psf, norm=LogNorm())
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("PASTIS image")
            plt.imshow(psf_am, norm=LogNorm())
            plt.colorbar()
            plt.show()
            #TODO: check image rotation, I think there is a 90 degree difference in them

    return contrast_webbpsf, contrast_am, contrast_matrix


if __name__ == '__main__':

    WORKDIRECTORY = "active"    # you can chose here what data directory to work in
    matrix = "analytical"       # "analytical" or "numerical" PASTIS matrix to use
    total_rms = 1 * u.nm
    pastis_vs_e2e(WORKDIRECTORY, matrix_mode=matrix, rms=total_rms, im_pastis=True)
