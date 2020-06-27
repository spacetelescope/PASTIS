"""
This module has functions that compares the contrast calculation from different methods:
1. E2E coronagraph
2. (Image-based PASTIS)
3. Matrix-based PASTIS

All three methods are currently only supported for JWST, and you can pick between the analytical or numerical matrix.
HiCAT and LUVOIR only have an E2E vs numerical PASTIS comparison (1 + 3).
"""

import os
import time
import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import hcipy as hc

from config import CONFIG_INI
import util_pastis as util
import image_pastis as impastis
from e2e_simulators.luvoir_imaging import LuvoirAPLC


@u.quantity_input(rms=u.nm)
def contrast_jwst_ana_num(matdir, matrix_mode="analytical", rms=1. * u.nm, im_pastis=False, plotting=False):
    """
    Calculate the contrast for an RMS WFE with image PASTIS, matrix PASTIS
    :param matdir: data directory to use for matrix and calibration coefficients from
    :param matrix_mode: use 'analytical or 'numerical' matrix
    :param rms: RMS wavefront error in pupil to calculate contrast for; in NANOMETERS
    :param im_pastis: default False, whether to also calculate contrast from image PASTIS
    :param plotting: default False, whether to save E2E and PASTIS DH PSFs; works only if im_pastis=True
    :return:
    """
    from e2e_simulators import webbpsf_imaging as webbim

    print("THIS ONLY WORKS FOR PISTON FOR NOW")

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Parameters
    dataDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), matdir)
    which_tel = CONFIG_INI.get('telescope', 'name')
    nb_seg = CONFIG_INI.getint(which_tel, 'nb_subapertures')
    filter = CONFIG_INI.get(which_tel, 'filter_name')
    fpm = CONFIG_INI.get(which_tel, 'focal_plane_mask')         # focal plane mask
    lyot_stop = CONFIG_INI.get(which_tel, 'pupil_plane_stop')   # Lyot stop
    inner_wa = CONFIG_INI.getint(which_tel, 'IWA')
    outer_wa = CONFIG_INI.getint(which_tel, 'OWA')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    #real_samp = sampling * tel_size_px / im_size
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    # Import PASTIS matrix
    matrix_pastis = None
    if matrix_mode == 'numerical':
        filename = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_numerical', filename + '.fits'))
    elif matrix_mode == 'analytical':
        filename = 'PASTISmatrix_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        matrix_pastis = fits.getdata(os.path.join(dataDir, 'matrix_analytical', filename + '.fits'))

    # Create random aberration coefficients
    aber = np.random.random([nb_seg])   # piston values in input units
    #print('PISTON ABERRATIONS:', aber)

    # Normalize to the RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    print("Calculated RMS:", calc_rms)

    # Remove global piston
    aber -= np.mean(aber)

    # Make equivalent aberration array that goes into the WebbPSF function
    Aber_WSS = np.zeros([nb_seg, zern_max])
    Aber_WSS[:,0] = aber.to(u.m).value   # index "0" works because we're using piston currently; convert to meters

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    print('Generating baseline PSF from E2E - no coronagraph, no aberrations')
    psf_perfect = webbim.nircam_nocoro(filter, np.zeros_like(Aber_WSS))
    normp = np.max(psf_perfect)
    psf_perfect = psf_perfect / normp

    ### WEBBPSF
    print('Generating E2E coro contrast')
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

    #TODO: save plots of phase on segmented pupil

    # Load in baseline contrast
    contrastname = 'base-contrast_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    coro_floor = float(np.loadtxt(os.path.join(dataDir, 'calibration', contrastname+'.txt')))

    ### IMAGE PASTIS
    contrast_am = np.nan
    if im_pastis:
        print('Generating contrast from image-PASTIS')
        start_impastis = time.time()
        # Create calibrated image from analytical model
        psf_am, full_psf = impastis.analytical_model(zern_number, aber, cali=True)
        # Get the mean contrast from image PASTIS
        contrast_am = np.mean(psf_am[np.where(psf_am != 0)]) + coro_floor
        end_impastis = time.time()

    ### MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ratio = None
    if im_pastis:
        ratio = contrast_am / contrast_matrix

    # Outputs
    print('\n--- CONTRASTS: ---')
    print('Mean contrast from E2E:', contrast_webbpsf)
    print('Mean contrast with image PASTIS:', contrast_am)
    print('Contrast from matrix PASTIS:', contrast_matrix)
    print('Ratio image PASTIS / matrix PASTIS:', ratio)

    print('\n--- RUNTIMES: ---')
    print('E2E: ', end_webb-start_webb, 'sec =', (end_webb-start_webb)/60, 'min')
    if im_pastis:
        print('Image PASTIS: ', end_impastis-start_impastis, 'sec =', (end_impastis-start_impastis)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for contrast_calculation_simple.py: {} sec = {} min'.format(runtime, runtime/60))

    # Save the PSFs
    if im_pastis:
        if plotting:

            # As fits files
            util.write_fits(util.zoom_cen(webb_dh_psf, psf_am.shape[0]/2), os.path.join(dataDir, 'results',
                            'dh_images_'+matrix_mode, '{:.2e}'.format(rms.value)+str(rms.unit)+'RMS_e2e.fits'))
            util.write_fits(psf_am, os.path.join(dataDir, 'results', 'dh_images_'+matrix_mode,
                                                 '{:.2e}'.format(rms.value)+str(rms.unit)+'RMS_am.fits'))

            # As PDF plot
            plt.clf()
            plt.figure()
            plt.suptitle('{:.2e}'.format(rms.value) + str(rms.unit) + " RMS")
            plt.subplot(1, 2, 1)
            plt.title("E2E")
            plt.imshow(util.zoom_cen(webb_dh_psf, psf_am.shape[0]/2), norm=LogNorm())
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("PASTIS image")
            plt.imshow(psf_am, norm=LogNorm())
            plt.colorbar()
            plt.savefig(os.path.join(dataDir, 'results', 'dh_images_'+matrix_mode,
                                     '{:.2e}'.format(rms.value)+'DH_PSFs.pdf'))
            #TODO: check image rotation, I think there is a 90 degree difference in them for the JWST simulations

    return contrast_webbpsf, contrast_am, contrast_matrix


def contrast_hicat_num(matrix_dir, matrix_mode='hicat', rms=1*u.nm):
    """
    Compute the contrast for a random IrisAO mislignment on the HiCAT simulator.
    :param matrix_dir: str, directory of saved matrix
    :param matrix_mode: str, analytical or numerical; currently only numerical supported
    :param rms: astropy quantity, rms wfe to be put randomly on the SM
    :return: 2x float, E2E and matrix contrast
    """
    import hicat.simulators

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Parameters
    nb_seg = CONFIG_INI.getint('HiCAT', 'nb_subapertures')
    iwa = CONFIG_INI.getfloat('HiCAT', 'IWA')
    owa = CONFIG_INI.getfloat('HiCAT', 'OWA')

    # Import numerical PASTIS matrix for HiCAT sim
    filename = 'PASTISmatrix_num_HiCAT_piston_Noll1'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients
    aber = np.random.random([nb_seg])   # piston values in input units
    print('PISTON ABERRATIONS:', aber)

    # Normalize to the RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    print("Calculated RMS:", calc_rms)

    # Remove global piston
    aber -= np.mean(aber)

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    print('Generating baseline PSF from E2E - no coronagraph, no aberrations')
    hc = hicat.simulators.hicat_sim.HICAT_Sim()
    hc.iris_ao = 'iris_ao'
    hc.apodizer = 'cnt1_apodizer'
    hc.lyot_stop = 'cnt1_apodizer_lyot_stop'
    hc.include_fpm = False

    psf_perfect = hc.calc_psf(display=False, return_intermediates=False)
    normp = np.max(psf_perfect[0].data)
    #psf_perfect = psf_perfect[0].data / normp   don't actually need the perfect PSF

    ### HiCAT sim
    start_e2e = time.time()
    # Set up the HiCAT simulator, get PSF
    hc.apodizer = 'cnt1_apodizer'
    hc.lyot_stop = 'cnt1_apodizer_lyot_stop'
    hc.include_fpm = True

    # Calculate coro PSF without aberrations
    psf_coro = hc.calc_psf(display=False, return_intermediates=False)
    psf_coro = psf_coro[0].data / normp


    print('Calculating E2E contrast...')
    # Put aberration on Iris AO
    for nseg in range(nb_seg):
        hc.iris_dm.set_actuator(nseg+1, aber[nseg], 0, 0)

    psf_hicat = hc.calc_psf(display=False, return_intermediates=False)
    psf_hicat = psf_hicat[0].data / normp

    # Create DH
    dh_mask = util.create_dark_hole(psf_hicat, iwa=iwa, owa=owa, samp=13 / 4)
    # Get the mean contrast
    hicat_dh_psf = psf_hicat * dh_mask
    contrast_hicat = np.mean(hicat_dh_psf[np.where(hicat_dh_psf != 0)])
    end_e2e = time.time()

    ###
    # Calculate coronagraph contrast floor
    baseline_dh = psf_coro * dh_mask
    coro_floor = np.mean(baseline_dh[np.where(baseline_dh != 0)])

    ## MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    print('\n--- CONTRASTS: ---')
    print('Mean contrast from E2E:', contrast_hicat)
    print('Contrast from matrix PASTIS:', contrast_matrix)

    print('\n--- RUNTIMES: ---')
    print('E2E: ', end_e2e-start_e2e, 'sec =', (end_e2e-start_e2e)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for contrast_calculation_simple.py: {} sec = {} min'.format(runtime, runtime/60))

    return contrast_hicat, contrast_matrix


def contrast_luvoir_num(apodizer_choice, matrix_dir, rms=1*u.nm):
    """
    Compute the contrast for a random segmented mirror misalignment on the LUVOIR simulator.
    :param matrix_dir: str, directory of saved matrix
    :param rms: astropy quantity (e.g. m or nm), WFE rms (OPD) to be put randomly over the entire segmented mirror
    :return: 2x float, E2E and matrix contrast
    """

    # Keep track of time
    start_time = time.time()

    # Parameters
    nb_seg = CONFIG_INI.getint('LUVOIR', 'nb_subapertures')
    sampling = 4

    # Import numerical PASTIS matrix for HiCAT sim
    filename = 'PASTISmatrix_num_piston_Noll1'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients
    aber = np.random.random([nb_seg])   # piston values in input units
    print('PISTON ABERRATIONS:', aber)

    # Normalize to the WFE RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    print("Calculated WFE RMS:", calc_rms)

    # Remove global piston
    aber -= np.mean(aber)

    # Coronagraph parameters
    # The LUVOIR STDT delivery in May 2018 included three different apodizers
    # we can work with, so I will implement an easy way of making a choice between them.
    design = apodizer_choice
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')

    # Instantiate LUVOIR telescope with APLC
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    # and coro PSF without aberrations
    start_e2e = time.time()
    print('Generating baseline PSF from E2E - no coronagraph, no aberrations')
    print('Also generating coro PSF without aberrations')
    psf_perfect, ref = luvoir.calc_psf(ref=True)
    normp = np.max(ref)
    psf_coro = psf_perfect / normp

    print('Calculating E2E contrast...')
    # Put aberrations on segmented mirror
    for nseg in range(nb_seg):
        luvoir.set_segment(nseg+1, aber[nseg].to(u.m).value/2, 0, 0)

    psf_luvoir = luvoir.calc_psf()
    psf_luvoir /= normp

    # Create DH
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    # Get the mean contrast
    dh_intensity = psf_luvoir * dh_mask
    contrast_luvoir = np.mean(dh_intensity[np.where(dh_intensity != 0)])
    end_e2e = time.time()

    ###
    # Calculate coronagraph contrast floor
    baseline_dh = psf_coro * dh_mask
    coro_floor = np.mean(baseline_dh[np.where(baseline_dh != 0)])
    print('Baseline contrast: {}'.format(coro_floor))

    ## MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    print('\n--- CONTRASTS: ---')
    print('Mean contrast from E2E:', contrast_luvoir)
    print('Contrast from matrix PASTIS:', contrast_matrix)

    print('\n--- RUNTIMES: ---')
    print('E2E: ', end_e2e-start_e2e, 'sec =', (end_e2e-start_e2e)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for contrast_calculation_simple.py: {} sec = {} min'.format(runtime, runtime/60))

    return contrast_luvoir, contrast_matrix


if __name__ == '__main__':

    # Test JWST
    # WORKDIRECTORY = "active"    # you can chose here what data directory to work in
    # matrix = "analytical"       # "analytical" or "numerical" PASTIS matrix to use
    # total_rms = 100 * u.nm
    # contrast_jwst_ana_num(WORKDIRECTORY, matrix_mode=matrix, rms=total_rms, im_pastis=True)

    # Test HiCAT
    #c_e2e, c_matrix = contrast_hicat_num(matrix_dir='/Users/ilaginja/Documents/Git/PASTIS/Jupyter Notebooks/HiCAT', rms=10*u.nm)
    #c_e2e, c_matrix = contrast_hicat_num(matrix_dir='/Users/ilaginja/Documents/data_from_repos/pastis_data/active/matrix_numerical', rms=10*u.nm)

    # Test LUVOIR
    c_e2e, c_matrix = contrast_luvoir_num(matrix_dir='/Users/ilaginja/Documents/data_from_repos/pastis_data/active/matrix_numerical',
                                         rms=1 * u.nm)
