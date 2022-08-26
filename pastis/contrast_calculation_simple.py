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
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pastis.config import CONFIG_PASTIS
from pastis.simulators.hicat_imaging import set_up_hicat
from pastis.simulators.luvoir_imaging import LuvoirAPLC
import pastis.simulators.webbpsf_imaging as webbpsf_imaging
import pastis.analytical_pastis.image_pastis as impastis
import pastis.util as util

log = logging.getLogger()


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
    from simulators import webbpsf_imaging as webbim

    log.warning("THIS ONLY WORKS FOR PISTON FOR NOW")

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Parameters
    dataDir = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), matdir)
    which_tel = CONFIG_PASTIS.get('telescope', 'name')
    nb_seg = CONFIG_PASTIS.getint(which_tel, 'nb_subapertures')
    filter = CONFIG_PASTIS.get(which_tel, 'filter_name')
    fpm = CONFIG_PASTIS.get(which_tel, 'focal_plane_mask')         # focal plane mask
    lyot_stop = CONFIG_PASTIS.get(which_tel, 'pupil_plane_stop')   # Lyot stop
    inner_wa = CONFIG_PASTIS.getint(which_tel, 'IWA')
    outer_wa = CONFIG_PASTIS.getint(which_tel, 'OWA')
    tel_size_px = CONFIG_PASTIS.getint('numerical', 'tel_size_px')
    sampling = CONFIG_PASTIS.getfloat(which_tel, 'sampling')
    #real_samp = sampling * tel_size_px / im_size
    zern_number = CONFIG_PASTIS.getint('calibration', 'local_zernike')
    zern_mode = util.ZernikeMode(zern_number)
    zern_max = CONFIG_PASTIS.getint('zernikes', 'max_zern')

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
    #log.info(f'PISTON ABERRATIONS: {aber}')

    # Normalize to the RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    log.info(f"Calculated RMS: {calc_rms}")

    # Remove global piston
    aber -= np.mean(aber)

    # Make equivalent aberration array that goes into the WebbPSF function
    Aber_WSS = np.zeros([nb_seg, zern_max])
    Aber_WSS[:,0] = aber.to(u.m).value   # index "0" works because we're using piston currently; convert to meters

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    log.info('Generating baseline PSF from E2E - no coronagraph, no aberrations')
    psf_perfect = webbim.nircam_nocoro(filter, np.zeros_like(Aber_WSS))
    normp = np.max(psf_perfect)
    psf_perfect = psf_perfect / normp

    ### WEBBPSF
    log.info('Generating E2E coro contrast')
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
        log.info('Generating contrast from image-PASTIS')
        start_impastis = time.time()
        # Create calibrated image from analytical model
        psf_am, full_psf = impastis.analytical_model(zern_number, aber, cali=True)
        # Get the mean contrast from image PASTIS
        contrast_am = np.mean(psf_am[np.where(psf_am != 0)]) + coro_floor
        end_impastis = time.time()

    ### MATRIX PASTIS
    log.info('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ratio = None
    if im_pastis:
        ratio = contrast_am / contrast_matrix

    # Outputs
    log.info('\n--- CONTRASTS: ---')
    log.info(f'Mean contrast from E2E: {contrast_webbpsf}')
    log.info(f'Mean contrast with image PASTIS: {contrast_am}')
    log.info(f'Contrast from matrix PASTIS: {contrast_matrix}')
    log.info(f'Ratio image PASTIS / matrix PASTIS: {ratio}')

    log.info('\n--- RUNTIMES: ---')
    log.info(f'E2E: {end_webb-start_webb}sec = {(end_webb-start_webb)/60}min')
    if im_pastis:
        log.info(f'Image PASTIS: {end_impastis-start_impastis}sec = {(end_impastis-start_impastis)/60}min')
    log.info(f'Matrix PASTIS: {end_matrixpastis-start_matrixpastis}sec = {(end_matrixpastis-start_matrixpastis)/60}min')

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for contrast_calculation_simple.py: {runtime} sec = {runtime/60} min')

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


def contrast_hicat_num(coro_floor, norm, matrix_dir, rms=1*u.nm):
    """
    Compute the contrast for a random IrisAO misalignment on the HiCAT simulator.

    :param coro_floor: float, coronagraph contrast floor
    :param norm: float, normalization factor for PSFs: peak of unaberrated direct PSF
    :param matrix_dir: str, directory of saved matrix
    :param rms: astropy quantity, rms wfe to be put randomly on the SM
    :return: E2E and matrix contrast, both floats
    """

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Parameters
    nb_seg = CONFIG_PASTIS.getint('HiCAT', 'nb_subapertures')
    iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
    owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')

    # Import numerical PASTIS matrix
    filename = 'pastis_matrix'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients on segments, scaled to total rms
    aber = util.create_random_rms_values(nb_seg, rms)

    ### E2E HiCAT sim
    start_e2e = time.time()

    # Set HiCAT simulator to coro mode
    hicat_sim = set_up_hicat(apply_continuous_dm_maps=True)
    hicat_sim.include_fpm = True

    log.info('Calculating E2E contrast...')
    # Put aberration on Iris AO
    for nseg in range(nb_seg):
        hicat_sim.iris_dm.set_actuator(nseg, aber[nseg], 0, 0)  # TODO: Test adding *u.nm in "aber[nseg]*u.nm"

    psf_hicat = hicat_sim.calc_psf(display=False, return_intermediates=False)
    psf_hicat = psf_hicat[0].data / norm

    # Create DH
    dh_mask = util.create_dark_hole(psf_hicat, iwa=iwa, owa=owa, samp=sampling)
    # Get the mean contrast
    contrast_hicat = util.dh_mean(psf_hicat, dh_mask)
    end_e2e = time.time()

    ## MATRIX PASTIS
    log.info('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    log.info('\n--- CONTRASTS: ---')
    log.info(f'Mean contrast from E2E: {contrast_hicat}')
    log.info(f'Contrast from matrix PASTIS: {contrast_matrix}')

    log.info('\n--- RUNTIMES: ---')
    log.info(f'E2E: {end_e2e-start_e2e}sec = {(end_e2e-start_e2e)/60}min')
    log.info(f'Matrix PASTIS: {end_matrixpastis-start_matrixpastis}sec = {(end_matrixpastis-start_matrixpastis)/60}min')

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for contrast_calculation_simple.py: {runtime} sec = {runtime/60} min')

    return contrast_hicat, contrast_matrix


def contrast_luvoir_num(coro_floor, norm, design, matrix_dir, rms=1*u.nm):
    """
    Compute the contrast for a random segmented mirror misalignment on the LUVOIR simulator.

    :param coro_floor: float, coronagraph contrast floor
    :param norm: float, normalization factor for PSFs: peak of unaberrated direct PSF
    :param matrix_dir: str, directory of saved matrix
    :param rms: astropy quantity (e.g. m or nm), WFE rms (OPD) to be put randomly over the entire segmented mirror
    :return: 2x float, E2E and matrix contrast
    """

    # Keep track of time
    start_time = time.time()

    # Parameters
    nb_seg = CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures')
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')

    # Import numerical PASTIS matrix
    filename = 'pastis_matrix'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients on segments, scaled to total rms
    aber = util.create_random_rms_values(nb_seg, rms)

    start_e2e = time.time()
    # Coronagraph parameters
    # The LUVOIR STDT delivery in May 2018 included three different apodizers
    # we can work with, so I will implement an easy way of making a choice between them.
    optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))

    # Instantiate LUVOIR telescope with APLC
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    log.info('Calculating E2E contrast...')
    # Put aberrations on segmented mirror
    for nseg in range(nb_seg):
        luvoir.set_segment(nseg+1, aber[nseg].to(u.m).value/2, 0, 0)
    psf_luvoir = luvoir.calc_psf()
    psf_luvoir /= norm

    # Get the mean contrast
    contrast_luvoir = util.dh_mean(psf_luvoir, luvoir.dh_mask)
    end_e2e = time.time()

    ## MATRIX PASTIS
    log.info('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    log.info('\n--- CONTRASTS: ---')
    log.info(f'Mean contrast from E2E: {contrast_luvoir}')
    log.info(f'Contrast from matrix PASTIS: {contrast_matrix}')

    log.info('\n--- RUNTIMES: ---')
    log.info(f'E2E: {end_e2e-start_e2e}sec = {(end_e2e-start_e2e)/60}min')
    log.info(f'Matrix PASTIS: {end_matrixpastis-start_matrixpastis}sec = {(end_matrixpastis-start_matrixpastis)/60}min')

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for contrast_calculation_simple.py: {runtime} sec = {runtime/60} min')

    return contrast_luvoir, contrast_matrix


def contrast_jwst_num(coro_floor, norm, matrix_dir, rms=50*u.nm):
    """
    Compute the contrast for a random segmented OTE misalignment on the JWST simulator.

    :param coro_floor: float, coronagraph contrast floor
    :param norm: float, normalization factor for PSFs: peak of unaberrated direct PSF
    :param matrix_dir: str, directory of saved matrix
    :param rms: astropy quantity (e.g. m or nm), WFE rms (OPD) to be put randomly over the entire segmented mirror
    :return: 2x float, E2E and matrix contrast
    """
    # Keep track of time
    start_time = time.time()

    # Parameters
    nb_seg = CONFIG_PASTIS.getint('JWST', 'nb_subapertures')
    iwa = CONFIG_PASTIS.getfloat('JWST', 'IWA')
    owa = CONFIG_PASTIS.getfloat('JWST', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('JWST', 'sampling')

    # Import numerical PASTIS matrix
    filename = 'pastis_matrix'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients on segments, scaled to total rms
    aber = util.create_random_rms_values(nb_seg, rms)

    ### E2E JWST sim
    start_e2e = time.time()

    jwst_sim = webbpsf_imaging.set_up_nircam()
    jwst_sim[0].image_mask = CONFIG_PASTIS.get('JWST', 'focal_plane_mask')

    log.info('Calculating E2E contrast...')
    # Put aberration on OTE
    jwst_sim[1].zero()
    for nseg in range(nb_seg):    # TODO: there is probably a single function that puts the aberration on the OTE at once
        seg_num = webbpsf_imaging.WSS_SEGS[nseg].split('-')[0]
        jwst_sim[1].move_seg_local(seg_num, piston=aber[nseg].value/2, trans_unit='nm')   # this function works with physical motions, meaning the piston is in surface

    image = jwst_sim[0].calc_psf(nlambda=1)
    psf_jwst = image[0].data / norm

    # Create DH
    dh_mask = util.create_dark_hole(psf_jwst, iwa=iwa, owa=owa, samp=sampling)
    # Get the mean contrast
    contrast_jwst = util.dh_mean(psf_jwst, dh_mask)
    end_e2e = time.time()

    ## MATRIX PASTIS
    log.info('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    log.info('\n--- CONTRASTS: ---')
    log.info(f'Mean contrast from E2E: {contrast_jwst}')
    log.info(f'Contrast from matrix PASTIS: {contrast_matrix}')

    log.info('\n--- RUNTIMES: ---')
    log.info(f'E2E: {end_e2e-start_e2e}sec = {(end_e2e-start_e2e)/60}min')
    log.info(f'Matrix PASTIS: {end_matrixpastis-start_matrixpastis}sec = {(end_matrixpastis-start_matrixpastis)/60}min')

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for contrast_calculation_simple.py: {runtime} sec = {runtime/60} min')

    return contrast_jwst, contrast_matrix


def contrast_rst_num(coro_floor, norm, matrix_dir, rms=50*u.nm):
    """
    Compute the contrast for a random aberration over all DM actuators in the RST simulator.

    :param coro_floor: float, coronagraph contrast floor
    :param norm: float, normalization factor for PSFs: peak of unaberrated direct PSF
    :param matrix_dir: str, directory of saved matrix
    :param rms: astropy quantity (e.g. m or nm), WFE rms (OPD) to be put randomly over the entire continuous mirror
    :return: 2x float, E2E and matrix contrast
    """
    # Keep track of time
    start_time = time.time()

    # Parameters
    total_seg = CONFIG_PASTIS.getint('RST', 'nb_subapertures')

    # Import numerical PASTIS matrix
    filename = 'pastis_matrix'
    matrix_pastis = fits.getdata(os.path.join(matrix_dir, filename + '.fits'))

    # Create random aberration coefficients on segments, scaled to total rms
    aber = util.create_random_rms_values(total_seg, rms)

    ### E2E RST sim
    start_e2e = time.time()

    rst_sim = webbpsf_imaging.set_up_cgi()
    rst_sim.fpm = CONFIG_PASTIS.get('RST', 'fpm')
    nb_actu = rst_sim.nbactuator
    iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
    owa = CONFIG_PASTIS.getfloat('RST', 'OWA')

    # Put aberration on OTE
    rst_sim.dm1.flatten()
    for nseg in range(total_seg):
        actu_x, actu_y = util.seg_to_dm_xy(nb_actu, nseg)
        rst_sim.dm1.set_actuator(actu_x, actu_y, aber[nseg].value*u.nm)

    image = rst_sim.calc_psf(nlambda=1, fov_arcsec=1.6)
    psf = image[0].data / norm

    # Get the mean contrast
    rst_sim.working_area(im=psf, inner_rad=iwa, outer_rad=owa)
    dh_mask = rst_sim.WA
    contrast_rst = util.dh_mean(psf, dh_mask)
    end_e2e = time.time()

    ## MATRIX PASTIS
    log.info('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + coro_floor   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    ## Outputs
    log.info('\n--- CONTRASTS: ---')
    log.info(f'Mean contrast from E2E: {contrast_rst}')
    log.info(f'Contrast from matrix PASTIS: {contrast_matrix}')

    log.info('\n--- RUNTIMES: ---')
    log.info(f'E2E: {end_e2e-start_e2e}sec = {(end_e2e-start_e2e)/60}min')
    log.info(f'Matrix PASTIS: {end_matrixpastis-start_matrixpastis}sec = {(end_matrixpastis-start_matrixpastis)/60}min')

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for contrast_calculation_simple.py: {runtime} sec = {runtime/60} min')

    return contrast_rst, contrast_matrix


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
