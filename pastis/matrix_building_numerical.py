"""
This module contains functions that construct the matrix M for PASTIS *NUMERICALLY FROM THE RESPECTIVE E2E SIMULATOR*
 and saves it.

 Currently supported:
 JWST
 LUVOIR
 HiCAT
 """

import os
import time
import functools
import shutil
import astropy.units as u
import logging
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import hcipy
import webbpsf

from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.e2e_simulators.hicat_imaging import set_up_hicat
from pastis.e2e_simulators.luvoir_imaging import LuvoirAPLC
from pastis.e2e_simulators.webbpsf_imaging import set_up_nircam
import pastis.plotting as ppl

log = logging.getLogger()


def num_matrix_jwst():
    """
    Generate a numerical PASTIS matrix for a JWST coronagraph.
    -- Depracated function, the LUVOIR PASTIS matrix is better calculated with num_matrix_multiprocess(), which can
    do this for your choice of one of the implemented instruments (LUVOIR, HiCAT, JWST). --

    All inputs are read from the (local) configfile and saved to the specified output directory.
    """

    import webbpsf
    from e2e_simulators import webbpsf_imaging as webbim
    # Set WebbPSF environment variable
    os.environ['WEBBPSF_PATH'] = CONFIG_PASTIS.get('local', 'webbpsf_data_path')

    # Keep track of time
    start_time = time.time()   # runtime is currently around 21 minutes
    log.info('Building numerical matrix for JWST\n')

    # Parameters
    overall_dir = util.create_data_path(CONFIG_PASTIS.get('local', 'local_data_path'), telescope='jwst')
    resDir = os.path.join(overall_dir, 'matrix_numerical')
    which_tel = CONFIG_PASTIS.get('telescope', 'name')
    nb_seg = CONFIG_PASTIS.getint(which_tel, 'nb_subapertures')
    im_size_e2e = CONFIG_PASTIS.getint('numerical', 'im_size_px_webbpsf')
    inner_wa = CONFIG_PASTIS.getint(which_tel, 'IWA')
    outer_wa = CONFIG_PASTIS.getint(which_tel, 'OWA')
    sampling = CONFIG_PASTIS.getfloat(which_tel, 'sampling')
    fpm = CONFIG_PASTIS.get(which_tel, 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_PASTIS.get(which_tel, 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_PASTIS.get(which_tel, 'filter_name')
    wfe_aber = CONFIG_PASTIS.getfloat(which_tel, 'calibration_aberration') * u.nm
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_PASTIS.getint('zernikes', 'max_zern')
    zern_number = CONFIG_PASTIS.getint('calibration', 'local_zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling
    wss_zern_nb = util.noll_to_wss(zern_number)                     # Convert from Noll to WSS framework

    # Create necessary directories if they don't exist yet
    os.makedirs(overall_dir, exist_ok=True)
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(os.path.join(resDir, 'OTE_images'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'psfs'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'darkholes'), exist_ok=True)

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
    contrast_matrix = np.zeros([nb_seg, nb_seg])   # Generate empty matrix
    all_psfs = []
    all_dhs = []
    all_contrasts = []

    log.info(f'wfe_aber: {wfe_aber}')

    for i in range(nb_seg):
        for j in range(nb_seg):

            log.info(f'\nSTEP: {i+1}-{j+1} / {nb_seg}-{nb_seg}')

            # Get names of segments, they're being addressed by their names in the ote functions.
            seg_i = wss_segs[i].split('-')[0]
            seg_j = wss_segs[j].split('-')[0]

            # Put the aberration on the correct segments
            Aber_WSS = np.zeros([nb_seg, zern_max])         # The Zernikes here will be filled in the WSS order!!!
                                                            # Because it goes into _apply_hexikes_to_seg().
            Aber_WSS[i, wss_zern_nb - 1] = wfe_aber.to(u.m).value    # Aberration on the segment we're currently working on;
                                                            # convert to meters; -1 on the Zernike because Python starts
                                                            # numbering at 0.
            Aber_WSS[j, wss_zern_nb - 1] = wfe_aber.to(u.m).value    # same for other segment

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

            # Save OPD images for testing
            opd_name = f'opd_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{i+1}-{j+1}'
            plt.clf()
            ote_coro.display_opd()
            plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

            log.info('Calculating WebbPSF image')
            image = nc_coro.calc_psf(fov_pixels=int(im_size_e2e), oversample=1, nlambda=1)
            psf = image[0].data / normp

            # Save WebbPSF image to disk
            filename_psf = f'psf_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{i+1}-{j+1}'
            util.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'), header=None, metadata=None)
            all_psfs.append(psf)

            log.info('Calculating mean contrast in dark hole')
            dh_intensity = psf * dh_area
            contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
            log.info(f'contrast: {contrast}')

            # Save DH image to disk and put current contrast in list
            filename_dh = f'dh_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{i+1}-{j+1}'
            util.write_fits(dh_intensity, os.path.join(resDir, 'darkholes', filename_dh + '.fits'), header=None, metadata=None)
            all_dhs.append(dh_intensity)
            all_contrasts.append(contrast)

            # Fill according entry in the matrix
            contrast_matrix[i,j] = contrast

    # Transform saved lists to arrays
    all_psfs = np.array(all_psfs)
    all_dhs = np.array(all_dhs)
    all_contrasts = np.array(all_contrasts)

    # Filling the off-axis elements
    matrix_two_N = np.copy(contrast_matrix)      # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(contrast_matrix)     # This will be the final PASTIS matrix.

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                log.info(f'Off-axis for i{i+1}-j{j+1}: {matrix_off_val}')

    # Normalize matrix for the input aberration
    matrix_pastis /= np.square(wfe_aber.value)

    # Save matrix to file
    filename_matrix = f'PASTISmatrix_num_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}'
    util.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'), header=None, metadata=None)
    log.info(f'Matrix saved to: {os.path.join(resDir, filename_matrix + ".fits")}')

    # Save the PSF and DH image *cubes* as well (as opposed to each one individually)
    util.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube.fits'), header=None, metadata=None)
    util.write_fits(all_dhs, os.path.join(resDir, 'darkholes', 'dh_cube.fits'), header=None, metadata=None)
    np.savetxt(os.path.join(resDir, 'pair-wise_contrasts.txt'), all_contrasts, fmt='%e')

    # Tell us how long it took to finish.
    end_time = time.time()
    log.info(f'Runtime for matrix_building.py: {end_time - start_time}sec = {(end_time - start_time) / 60}min')
    log.info(f'Data saved to {resDir}')

    # -- Runtime notes: --
    #
    # im_size = 128
    # oversampling = 1
    # nb_seg = 18
    # runtime = 20 min


def num_matrix_luvoir(design, savepsfs=False, saveopds=True):
    """
    Generate a numerical PASTIS matrix for a LUVOIR A coronagraph.
    -- Depracated function, the LUVOIR PASTIS matrix is better calculated with num_matrix_multiprocess(), which can
    do this for your choice of one of the implemented instruments (LUVOIR, HiCAT, JWST). --

    All inputs are read from the (local) configfile and saved to the specified output directory.
    The LUVOIR STDT delivery in May 2018 included three different apodizers
    we can work with, you pick which of the three you want with the 'design' parameter.
    :param design: string, what coronagraph design to use - 'small', 'medium' or 'large'
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files, additionally to the
                     total PSF cube. If False, the total cube will still get saved at the very end of the script.
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :return overall_dir: string, experiment directory
    """

    # Keep track of time
    start_time = time.time()

    ### Parameters

    # System parameters
    overall_dir = util.create_data_path(CONFIG_PASTIS.get('local', 'local_data_path'), telescope='luvoir-'+design)
    os.makedirs(overall_dir, exist_ok=True)
    resDir = os.path.join(overall_dir, 'matrix_numerical')

    # Create necessary directories if they don't exist yet
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(os.path.join(resDir, 'OTE_images'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'psfs'), exist_ok=True)

    # Set up logger
    util.setup_pastis_logging(resDir, f'pastis_matrix_{design}')
    log.info('Building numerical matrix for LUVOIR\n')

    # Read calibration aberration
    zern_number = CONFIG_PASTIS.getint('calibration', 'local_zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    # General telescope parameters
    nb_seg = CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures')
    wvln = CONFIG_PASTIS.getfloat('LUVOIR', 'lambda') * 1e-9  # m
    diam = CONFIG_PASTIS.getfloat('LUVOIR', 'diameter')  # m
    wfe_aber = CONFIG_PASTIS.getfloat('LUVOIR', 'calibration_aberration') * 1e-9   # m

    # Image system parameters
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')

    # Record some of the defined parameters
    log.info(f'LUVOIR apodizer design: {design}')
    log.info(f'Wavelength: {wvln} m')
    log.info(f'Telescope diameter: {diam} m')
    log.info(f'Number of segments: {nb_seg}')
    log.info(f'Sampling: {sampling} px per lambda/D')
    log.info(f'wfe_aber: {wfe_aber} m')

    #  Copy configfile to resulting matrix directory
    util.copy_config(resDir)

    ### Instantiate Luvoir telescope with chosen apodizer design
    optics_input = CONFIG_PASTIS.get('LUVOIR', 'optics_path')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    ### Reference images for contrast normalization and coronagraph floor
    unaberrated_coro_psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    norm = np.max(ref)

    dh_intensity = (unaberrated_coro_psf / norm) * luvoir.dh_mask
    contrast_floor = np.mean(dh_intensity[np.where(luvoir.dh_mask != 0)])
    log.info(f'contrast floor: {contrast_floor}')

    ### Generating the PASTIS matrix and a list for all contrasts
    contrast_matrix = np.zeros([nb_seg, nb_seg])   # Generate empty matrix
    all_psfs = []
    all_contrasts = []

    for i in range(nb_seg):
        for j in range(nb_seg):

            log.info(f'\nSTEP: {i+1}-{j+1} / {nb_seg}-{nb_seg}')

            # Put aberration on correct segments. If i=j, apply only once!
            luvoir.flatten()
            luvoir.set_segment(i+1, wfe_aber/2, 0, 0)
            if i != j:
                luvoir.set_segment(j+1, wfe_aber/2, 0, 0)

            log.info('Calculating coro image...')
            image, inter = luvoir.calc_psf(ref=False, display_intermediate=False, return_intermediate='intensity')
            # Normalize PSF by reference image
            psf = image / norm
            all_psfs.append(psf.shaped)

            # Save image to disk
            if savepsfs:   # TODO: I might want to change this to matplotlib images since I save the PSF cube anyway.
                filename_psf = f'psf_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{i+1}-{j+1}'
                hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

            # Save OPD images for testing
            if saveopds:
                opd_name = f'opd_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{i+1}-{j+1}'
                plt.clf()
                hcipy.imshow_field(inter['seg_mirror'], mask=luvoir.aperture, cmap='RdBu')
                plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

            log.info('Calculating mean contrast in dark hole')
            dh_intensity = psf * luvoir.dh_mask
            contrast = np.mean(dh_intensity[np.where(luvoir.dh_mask != 0)])
            log.info(f'contrast: {float(contrast)}')    # contrast is a Field, here casting to normal float
            all_contrasts.append(contrast)

            # Fill according entry in the matrix and subtract baseline contrast
            contrast_matrix[i,j] = contrast - contrast_floor

    # Transform saved lists to arrays
    all_psfs = np.array(all_psfs)
    all_contrasts = np.array(all_contrasts)

    # Save the PSF image *cube* as well (as opposed to each one individually)
    hcipy.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube.fits'),)
    np.savetxt(os.path.join(resDir, 'pair-wise_contrasts.txt'), all_contrasts, fmt='%e')

    # Filling the off-axis elements
    log.info('\nCalculating off-axis matrix elements...')
    matrix_two_N = np.copy(contrast_matrix)      # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(contrast_matrix)     # This will be the final PASTIS matrix.

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                log.info(f'Off-axis for i{i+1}-j{j+1}: {matrix_off_val}')

    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    matrix_pastis /= np.square(wfe_aber * 1e9)    #  1e9 converts the calibration aberration back to nanometers

    # Save matrix to file
    filename_matrix = f'PASTISmatrix_num_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}'
    hcipy.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    log.info(f'Matrix saved to: {os.path.join(resDir, filename_matrix + ".fits")}')

    # Tell us how long it took to finish.
    end_time = time.time()
    log.info(f'Runtime for matrix_building.py: {end_time - start_time}sec = {(end_time - start_time) / 60}min')
    log.info(f'Data saved to {resDir}')
    
    return overall_dir


def calculate_unaberrated_contrast_and_normalization(instrument, design=None, return_coro_simulator=True, save_coro_floor=False, save_psfs=False, outpath=''):
    """
    Calculate the direct PSF peak and unaberrated coronagraph floor of an instrument.
    :param instrument: string, 'LUVOIR', 'HiCAT' or 'JWST'
    :param design: str, optional, default=None, which means we read from the configfile: what coronagraph design
                   to use - 'small', 'medium' or 'large'
    :param return_coro_simulator: bool, whether to return the coronagraphic simulator as third return, default True
    :param save: bool, if True, will save direct and coro PSF images to disk, default False
    :param outpath: string, where to save outputs to if save=True
    :return: contrast floor and PSF normalization factor, and optionally (by default) the simulator in coron mode
    """

    if instrument == 'LUVOIR':
        # Instantiate LuvoirAPLC class
        sampling = CONFIG_PASTIS.getfloat(instrument, 'sampling')
        optics_input = CONFIG_PASTIS.get('LUVOIR', 'optics_path')
        if design is None:
            design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
        luvoir = LuvoirAPLC(optics_input, design, sampling)

        # Calculate reference images for contrast normalization and coronagraph floor
        unaberrated_coro_psf, direct = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
        norm = np.max(direct)
        direct_psf = direct.shaped
        coro_psf = unaberrated_coro_psf.shaped / norm

        # Return the coronagraphic simulator and DH mask
        coro_simulator = luvoir
        dh_mask = luvoir.dh_mask.shaped

    if instrument == 'HiCAT':
        # Set up HiCAT simulator in correct state
        hicat_sim = set_up_hicat(apply_continuous_dm_maps=True)

        # Calculate direct reference images for contrast normalization
        hicat_sim.include_fpm = False
        direct = hicat_sim.calc_psf()
        direct_psf = direct[0].data
        norm = direct_psf.max()

        # Calculate unaberrated coronagraph image for contrast floor
        hicat_sim.include_fpm = True
        coro_image = hicat_sim.calc_psf()
        coro_psf = coro_image[0].data / norm

        iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
        owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
        sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')
        dh_mask = util.create_dark_hole(coro_psf, iwa, owa, sampling).astype('bool')

        # Return the coronagraphic simulator
        coro_simulator = hicat_sim

    if instrument == 'JWST':

        # Instantiate NIRCAM object
        jwst_sim = set_up_nircam()  # this returns a tuple of two: jwst_sim[0] is the nircam object, jwst_sim[1] its ote

        # Calculate direct reference images for contrast normalization
        jwst_sim[0].image_mask = None
        direct = jwst_sim[0].calc_psf(nlambda=1)
        direct_psf = direct[0].data
        norm = direct_psf.max()

        # Calculate unaberrated coronagraph image for contrast floor
        jwst_sim[0].image_mask = CONFIG_PASTIS.get('JWST', 'focal_plane_mask')
        coro_image = jwst_sim[0].calc_psf(nlambda=1)
        coro_psf = coro_image[0].data / norm

        iwa = CONFIG_PASTIS.getfloat('JWST', 'IWA')
        owa = CONFIG_PASTIS.getfloat('JWST', 'OWA')
        sampling = CONFIG_PASTIS.getfloat('JWST', 'sampling')
        dh_mask = util.create_dark_hole(coro_psf, iwa, owa, sampling).astype('bool')

        # Return the coronagraphic simulator (a tuple in the JWST case!)
        coro_simulator = jwst_sim

    # Calculate coronagraph floor in dark hole
    contrast_floor = util.dh_mean(coro_psf, dh_mask)
    log.info(f'contrast floor: {contrast_floor}')

    if save_coro_floor:
        # Save contrast floor to text file
        with open(os.path.join(outpath, 'coronagraph_floor.txt'), 'w') as file:
            file.write(f'Coronagraph floor: {contrast_floor}')

    if save_psfs:

        # Save direct PSF, unaberrated coro PSF and DH masked coro PSF as PDF
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Direct PSF")
        plt.imshow(direct_psf, norm=LogNorm())
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title("Unaberrated coro PSF")
        plt.imshow(coro_psf, norm=LogNorm())
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("Dark hole coro PSF")
        plt.imshow(np.ma.masked_where(~dh_mask, coro_psf), norm=LogNorm())
        plt.colorbar()
        plt.savefig(os.path.join(outpath, 'unaberrated_dh.pdf'))

    if return_coro_simulator:
        return contrast_floor, norm, coro_simulator
    else:
        return contrast_floor, norm


def _jwst_matrix_one_pair(norm, wfe_aber, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate JWST mean contrast of one aberrated segment pair in NIRCam; for num_matrix_luvoir_multiprocess().
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how JWST segments start numbering at 0 just because that's python indexing, with 0 being
                         the segment A1.
    :return: contrast as float, and segment pair as tuple
    """

    # Set up JWST simulator in coronagraphic state
    jwst_instrument, jwst_ote = set_up_nircam()
    jwst_instrument.image_mask = CONFIG_PASTIS.get('JWST', 'focal_plane_mask')

    # Put aberration on correct segments. If i=j, apply only once!
    log.info(f'PAIR: {segment_pair[0]}-{segment_pair[1]}')

    # Identify the correct JWST segments
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    seg_i = wss_segs[segment_pair[0]].split('-')[0]
    seg_j = wss_segs[segment_pair[1]].split('-')[0]

    # Put aberration on correct segments. If i=j, apply only once!
    jwst_ote.zero()
    jwst_ote.move_seg_local(seg_i, piston=wfe_aber, trans_unit='m')
    if segment_pair[0] != segment_pair[1]:
        jwst_ote.move_seg_local(seg_j, piston=wfe_aber, trans_unit='m')

    log.info('Calculating coro image...')
    image = jwst_instrument.calc_psf(nlambda=1)
    psf = image[0].data / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        plt.clf()
        plt.figure(figsize=(8, 8))
        ax2 = plt.subplot(111)
        jwst_ote.display_opd(ax=ax2, vmax=500, colorbar_orientation='horizontal', title='Aberrated segment pair')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    iwa = CONFIG_PASTIS.getfloat('JWST', 'IWA')
    owa = CONFIG_PASTIS.getfloat('JWST', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('JWST', 'sampling')
    dh_mask = util.create_dark_hole(psf, iwa, owa, sampling)
    contrast = util.dh_mean(psf, dh_mask)

    return contrast, segment_pair


def _luvoir_matrix_one_pair(design, norm, wfe_aber, zern_mode, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate LVUOIR-A mean contrast of one aberrated segment pair; for num_matrix_luvoir_multiprocess().
    :param design: str, what coronagraph design to use - 'small', 'medium' or 'large'
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: float, calibration aberration per segment in m
    :param zern_mode: Zernike mode object, local Zernike aberration
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how LUVOIR segments start numbering at 1, with 0 being the center segment that doesn't exist.
    :return: contrast as float, and segment pair as tuple
    """

    # Instantiate LUVOIR object
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    optics_input = CONFIG_PASTIS.get('LUVOIR', 'optics_path')
    luv = LuvoirAPLC(optics_input, design, sampling)

    log.info(f'PAIR: {segment_pair[0]+1}-{segment_pair[1]+1}')

    # Put aberration on correct segments. If i=j, apply only once!
    luv.flatten()
    luv.set_segment(segment_pair[0]+1, wfe_aber / 2, 0, 0)
    if segment_pair[0] != segment_pair[1]:
        luv.set_segment(segment_pair[1]+1, wfe_aber / 2, 0, 0)

    log.info('Calculating coro image...')
    image, inter = luv.calc_psf(ref=False, display_intermediate=False, return_intermediate='intensity')
    # Normalize PSF by reference image
    psf = image / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{segment_pair[0]+1}-{segment_pair[1]+1}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}_segs_{segment_pair[0]+1}-{segment_pair[1]+1}'
        plt.clf()
        hcipy.imshow_field(inter['seg_mirror'], grid=luv.aperture.grid, mask=luv.aperture, cmap='RdBu')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    dh_intensity = psf * luv.dh_mask
    contrast = np.mean(dh_intensity[np.where(luv.dh_mask != 0)])
    log.info(f'contrast: {float(contrast)}')    # contrast is a Field, here casting to normal float

    return float(contrast), segment_pair


def _hicat_matrix_one_pair(norm, wfe_aber, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate HiCAT mean contrast of one aberrated segment pair; for num_matrix_luvoir_multiprocess().
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how HiCAT segments start numbering at 0, with 0 being the center segment.
    :return: contrast as float, and segment pair as tuple
    """

    # Set up HiCAT simulator in correct state
    hicat_sim = set_up_hicat(apply_continuous_dm_maps=True)
    hicat_sim.include_fpm = True

    # Put aberration on correct segments. If i=j, apply only once!
    log.info(f'PAIR: {segment_pair[0]}-{segment_pair[1]}')
    hicat_sim.iris_dm.flatten()
    hicat_sim.iris_dm.set_actuator(segment_pair[0], wfe_aber, 0, 0)
    if segment_pair[0] != segment_pair[1]:
        hicat_sim.iris_dm.set_actuator(segment_pair[1], wfe_aber, 0, 0)

    log.info('Calculating coro image...')
    image, inter = hicat_sim.calc_psf(display=False, return_intermediates=True)
    psf = image[0].data / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        plt.clf()
        plt.imshow(inter[1].phase)
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
    owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')
    dh_mask = util.create_dark_hole(psf, iwa, owa, sampling)
    contrast = util.dh_mean(psf, dh_mask)

    return contrast, segment_pair


def pastis_from_contrast_matrix(contrast_matrix, seglist, wfe_aber):
    """
    Calculate the final PASTIS matrix from the input contrast matrix (coro floor already subtracted, but only half filled).

    The contrast matrix is a nseg x nseg matrix where only half of it is filled, including the diagonal, and the other
    half is filled with zeros. It holds the DH mean contrast values of each aberrated segment pair, with an WFE
    amplitude of the calibration aberration wfe_aber, in m. Hence, the input contrast matrix is not normalized to the
    desired units yet. The coronagraph floor already needs to be subtracted from it at this point though.
    This function calculates the off-axis elements of the PASTIS matrix and then normalizes it by the calibration
    aberration to get a matrix with units of contrast / nm^2.
    Finally, it symmetrizes the matrix to output the full PASTIS matrix.

    :param contrast_matrix: nd.array, nseg x nseg matrix holding DH mean contast values of all aberrated segment pairs,
                            with the coro floor already subtracted, but only half of the matrix has non-zero values
    :param seglist: list of segment indices (e.g. 0, 1, 2, ...36 [HiCAT]; or 1, 2, ..., 120 [LUVOIR])
    :param wfe_aber: float, calibration aberration in m, this is the aberration that was used to generate contrast_matrix
    :return: the finalized PASTIS matrix in units of contrast per nanometers squared, nd.array of nseg x nseg
    """

    # Calculate the off-axis elements in the PASTIS matrix
    log.info('Calculating off-axis elements of PASTIS matrix')
    matrix_pastis_half = calculate_off_axis_elements(contrast_matrix, seglist)

    # Symmetrize the half-PASTIS matrix
    log.info('Symmetrizing PASTIS matrix')
    matrix_pastis = util.symmetrize(matrix_pastis_half)

    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    log.info('Normalizing PASTIS matrix')
    matrix_pastis /= np.square(wfe_aber * 1e9)  # 1e9 converts the calibration aberration back to nanometers

    return matrix_pastis


def calculate_off_axis_elements(contrast_matrix, seglist):
    """
    Calculate the off-axis elements of the (half) PASTIS matrix, from the contrast matrix (coro floor already subtracted).

    :param contrast_matrix: nd.array, nseg x nseg matrix holding DH mean contast values of all aberrated segment pairs,
                            with the coro floor already subtracted
    :param seglist: list of segment indices (e.g. 0, 1, 2, ...36 [HiCAT]; or 1, 2, ..., 120 [LUVOIR])
    :return: unnormalized half-PASTIS matrix, nd.array of nseg x nseg where one of its matrix triangles will be all zeros
    """

    # Create future (half filled) PASTIS matrix
    matrix_pastis_half = np.copy(contrast_matrix)     # This will be the final PASTIS matrix.

    # Calculate the off-axis elements in the (half) PASTIS matrix
    for pair in util.segment_pairs_non_repeating(contrast_matrix.shape[0]):    # this util function returns a generator
        if pair[0] != pair[1]:    # exclude diagonal elements
            matrix_off_val = (contrast_matrix[pair[0], pair[1]] - contrast_matrix[pair[0], pair[0]] - contrast_matrix[pair[1], pair[1]]) / 2.
            matrix_pastis_half[pair[0], pair[1]] = matrix_off_val
            log.info(f'Off-axis for i{seglist[pair[0]]}-j{seglist[pair[1]]}: {matrix_off_val}')

    return matrix_pastis_half


def num_matrix_multiprocess(instrument, design=None, savepsfs=True, saveopds=True):
    """
    Generate a numerical/semi-analytical PASTIS matrix.

    Multiprocessed script to calculate PASTIS matrix. Implementation adapted from
    hicat.scripts.stroke_minimization.calculate_jacobian
    :param instrument: str, what instrument (LUVOIR, HiCAT, JWST) to generate the PASTIS matrix for
    :param design: str, optional, default=None, which means we read from the configfile: what coronagraph design
                   to use - 'small', 'medium' or 'large'
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files.
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :return: overall_dir: string, experiment directory
    """

    # Keep track of time
    start_time = time.time()   # runtime is currently around 150 minutes

    ### Parameters

    # Create directory names
    tel_suffix = f'{instrument.lower()}'
    if instrument == 'LUVOIR':
        if design is None:
            design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
        tel_suffix += f'-{design}'
    overall_dir = util.create_data_path(CONFIG_PASTIS.get('local', 'local_data_path'), telescope=tel_suffix)
    os.makedirs(overall_dir, exist_ok=True)
    resDir = os.path.join(overall_dir, 'matrix_numerical')

    # Create necessary directories if they don't exist yet
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(os.path.join(resDir, 'OTE_images'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'psfs'), exist_ok=True)

    # Set up logger
    util.setup_pastis_logging(resDir, f'pastis_matrix_{tel_suffix}')
    log.info(f'Building numerical matrix for {tel_suffix}\n')

    # Read calibration aberration
    zern_number = CONFIG_PASTIS.getint('calibration', 'local_zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    # General telescope parameters
    nb_seg = CONFIG_PASTIS.getint(instrument, 'nb_subapertures')
    seglist = util.get_segment_list(instrument)
    wvln = CONFIG_PASTIS.getfloat(instrument, 'lambda') * 1e-9  # m
    wfe_aber = CONFIG_PASTIS.getfloat(instrument, 'calibration_aberration') * 1e-9   # m

    # Record some of the defined parameters
    log.info(f'Instrument: {tel_suffix}')
    log.info(f'Wavelength: {wvln} m')
    log.info(f'Number of segments: {nb_seg}')
    log.info(f'Segment list: {seglist}')
    log.info(f'wfe_aber: {wfe_aber} m')
    log.info(f'Total number of segment pairs in {instrument} pupil: {len(list(util.segment_pairs_all(nb_seg)))}')
    log.info(f'Non-repeating pairs in {instrument} pupil calculated here: {len(list(util.segment_pairs_non_repeating(nb_seg)))}')

    #  Copy configfile to resulting matrix directory
    util.copy_config(resDir)

    # Calculate coronagraph floor, and normalization factor from direct image
    contrast_floor, norm = calculate_unaberrated_contrast_and_normalization(instrument, design, return_coro_simulator=False,
                                                                            save_coro_floor=True, save_psfs=False, outpath=overall_dir)

    # Figure out how many processes is optimal and create a Pool.
    # Assume we're the only one on the machine so we can hog all the resources.
    # We expect numpy to use multithreaded math via the Intel MKL library, so
    # we check how many threads MKL will use, and create enough processes so
    # as to use 100% of the CPU cores.
    # You might think we should divide number of cores by 2 to get physical cores
    # to account for hyperthreading, however empirical testing on telserv3 shows that
    # it is slightly more performant on telserv3 to use all logical cores
    num_cpu = multiprocessing.cpu_count()
    # try:
    #     import mkl
    #     num_core_per_process = mkl.get_max_threads()
    # except ImportError:
    #     # typically this is 4, so use that as default
    #     log.info("Couldn't import MKL; guessing default value of 4 cores per process")
    #     num_core_per_process = 4

    num_core_per_process = 1   # NOTE: this was changed by Scott Will in HiCAT and makes more sense, somehow
    num_processes = int(num_cpu // num_core_per_process)
    log.info(f"Multiprocess PASTIS matrix for {instrument} will use {num_processes} processes (with {num_core_per_process} threads per process)")

    # Set up a function with all arguments fixed except for the last one, which is the segment pair tuple
    if instrument == 'LUVOIR':
        calculate_matrix_pair = functools.partial(_luvoir_matrix_one_pair, design, norm, wfe_aber, zern_mode, resDir,
                                                  savepsfs, saveopds)

    if instrument == 'HiCAT':
        # Copy used BostonDM maps to matrix folder
        shutil.copytree(CONFIG_PASTIS.get('HiCAT', 'dm_maps_path'), os.path.join(resDir, 'hicat_boston_dm_commands'))

        calculate_matrix_pair = functools.partial(_hicat_matrix_one_pair, norm, wfe_aber, resDir, savepsfs, saveopds)

    if instrument == 'JWST':
        calculate_matrix_pair = functools.partial(_jwst_matrix_one_pair, norm, wfe_aber, resDir, savepsfs, saveopds)

    # Iterate over all segment pairs via a multiprocess pool
    mypool = multiprocessing.Pool(num_processes)
    t_start = time.time()
    results = mypool.map(calculate_matrix_pair, util.segment_pairs_non_repeating(nb_seg))    # this util function returns a generator
    t_stop = time.time()

    log.info(f"Multiprocess calculation complete in {t_stop-t_start}sec = {(t_stop-t_start)/60}min")

    # Unscramble results
    # results is a list of tuples that contain the return from the partial function, in this case: result[i] = (c, (seg1, seg2))
    contrast_matrix = np.zeros([nb_seg, nb_seg])  # Generate empty matrix
    for i in range(len(results)):
        # Fill according entry in the matrix and subtract baseline contrast
        contrast_matrix[results[i][1][0], results[i][1][1]] = results[i][0] - contrast_floor
    mypool.close()

    # Save all contrasts to disk, WITH subtraction of coronagraph floor
    hcipy.write_fits(contrast_matrix, os.path.join(resDir, 'pair-wise_contrasts.fits'))
    plt.figure(figsize=(10, 10))
    plt.imshow(contrast_matrix)
    plt.colorbar()
    plt.savefig(os.path.join(resDir, 'contrast_matrix.pdf'))

    # Calculate the PASTIS matrix from the contrast matrix: off-axis elements and normalization
    matrix_pastis = pastis_from_contrast_matrix(contrast_matrix, seglist, wfe_aber)

    # Save matrix to file
    filename_matrix = f'PASTISmatrix_num_{zern_mode.name}_{zern_mode.convention + str(zern_mode.index)}'
    hcipy.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    ppl.plot_pastis_matrix(matrix_pastis, wvln, out_dir=resDir, save=True)
    log.info(f'Matrix saved to: {os.path.join(resDir, filename_matrix + ".fits")}')

    # Tell us how long it took to finish.
    end_time = time.time()
    log.info(f'Runtime for matrix_building_numerical.py/multiprocess: {end_time - start_time}sec = {(end_time - start_time)/60}min')
    log.info(f'Data saved to {resDir}')

    return overall_dir


if __name__ == '__main__':

        # Pick the function of the telescope you want to run
        #num_matrix_jwst()

        #num_matrix_luvoir(design='small')
        #num_matrix_multiprocess(instrument='LUVOIR', design='small')
        num_matrix_multiprocess(instrument='HiCAT')
