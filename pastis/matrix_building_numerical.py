"""
This module contains functions that construct the matrix M for PASTIS *NUMERICALLY FROM THE RESPECTIVE E2E SIMULATOR*
 and saves it.

 Currently supported:
 JWST
 LUVOIR
 #TODO: HiCAT (already exists in notebook HiCAT/4)
 """

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import hcipy as hc

from config import CONFIG_INI
import util_pastis as util

# Set WebbPSF environment variable
os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')


def num_matrix_jwst():
    """
    Generate a numerical PASTIS matrix for a JWST coronagraph.

    All inputs are read from the (local) configfile and saved to the specified output directory.
    """

    import webbpsf
    from e2e_simulators import webbpsf_imaging as webbim

    # Keep track of time
    start_time = time.time()   # runtime is currently around 21 minutes
    print('Building numerical matrix for JWST\n')

    # Parameters
    resDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'active', 'matrix_numerical')
    which_tel = CONFIG_INI.get('telescope', 'name')
    nb_seg = CONFIG_INI.getint(which_tel, 'nb_subapertures')
    im_size_e2e = CONFIG_INI.getint('numerical', 'im_size_px_webbpsf')
    inner_wa = CONFIG_INI.getint(which_tel, 'IWA')
    outer_wa = CONFIG_INI.getint(which_tel, 'OWA')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    fpm = CONFIG_INI.get(which_tel, 'focal_plane_mask')                 # focal plane mask
    lyot_stop = CONFIG_INI.get(which_tel, 'pupil_plane_stop')   # Lyot stop
    filter = CONFIG_INI.get(which_tel, 'filter_name')
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration') * u.nm
    wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling
    wss_zern_nb = util.noll_to_wss(zern_number)                     # Convert from Noll to WSS framework

    # If subfolder "matrix_numerical" doesn't exist yet, create it.
    if not os.path.isdir(resDir):
        os.mkdir(resDir)

    # If subfolder "OTE_images" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(resDir, 'OTE_images')):
        os.mkdir(os.path.join(resDir, 'OTE_images'))

    # If subfolder "psfs" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(resDir, 'psfs')):
        os.mkdir(os.path.join(resDir, 'psfs'))

    # If subfolder "darkholes" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(resDir, 'darkholes')):
        os.mkdir(os.path.join(resDir, 'darkholes'))

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
    matrix_direct = np.zeros([nb_seg, nb_seg])   # Generate empty matrix
    all_psfs = []
    all_dhs = []
    all_contrasts = []

    print('nm_aber: {}'.format(nm_aber))

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('\nSTEP: {}-{} / {}-{}'.format(i+1, j+1, nb_seg, nb_seg))

            # Get names of segments, they're being addressed by their names in the ote functions.
            seg_i = wss_segs[i].split('-')[0]
            seg_j = wss_segs[j].split('-')[0]

            # Put the aberration on the correct segments
            Aber_WSS = np.zeros([nb_seg, zern_max])         # The Zernikes here will be filled in the WSS order!!!
                                                            # Because it goes into _apply_hexikes_to_seg().
            Aber_WSS[i, wss_zern_nb - 1] = nm_aber.to(u.m).value    # Aberration on the segment we're currently working on;
                                                            # convert to meters; -1 on the Zernike because Python starts
                                                            # numbering at 0.
            Aber_WSS[j, wss_zern_nb - 1] = nm_aber.to(u.m).value    # same for other segment

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
            matrix_direct[i,j] = contrast

    # Transform saved lists to arrays
    all_psfs = np.array(all_psfs)
    all_dhs = np.array(all_dhs)
    all_contrasts = np.array(all_contrasts)

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_direct)      # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(matrix_direct)     # This will be the final PASTIS matrix.

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                print('Off-axis for i{}-j{}: {}'.format(i+1, j+1, matrix_off_val))

    # Normalize matrix for the input aberration
    matrix_pastis /= np.square(nm_aber.value)

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    util.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'), header=None, metadata=None)
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Save the PSF and DH image *cubes* as well (as opposed to each one individually)
    util.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube' + '.fits'), header=None, metadata=None)
    util.write_fits(all_dhs, os.path.join(resDir, 'darkholes', 'dh_cube' + '.fits'), header=None, metadata=None)
    np.savetxt(os.path.join(resDir, 'contrasts.txt'), all_contrasts, fmt='%e')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))

    # -- Runtime notes: --
    #
    # im_size = 128
    # oversampling = 1
    # nb_seg = 18
    # runtime = 20 min


def num_matrix_luvoir():
    """
    Generate a numerical PASTIS matrix for a LUVOIR A coronagraph.

    All inputs are read from the (local) configfile and saved to the specified output directory.
    """

    from e2e_simulators.luvoir_imaging import SegmentedTelescopeAPLC

    # Keep track of time
    start_time = time.time()   # runtime is currently around 150 minutes
    print('Building numerical matrix for LUVOIR\n')

    ### Parameters

    # System parameters
    resDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'active', 'matrix_numerical')
    zern_number = CONFIG_INI.getint('calibration', 'zernike')
    zern_mode = util.ZernikeMode(zern_number)                       # Create Zernike mode object for easier handling

    # General telescope parameters
    nb_seg = CONFIG_INI.getint('LUVOIR', 'nb_subapertures')
    wvln = CONFIG_INI.getfloat('LUVOIR', 'lambda') * 1e-9  # m
    diam = CONFIG_INI.getfloat('LUVOIR', 'diameter')  # m
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration') * 1e-9   # m

    # Image system parameters
    im_lamD = 30  # image size in lambda/D
    sampling = 4

    # Coronagraph parameters
    # The LUVOIR STDT delivery in May 2018 included three different apodizers
    # we can work with, so I will implement an easy way of making a choice between them.
    design = 'small'
    datadir = '/Users/ilaginja/Documents/LabWork/ultra/LUVOIR_delivery_May2019/'
    apod_dict = {'small': {'pxsize': 1000, 'fpm_rad': 3.5, 'fpm_px': 150, 'iwa': 3.4, 'owa': 12.,
                           'fname': '0_LUVOIR_N1000_FPM350M0150_IWA0340_OWA01200_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                 'medium': {'pxsize': 1000, 'fpm_rad': 6.82, 'fpm_px': 250, 'iwa': 6.72, 'owa': 23.72,
                            'fname': '0_LUVOIR_N1000_FPM682M0250_IWA0672_OWA02372_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                 'large': {'pxsize': 1000, 'fpm_rad': 13.38, 'fpm_px': 400, 'iwa': 13.28, 'owa': 46.88,
                           'fname': '0_LUVOIR_N1000_FPM1338M0400_IWA1328_OWA04688_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'}}

    pup_px = apod_dict[design]['pxsize']
    fpm_rad = apod_dict[design]['fpm_rad']  # lambda/D
    fpm_px = apod_dict[design]['fpm_px']
    samp_foc = fpm_px / (fpm_rad * 2)  # sampling of focal plane mask
    iwa = apod_dict[design]['iwa']  # lambda/D
    owa = apod_dict[design]['owa']  # lambda/D

    # Print some of the defined parameters
    print('LUVOIR apodizer design: {}'.format(design))
    print()
    print('Wavelength: {} m'.format(wvln))
    print('Telescope diameter: {} m'.format(diam))
    print('Number of segments: {}'.format(nb_seg))
    print()
    print('IWA: {} lambda/D'.format(iwa))
    print('OWA: {} lambda/D'.format(owa))
    print('Pupil size: {} pixels'.format(pup_px))
    print('Image size: {} lambda/D'.format(im_lamD))
    print('Sampling: {} px per lambda/D'.format(sampling))
    print('FPM radius: {} lambda/D'.format(fpm_rad))
    print('Pixels in FPM: {} pixels'.format(fpm_px))

    ### Setting up the paths

    # If subfolder "matrix_numerical" doesn't exist yet, create it.
    if not os.path.isdir(resDir):
        os.mkdir(resDir)

    # If subfolder "OTE_images" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(resDir, 'OTE_images')):
        os.mkdir(os.path.join(resDir, 'OTE_images'))

    # If subfolder "psfs" doesn't exist yet, create it.
    if not os.path.isdir(os.path.join(resDir, 'psfs')):
        os.mkdir(os.path.join(resDir, 'psfs'))

    ### Preparing the optical elements

    # Pupil plane optics
    aper_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000.fits'
    aper_ind_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000_indexed.fits'
    apod_path = os.path.join(datadir, 'luvoir_stdt_baseline_bw10', design + '_fpm', 'solutions',
                             apod_dict[design]['fname'])
    ls_fname = 'inputs/LS_LUVOIR_ID0120_OD0982_no_struts_gy_ovsamp4_N1000.fits'

    pup_read = hc.read_fits(os.path.join(datadir, aper_path))
    aper_ind_read = hc.read_fits(os.path.join(datadir, aper_ind_path))
    apod_read = hc.read_fits(os.path.join(datadir, apod_path))
    ls_read = hc.read_fits(os.path.join(datadir, ls_fname))

    # Cast the into Fields on a pupil plane grid
    pupil_grid = hc.make_pupil_grid(dims=pup_px, diameter=diam)

    aperture = hc.Field(pup_read.ravel(), pupil_grid)
    aper_ind = hc.Field(aper_ind_read.ravel(), pupil_grid)
    apod = hc.Field(apod_read.ravel(), pupil_grid)
    ls = hc.Field(ls_read.ravel(), pupil_grid)

    ### Segment positions

    # Load segment positions form fits header
    hdr = fits.getheader(os.path.join(datadir, aper_ind_path))

    poslist = []
    for i in range(nb_seg):
        segname = 'SEG' + str(i + 1)
        xin = hdr[segname + '_X']
        yin = hdr[segname + '_Y']
        poslist.append((xin, yin))

    poslist = np.transpose(np.array(poslist))

    # Cast into HCIPy CartesianCoordinates (because that's what the SM needs)
    seg_pos = hc.CartesianGrid(poslist)

    ### Focal plane mask

    # Make focal grid for FPM
    focal_grid_fpm = hc.make_focal_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=fpm_rad, wavelength=wvln)

    # Also create detector plane focal grid
    focal_grid_det = hc.make_focal_grid(pupil_grid=pupil_grid, q=sampling, num_airy=im_lamD, wavelength=wvln)

    # Let's figure out how much 1 lambda/D is in radians (needed for focal plane)
    lam_over_d = wvln / diam  # rad

    # Create FPM on a focal grid, with radius in lambda/D
    fpm = 1 - hc.circular_aperture(2 * fpm_rad * lam_over_d)(focal_grid_fpm)

    ### Telescope simulator

    # Create parameter dictionary
    luvoir_params = {'wavelength': wvln, 'diameter': diam, 'imlamD': im_lamD, 'fpm_rad': fpm_rad}

    # Instantiate LUVOIR telescope with APLC
    luvoir = SegmentedTelescopeAPLC(aperture, aper_ind, seg_pos, apod, ls, fpm, focal_grid_det, luvoir_params)

    ### Dark hole mask
    dh_outer = hc.circular_aperture(2 * owa * lam_over_d)(focal_grid_det)
    dh_inner = hc.circular_aperture(2 * iwa * lam_over_d)(focal_grid_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    ### Reference images for contrast normalization and coronagraph floor
    unaberrated_coro_psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    norm = np.max(ref)

    dh_intensity = unaberrated_coro_psf / norm * dh_mask
    contrast_floor = np.mean(dh_intensity[np.where(dh_intensity != 0)])
    print(contrast_floor)

    ### Generating the PASTIS matrix and a list for all contrasts
    matrix_direct = np.zeros([nb_seg, nb_seg])   # Generate empty matrix
    all_psfs = []
    all_contrasts = []

    print('nm_aber: {} m'.format(nm_aber))

    for i in range(nb_seg):
        for j in range(nb_seg):

            print('\nSTEP: {}-{} / {}-{}'.format(i+1, j+1, nb_seg, nb_seg))

            # Put aberration on correct segments. If i=j, apply only once!
            luvoir.flatten()
            luvoir.set_segment(i+1, nm_aber/2, 0, 0)
            if i != j:
                luvoir.set_segment(j+1, nm_aber/2, 0, 0)

            print('Calculating coro image...')
            image, inter = luvoir.calc_psf(ref=False, display_intermediate=False, return_intermediate='intensity')
            # Normalize PSF by reference image
            psf = image / norm

            # Save image to disk
            filename_psf = 'psf_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(i+1) + '-' + str(j+1)
            hc.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))
            all_psfs.append(psf)

            # Save OPD images for testing (are these actually surface images, not OPD?)
            opd_name = 'opd_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index) + '_segs_' + str(
                i + 1) + '-' + str(j + 1)
            plt.clf()
            hc.imshow_field(inter['seg_mirror'], mask=aperture, cmap='RdBu')
            plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

            print('Calculating mean contrast in dark hole')
            dh_intensity = psf * dh_mask
            contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
            print('contrast:', contrast)
            all_contrasts.append(contrast)

            # Fill according entry in the matrix and subtract baseline contrast
            matrix_direct[i,j] = contrast - contrast_floor

    # Transform saved lists to arrays
    all_psfs = np.array(all_psfs)
    all_contrasts = np.array(all_contrasts)

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_direct)      # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(matrix_direct)     # This will be the final PASTIS matrix.

    for i in range(nb_seg):
        for j in range(nb_seg):
            if i != j:
                matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
                matrix_pastis[i,j] = matrix_off_val
                print('Off-axis for i{}-j{}: {}'.format(i+1, j+1, matrix_off_val))

    # Normalize matrix for the input aberration - the whole code is set up to be normalized to 1 nm, and even if
    # the units entered are in m for the sake of HCIPy, everything else is assuming the baseline is 1nm, so the
    # normalization can be taken out if we're working with exactly 1 nm for the aberration, even if entered in meters.
    #matrix_pastis /= np.square(nm_aber)

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
    hc.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Save the PSF image *cube* as well (as opposed to each one individually)
    hc.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube' + '.fits'),)
    np.savetxt(os.path.join(resDir, 'contrasts.txt'), all_contrasts, fmt='%e')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))


if __name__ == '__main__':

        # Pick the function of the telescope you want to run
        #num_matrix_jwst()
        num_matrix_luvoir()
