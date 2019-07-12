"""
This script computes the contrast for a random IrisAO mislignment on the HiCAT simulator.
"""

import os
import time
import numpy as np
from astropy.io import fits
import astropy.units as u
#import hicat.simulators
import hcipy as hc

import util_pastis as util
from e2e_simulators.luvoir_imaging import SegmentedTelescopeAPLC

def contrast_hicat_num(dir, matrix_mode='hicat', rms=1*u.nm):

    nb_seg = 36

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Import numerical PASTIS matrix for HiCAT sim
    filename = 'PASTISmatrix_num_HiCAT_piston_Noll1'
    matrix_pastis = fits.getdata(os.path.join(dir, filename + '.fits'))

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
    dh_mask = util.create_dark_hole(psf_hicat, iwa=5, owa=12, samp=13 / 4)
    # Get the mean contrast
    hicat_dh_psf = psf_hicat * dh_mask
    contrast_hicat = np.mean(hicat_dh_psf[np.where(hicat_dh_psf != 0)])
    end_e2e = time.time()

    ###
    # Calculate baseline contrast
    baseline_dh = psf_coro * dh_mask
    contrast_base = np.mean(baseline_dh[np.where(baseline_dh != 0)])

    ## MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
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


def contrast_luvoir_num(dir, matrix_mode='luvoir', rms=1*u.nm):

    nb_seg = 120

    # Keep track of time
    start_time = time.time()   # runtime currently is around ? min

    # Import numerical PASTIS matrix for HiCAT sim
    filename = 'PASTISmatrix_num_piston_Noll1'
    matrix_pastis = fits.getdata(os.path.join(dir, filename + '.fits'))

    # Fix false normalization
    #matrix_pastis *= np.square(1e-9)

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

    # General telescope parameters
    nb_seg = 120
    wvln = 638e-9  # m
    diam = 15.  # m

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
    dh_outer = hc.circular_aperture(2 * owa * lam_over_d)(focal_grid_det)
    dh_inner = hc.circular_aperture(2 * iwa * lam_over_d)(focal_grid_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    # Get the mean contrast
    dh_intensity = psf_luvoir * dh_mask
    contrast_luvoir = np.mean(dh_intensity[np.where(dh_intensity != 0)])
    end_e2e = time.time()

    ###
    # Calculate baseline contrast
    baseline_dh = psf_coro * dh_mask
    contrast_base = np.mean(baseline_dh[np.where(baseline_dh != 0)])
    print('Baseline contrast: {}'.format(contrast_base))

    ## MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
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

    #c_e2e, c_matrix = contrast_hicat_num(dir='/Users/ilaginja/Documents/Git/PASTIS/Jupyter Notebooks/HiCAT', rms=10*u.nm)

    c_e2e, c_matrix = contrast_luvoir_num(dir='/Users/ilaginja/Documents/data_from_repos/pastis_data/active/matrix_numerical',
                                         rms=1 * u.nm)
