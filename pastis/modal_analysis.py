"""
This is a module that lets you do a full PASTIS modal analysis from a PASTIS matrix.

Currently supports only LUVOIR.
"""
import os
import time
import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import hcipy as hc
from hcipy.optics.segmented_mirror import SegmentedMirror

from config import CONFIG_INI
import util_pastis as util
from e2e_simulators.luvoir_imaging import LuvoirAPLC


def modes_from_matrix(datadir, saving=True):
    """
    Calculate mode basis and singular values from PASTIS matrix using an SVD.
    :param datadir: string, path to overall data directory containing matrix and results folder
    :return: pmodes, svals
    """

    # Read matrix
    matrix = fits.getdata(os.path.join(datadir, 'matrix_numerical/PASTISmatrix_num_piston_Noll1.fits'))

    # Get singular modes and values from SVD
    pmodes, svals, vh = np.linalg.svd(matrix, full_matrices=True)

    # Check for results directory and create if it doesn't exits
    if not os.path.isdir(os.path.join(datadir, 'results')):
        os.mkdir(os.path.join(datadir, 'results'))

    # Save singular values and modes (singular vectors)
    if saving:
        np.savetxt(os.path.join(datadir, 'results', 'singular_values.txt'), svals)
        np.savetxt(os.path.join(datadir, 'results', 'singular_modes.txt'), pmodes)

    # Plot singular values and save
    if saving:
        plt.figure()
        plt.plot(svals)
        plt.title('PASTIS singular values')
        plt.semilogy()
        plt.xlabel('Mode')
        plt.ylabel('Log singluar value')
        plt.savefig(os.path.join(datadir, 'results', 'singular_values.pdf'))

    return pmodes, svals


def modes_from_file(datadir):
    """
    Read mode basis and singular values of a PASTIS matrix from file.
    :param datadir: string, path to overall data directory containing matrix and results folder
    :return: pmodes, svals
    """

    svals = np.loadtxt(os.path.join(datadir, 'results', 'singular_values.txt'))
    pmodes = np.loadtxt(os.path.join(datadir, 'results', 'singular_modes.txt'))

    return pmodes, svals


def full_modes_from_themselves(pmodes, datadir, sm, wf_aper, saving=False):
    """
    Put all modes onto the pupuil SM and get full 2D modes.

    Take the pmodes array of all modes (shape [segnum, modenum] = [nseg, nseg]) and apply them onto a segmenter mirror
    in the pupil. This phase gets returned both as an array of hcipy.Fields, as well as a standard array of 2D arrays.
    Optionally, save a PDF displaying all modes, a fits cube and individual PDF images.
    :param pmodes: array of PASTIS modes [segnum, modenum]
    :param datadir: string, path to overall data directory containing matrix and results folder
    :param saving: bool, whether to save figure to disk or not
    :return: all_modes as array of Fields, mode_cube as array of 2D arrays (hcipy vs matplotlib)
    """

    nseg = 120

    ### Put all modes on the SM and get their phase
    all_modes = []
    for thismode in range(nseg):
        print('Working on mode {}/{}.'.format(thismode + 1, nseg))

        wf_sm = apply_mode_to_sm(pmodes[:, thismode], sm, wf_aper)
        all_modes.append(wf_sm.phase / wf_sm.wavenumber)    # wf.phase is in rad, so this converts it to meters

    ### Check for results directory structure and create if it doesn't exist
    if saving:
        subdirs = [os.path.join(datadir, 'results'),
                   os.path.join(datadir, 'results', 'modes'),
                   os.path.join(datadir, 'results', 'modes', 'fits'),
                   os.path.join(datadir, 'results', 'modes', 'pdf')]
        for place in subdirs:
            if not os.path.isdir(place):
                os.mkdir(place)

    ### Plot all modes together and save
    if saving:
        print('Saving all PASTIS modes...')
        plt.figure(figsize=(36, 30))
        for thismode in range(nseg):
            plt.subplot(12, 10, thismode + 1)
            hc.imshow_field(all_modes[thismode], cmap='RdBu')
            plt.axis('off')
            plt.title('Mode ' + str(thismode + 1))
            plt.savefig(os.path.join(datadir, 'results', 'modes', 'modes_piston.pdf'))

    ### Plot them individually and save as fits and pdf
    mode_cube = []  # to save as a fits cube
    for thismode in range(nseg):

        # pdf
        plt.clf()
        hc.imshow_field(all_modes[thismode], cmap='RdBu')
        plt.axis('off')
        plt.title('Mode ' + str(thismode + 1), size=30)
        if saving:
            plt.savefig(os.path.join(datadir, 'results', 'modes', 'pdf', 'mode'+str(thismode+1)+'.pdf'))

        # for the fits cube
        mode_cube.append(all_modes[thismode].shaped)

    # fits cube
    mode_cube = np.array(mode_cube)
    if saving:
        hc.write_fits(mode_cube, os.path.join(datadir, 'results', 'modes', 'fits', 'cube_modes.fits'))

    return all_modes, mode_cube


def apply_mode_to_sm(pmode, sm, wf_aper):
    """
    Apply a PASTIS mode to the SM and return the propagated wavefront.

    This function first flattens the SM and then applies al segment coefficients from the input mode one by one to the
    SM.
    :param pmode: array, a single PASTIS mode [nseg]
    :param sm: hcipy.SegmentedMirror
    :param wf_aper: hcipy.Wavefront of the aperture
    :return: wf_sm: hcipy.Wavefront of the MS propagation
    """

    #Flatten SM to be sure we have no residual aberrations
    sm.flatten()

    # Loop through all segments to put them on the SM one by one
    for seg, val in enumerate(pmode):
        val *= u.nm  # the LUVOIR modes come out in units of nanometers
        sm.set_segment(seg + 1, val.to(u.m).value / 2, 0, 0)  # /2 because this SM works in surface, not OPD

    # Propagate the aperture wavefront through the SM
    wf_sm = sm(wf_aper)

    return wf_sm


def full_modes_from_file(datadir):
    """
    Read all modes into an array of hcipy.Fields and an array of 2D arrays.
    :param datadir: string, path to overall data directory containing matrix and results folder
    :return: all_modes as array of Fields, mode_cube as array of 2D arrays (hcipy vs matplotlib)
    """

    mode_cube = hc.read_fits(os.path.join(datadir, 'results', 'modes', 'fits', 'cube_modes.fits'))
    all_modes = hc.Field(mode_cube.ravel())

    return all_modes, mode_cube


def calculate_sigma(cstat, nmodes, svalue, c_floor):
    """
    Calculate the maximum mode contribution from the static contrast target and the singular values
    :param cstat: float, static contrast requirement
    :param nmodes: int, number of contributing PASTIS modes
    :param svalue: float, singular value of the mode we are calculating the sigma for
    :param c_floor: float, coronagraph floor (baseline contrast without aberrations)
    :return: float, maximum mode contribution sigma
    """
    sigma = np.sqrt((cstat - c_floor) / (nmodes * svalue))
    return sigma


def calculate_delta_sigma(cdyn, nmodes, svalue):
    """
    Calculate dynamic contrast contribution of a mode.
    :param cdyn: float, dynamic contrast requirement
    :param nseg: float, dynamic contrast requirement
    :param svalue: float, singular value of the mode we are calculating delta sigma for
    :return: float, dynamic contrast contribution
    """
    del_sigma = np.sqrt(cdyn / (np.sqrt(nmodes)*svalue))
    return del_sigma


def cumulative_contrast_e2e(pmodes, sigmas, luvoir, dh_mask, saving=True, datadir=None):

    cont_cum_e2e = []
    for maxmode in range(pmodes.shape[0]):

        opd = np.nansum(pmodes[:, :maxmode+1] * sigmas[:maxmode + 1], axis=1)

        luvoir.flatten()
        for seg, val in enumerate(opd):
            val *= u.nm    # the LUVOIR modes come out in units of nanometers
            luvoir.set_segment(seg + 1, val.to(u.m).value/2, 0, 0)

        # Get PSF from putting this OPD on the SM
        psf, ref = luvoir.calc_psf(ref=True)
        norm = ref.max()

        # Calculate the contrast from that PSF
        dh_intensity = psf / norm * dh_mask
        contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
        cont_cum_e2e.append(contrast)

    # Save to file
    if saving:
        if not os.path.isdir(os.path.join(datadir, 'results')):
            os.mkdir(os.path.join(datadir, 'results'))

        np.savetxt(os.path.join(datadir, 'results', 'cumulative_contrast_e2e.txt'), cont_cum_e2e)

    return cont_cum_e2e


def cumulative_contrast_mastix(pmodes, sigmas, matrix, c_floor, saving=True, datadir=None):
    # Calculate cumulative contrast
    cont_cum_pastis = []
    for maxmode in range(pmodes.shape[0]):
        aber = np.nansum(pmodes[:, :maxmode + 1] * sigmas[:maxmode + 1], axis=1)
        aber *= u.nm

        contrast_matrix = util.pastis_contrast(aber, matrix) + c_floor
        cont_cum_pastis.append(contrast_matrix)

    if saving:
        if not os.path.isdir(os.path.join(datadir, 'results')):
            os.mkdir(os.path.join(datadir, 'results'))

        np.savetxt(os.path.join(datadir, 'results', 'cumulative_contrast_pastis.txt'), cont_cum_pastis)

    return cont_cum_pastis


def calculate_segment_constraints(pmodes, sigmas, segnum):
    """
    Calculate segment-based PASTIS constraints from PASTIS modes and mode-based constraints.
    :param pmodes: array, PASTIS modes [nseg, nseg]
    :param sigmas: array, mode-based PASTIS constraints  [nseg]
    :param segnum: int, segment number for which to calculate the segment-based constraint
    :return: mu: float, segment-based PASTIS constraint for segment segnum
    """
    mu = np.nansum(pmodes[segnum,:]**2 * sigmas)
    return mu


def calc_random_e2e_configuration(nseg, luvoir, mus, psf_unaber, dh_mask):
    """
    Calculate the PSF after applying a randomly weighted set of segment-based PASTIS constraints on the pupil.
    :param nseg: int, number of segments
    :param luvoir: LuvoirAPLC
    :param mus: array, segment-based PASTIS constraints
    :param psf_unaber: hcipy.Field, unaberrated coronagraphic PSF
    :return: rand_contrast: float, mean contrast of the calculated PSF
    """

    # Create as many random numbers between 0 and 1 as we have segments
    rand = np.random.random(nseg) * u.nm

    # Multiply each segment mu by one of these random numbers,
    # put that on the LUVOIR SM and calculate the PSF.
    luvoir.flatten()
    for seg, (mu, randval) in enumerate(zip(mus, rand)):
        luvoir.set_segment(seg+1, (mu*randval).to(u.m).value/2, 0, 0)
    psf, ref = luvoir.calc_psf(ref=True)

    rand_contrast = util.dh_mean(psf/ref.max() - psf_unaber/ref.max(), dh_mask)

    return rand_contrast


if __name__ == '__main__':

    ### Preparations
    run_choice = '2019-8-07_002_1nm'
    workdir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), run_choice)

    # Which parts are we running?
    calculate_modes = False
    calculate_sigmas = False
    calc_cumulative_contrast = False
    calculate_mus = False
    run_monte_carlo = True

    # LUVOIR coronagraph parameters
    sampling = 4
    apodizer_design = 'small'

    # Define contrast requirements
    c_stat = 1e-10
    c_dyn = 1e-11    # not working with this yet

    # How many repetitions for Monte Carlo?
    n_repeat = 100

    nseg = 120
    wvln = 638e-9

    print('Setting up optics...')
    print('Data folder: {}'.format(workdir))
    print('Coronagraph: {}'.format(apodizer_design))

    # Create SM
    # Read pupil and indexed pupil
    inputdir = '/Users/ilaginja/Documents/LabWork/ultra/LUVOIR_delivery_May2019/'
    aper_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000.fits'
    aper_ind_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000_indexed.fits'
    aper_read = hc.read_fits(os.path.join(inputdir, aper_path))
    aper_ind_read = hc.read_fits(os.path.join(inputdir, aper_ind_path))

    # Sample them on a pupil grid and make them hc.Fields
    pupil_grid = hc.make_pupil_grid(dims=aper_ind_read.shape[0], diameter=15)
    aper = hc.Field(aper_read.ravel(), pupil_grid)
    aper_ind = hc.Field(aper_ind_read.ravel(), pupil_grid)

    # Create the wavefront on the aperture
    wf_aper = hc.Wavefront(aper, wvln)

    # Load segment positions from fits header
    hdr = fits.getheader(os.path.join(inputdir, aper_ind_path))
    poslist = []
    for i in range(nseg):
        segname = 'SEG' + str(i + 1)
        xin = hdr[segname + '_X']
        yin = hdr[segname + '_Y']
        poslist.append((xin, yin))
    poslist = np.transpose(np.array(poslist))
    seg_pos = hc.CartesianGrid(poslist)

    # Instantiate SM
    sm = SegmentedMirror(aper_ind, seg_pos)

    # Instantiate LUVOIR
    optics_input = '/Users/ilaginja/Documents/LabWork/ultra/LUVOIR_delivery_May2019/'
    luvoir = LuvoirAPLC(optics_input, apodizer_design, sampling)

    # Generate reference PSF and coronagraph baseline
    luvoir.flatten()
    psf_unaber, ref = luvoir.calc_psf(ref=True)
    norm = ref.max()

    # Make dark hole mask
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[apodizer_design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[apodizer_design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    # Calculate coronagraph floor
    coro_floor = util.dh_mean(psf_unaber/norm, dh_mask)

    # Read the matrix
    matrix = fits.getdata(os.path.join(workdir, 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits'))

    ### Get PASTIS modes and singular values
    if calculate_modes:
        print('Calculating all PASTIS modes')
        pmodes, svals = modes_from_matrix(workdir)

        ### Get full 2D modes and save them
        all_modes, mode_cube = full_modes_from_themselves(pmodes, workdir, sm, wf_aper, saving=True)

    else:
        print('Reading PASTIS modes from {}'.format(workdir))
        pmodes, svals = modes_from_file(workdir)

    ### Calculate mode-based static constraints
    if calculate_sigmas:
        print('Calculating static sigmas')
        sigmas = calculate_sigma(c_stat, nseg-1, svals, coro_floor)   # -1 because I want to ignore global piston
        np.savetxt(os.path.join(workdir, 'results', 'sigmas.txt'), sigmas)

        # Plot stastic mode constraints
        plt.figure()
        plt.plot(sigmas)
        plt.semilogy()
        plt.title('Constraints per mode', size=15)
        plt.xlabel('Mode', size=15)
        plt.ylabel('Max mode contribution $\sigma_p$ (nm)', size=15)
        plt.savefig(os.path.join(workdir, 'results', 'sigmas.pdf'))

    else:
        sigmas = np.loadtxt(os.path.join(workdir, 'results', 'sigmas.txt'))

    ###  Calculate cumulative contrast plot with E2E simulator and matrix product
    if calc_cumulative_contrast:
        print('Calculating cumulative contrast plot')
        cumulative_e2e = cumulative_contrast_e2e(pmodes, sigmas, luvoir, dh_mask, saving=True, datadir=workdir)
        cumulative_pastis = cumulative_contrast_mastix(pmodes, sigmas, matrix, coro_floor, saving=True, datadir=workdir)

        np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_e2e.txt'), cumulative_e2e)
        np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_pastis.txt'), cumulative_pastis)

        # Plot the cumulative contrast from E2E simulator and matrix
        plt.figure(figsize=(16, 10))
        plt.plot(cumulative_e2e, label='E2E simulator')
        plt.plot(cumulative_pastis, label='PASTIS')
        plt.title('E2E cumulative contrast for target $C$ = ' + str(c_stat), size=15)
        plt.xlabel('Mode number', size=15)
        plt.ylabel('Constrast', size=15)
        plt.legend()
        plt.savefig(os.path.join(workdir, 'results', 'cumulative_contrast_plot.pdf'))

    ### Calculate segment-based static constraints
    if calculate_mus:
        print('Calculating static segment-based constraints')
        mus = np.zeros_like(sigmas)
        for segnum in range(nseg):
            mus[segnum] = calculate_segment_constraints(pmodes, sigmas, segnum)

        np.savetxt(os.path.join(workdir, 'results', 'mus.txt'), mus)

        # Put mus on SM and plot
        wf_constraints = apply_mode_to_sm(mus, sm, wf_aper)

        plt.figure()
        hc.imshow_field(wf_constraints.phase / wf_constraints.wavenumber, cmap='RdBu')  # in meters
        plt.title('Static segment constraints $\mu_p$', size=20)
        plt.colorbar()
        plt.savefig(os.path.join(workdir, 'results', 'static_constraints.pdf'))

    else:
        mus = np.loadtxt(os.path.join(workdir, 'results', 'mus.txt'))

    ### Calculate Monte Carlo confirmation with E2E
    if run_monte_carlo:
        print('Running Monte Carlo simulation')
        # Keep track of time
        start_monte_carlo = time.time()

        all_contr_rand = []
        for rep in range(n_repeat):
            print('Realization {}/{}'.format(rep + 1, n_repeat))
            one_contrast = calc_random_e2e_configuration(nseg, luvoir, mus, psf_unaber, dh_mask)
            all_contr_rand.append(one_contrast)

        end_monte_carlo = time.time()

        np.savetxt(os.path.join(workdir, 'results', 'random_contrasts.txt'), all_contr_rand)

        # Plot histogram
        plt.figure(figsize=(16, 10))
        n, bins, patches = plt.hist(all_contr_rand, int(n_repeat/10))
        plt.title('E2E raw contrast', size=20)
        plt.xlabel('Mean contrast in DH', size=20)
        plt.ylabel('PDF', size=20)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
        plt.savefig(os.path.join(workdir, 'results', 'random_mu_distribution.pdf'))

    print('All saved in {}'.format(os.path.join(workdir, 'results')))

    print('\nRuntimes:')
    print('Monte Carlo with {} iterations: {} sec = {} min'.format(n_repeat, end_monte_carlo-start_monte_carlo,
                                                                   (end_monte_carlo-start_monte_carlo)/60))

    print('\nGood job')
