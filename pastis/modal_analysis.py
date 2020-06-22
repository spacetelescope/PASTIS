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
from matplotlib.colors import LogNorm
import hcipy as hc
from hcipy.optics.segmented_mirror import SegmentedMirror

from config import CONFIG_INI
import util_pastis as util
from e2e_simulators.luvoir_imaging import LuvoirAPLC
from optimized_error_budget import build_mode_based_error_budget


def modes_from_matrix(datadir, saving=True):
    """
    Calculate mode basis and singular values from PASTIS matrix using an SVD.
    :param datadir: string, path to overall data directory containing matrix and results folder
    :param saving: string, whether to save singular values, modes and their plots or not; default=True
    :return: pmodes, svals
    """

    # Read matrix
    matrix = fits.getdata(os.path.join(datadir, 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits'))

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

    nseg = pmodes.shape[0]

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
            plt.title('Mode {}'.format(thismode + 1))
        plt.savefig(os.path.join(datadir, 'results', 'modes', 'modes_piston.pdf'))

    ### Plot them individually and save as fits and pdf
    mode_cube = []  # to save as a fits cube
    for thismode in range(nseg):

        # pdf
        plt.clf()
        hc.imshow_field(all_modes[thismode], cmap='RdBu')
        plt.axis('off')
        plt.title('Mode {}'.format(thismode + 1), size=30)
        if saving:
            plt.savefig(os.path.join(datadir, 'results', 'modes', 'pdf', 'mode{}.pdf'.format(thismode+1)))

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
    :return: wf_sm: hcipy.Wavefront of the SM propagation
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


def cumulative_contrast_e2e(pmodes, sigmas, luvoir, dh_mask, individual=False):
    """
    Calculate the cumulative contrast or contrast per mode of a set of PASTIS modes with mode weights sigmas,
    using an E2E simulator.
    :param pmodes: array, PASTIS modes [nseg, nmodes]
    :param sigmas: array, weights per PASTIS mode
    :param luvoir: LuvoirAPLC
    :param dh_mask: hcipy.Field, dh_mask that goes together with the instance of the LUVOIR simulator
    :param individual: bool, if False (default), calculates cumulative contrast, if True, calculates contrast per mode
    :return: cont_cum_e2e, list of cumulative contrasts
    """

    cont_cum_e2e = []
    for maxmode in range(pmodes.shape[0]):

        if individual:
            opd = pmodes[:, maxmode] * sigmas[maxmode]
        else:
            opd = np.nansum(pmodes[:, :maxmode+1] * sigmas[:maxmode+1], axis=1)

        luvoir.flatten()
        for seg, val in enumerate(opd):
            val *= u.nm    # the LUVOIR modes come out in units of nanometers
            luvoir.set_segment(seg + 1, val.to(u.m).value/2, 0, 0)

        # Get PSF from putting this OPD on the SM
        psf, ref = luvoir.calc_psf(ref=True)
        norm = ref.max()

        # Calculate the contrast from that PSF
        contrast = util.dh_mean(psf/norm, dh_mask)
        cont_cum_e2e.append(contrast)

    return cont_cum_e2e


def cumulative_contrast_matrix(pmodes, sigmas, matrix, c_floor, individual=False):
    """
    Calculate the cumulative contrast or contrast per mode of a set of PASTIS modes with mode weights sigmas,
    using PASTIS propagation.
    :param pmodes: array, PASTIS modes [nseg, nmodes]
    :param sigmas: array, weights per PASTIS mode
    :param matrix: array, PASTIS matrix [nseg, nseg]
    :param c_floor: float, coronagraph floor contrast
    :param individual: bool, if False (default), calculates cumulative contrast, if True, calculates contrast per mode
    :return: cont_cum_pastis, list of cumulative contrasts
    """
    cont_cum_pastis = []
    for maxmode in range(pmodes.shape[0]):

        if individual:
            aber = pmodes[:, maxmode] * sigmas[maxmode]
        else:
            aber = np.nansum(pmodes[:, :maxmode+1] * sigmas[:maxmode+1], axis=1)
        aber *= u.nm

        contrast_matrix = util.pastis_contrast(aber, matrix) + c_floor
        cont_cum_pastis.append(contrast_matrix)

    return cont_cum_pastis


def calculate_segment_constraints(pmodes, pastismatrix, c_target, baseline_contrast):
    """
    Calculate segment-based PASTIS constraints from PASTIS matrix and PASTIS modes.
    :param pmodes: array, PASTIS modes [nseg, nmodes]
    :param pastismatrix: array, full PASTIS matrix [nseg, nseg]
    :param c_target: float, static target contrast
    :param baseline_contrast: float, coronagraph floor contrast
    :return: mu_map: array, map of segment-based PASTIS constraints
    """
    nmodes = pmodes.shape[0]

    # Calculate the inverse of the pastis MODE matrix
    modestosegs = np.linalg.pinv(pmodes)

    # Calculate all mean contrasts of the pastis modes directly (as-is, with natural normalization)
    c_avg = []
    for i in range(nmodes):
        c_avg.append(util.pastis_contrast(pmodes[:, i] * u.nm, pastismatrix) + baseline_contrast)

    # Calculate segment requirements
    mu_map = np.sqrt(
        ((c_target - baseline_contrast) / nmodes) / (np.dot(c_avg - baseline_contrast, np.square(modestosegs))))

    return mu_map


def calc_random_segment_configuration(luvoir, mus, dh_mask):
    """
    Calculate the PSF after applying a randomly weighted set of segment-based PASTIS constraints on the pupil.
    :param luvoir: LuvoirAPLC
    :param mus: array, segment-based PASTIS constraints
    :param dh_mask: hcipy.Field, dark hole mask for PSF produced by luvoir
    :return: random_map: list, random segment map used in this PSF calculation in m;
             rand_contrast: float, mean contrast of the calculated PSF
    """

    # Draw a normal distribution where the stddev gets scaled to mu later on
    rand = np.random.normal(0, 1, mus.shape[0])

    mus *= u.nm

    # Multiply each segment mu by one of these random numbers,
    # put that on the LUVOIR SM and calculate the PSF.
    luvoir.flatten()
    random_map = []
    for seg, (mu, randval) in enumerate(zip(mus, rand)):
        random_seg = mu * randval
        random_map.append(random_seg.to(u.m).value)
        luvoir.set_segment(seg+1, (random_seg).to(u.m).value/2, 0, 0)
    psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False)

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # hc.imshow_field(dh_mask)
    # plt.subplot(1, 3, 2)
    # hc.imshow_field(psf, norm=LogNorm(), mask=dh_mask)
    # plt.subplot(1, 3, 3)
    # hc.imshow_field(psf, norm=LogNorm())
    # plt.show()

    rand_contrast = util.dh_mean(psf / ref.max(), dh_mask)

    return random_map, rand_contrast


def calc_random_mode_configurations(pmodes, luvoir, sigmas, dh_mask):
    """
    Calculate the PSF after weighting the PASTIS modes with weights from a normal distribution with stddev = sigmas.
    :param pmodes: array, pastis MODE matrix [nseg, nmodes]
    :param luvoir: LuvoirAPLC
    :param sigmas: array, mode-based PASTIS constraints
    :param dh_mask: hcipy.Field, dark hole mask for PSF produced by luvoir
    :return: random_weights: array, random weights used in this PSF calculation
             rand_contrast: float, mean contrast of the calculated PSF
    """

    # Create normal distribution
    rand = np.random.normal(0, 1, sigmas.shape[0])
    random_weights = sigmas * rand

    # Sum up all modes with randomly scaled sigmas to make total OPD
    opd = np.nansum(pmodes[:, :] * random_weights, axis=1)
    opd *= u.nm

    luvoir.flatten()
    for seg, aber in enumerate(opd):
        luvoir.set_segment(seg + 1, aber.to(u.m).value / 2, 0, 0)
    psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False)

    rand_contrast = util.dh_mean(psf / ref.max(), dh_mask)

    return random_weights, rand_contrast


def run_full_pastis_analysis_luvoir(design, run_choice, c_stat=1e-10, n_repeat=100):

    # Which parts are we running?
    calculate_modes = True
    calculate_sigmas = True
    run_monte_carlo_modes = True
    calc_cumulative_contrast = True
    calculate_mus = True
    run_monte_carlo_segments = True
    optimized_error_budget = True

    # Data directory
    workdir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), run_choice)

    # LUVOIR coronagraph parameters
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')

    nseg = CONFIG_INI.getint('LUVOIR', 'nb_subapertures')
    wvln = CONFIG_INI.getfloat('LUVOIR', 'lambda') * 1e-9   # [m]

    print('Setting up optics...')
    print('Data folder: {}'.format(workdir))
    print('Coronagraph: {}'.format(design))

    # Create SM
    # Read pupil and indexed pupil
    inputdir = CONFIG_INI.get('LUVOIR', 'optics_path')
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
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    # Generate reference PSF and coronagraph baseline
    luvoir.flatten()
    psf_unaber, ref = luvoir.calc_psf(ref=True, display_intermediate=False)
    norm = ref.max()

    # Make dark hole mask
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Dark hole mask")
    hc.imshow_field(dh_mask)
    plt.subplot(1, 3, 2)
    plt.title("Unaberrated PSF")
    hc.imshow_field(psf_unaber, norm=LogNorm(), mask=dh_mask)
    plt.subplot(1, 3, 3)
    plt.title("Unaberrated PSF (masked)")
    hc.imshow_field(psf_unaber, norm=LogNorm())
    plt.savefig(os.path.join(workdir, 'unaberrated_dh.pdf'))

    # Calculate coronagraph floor
    coro_floor = util.dh_mean(psf_unaber/norm, dh_mask)
    print('Coronagraph floor: {}'.format(coro_floor))
    with open(os.path.join(workdir, 'coronagraph_floor.txt'), 'w') as file:
        file.write(f'{coro_floor}')

    # Read the PASTIS matrix
    matrix = fits.getdata(os.path.join(workdir, 'matrix_numerical', 'PASTISmatrix_num_piston_Noll1.fits'))

    ### Calculate PASTIS modes and singular values
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
        np.savetxt(os.path.join(workdir, 'results', 'sigmas_{}.txt'.format(c_stat)), sigmas)

        # Plot stastic mode constraints
        plt.figure()
        plt.plot(sigmas)
        plt.semilogy()
        plt.title('Constraints per mode', size=15)
        plt.xlabel('Mode', size=15)
        plt.ylabel('Max mode contribution $\sigma_p$ (nm)', size=15)
        plt.savefig(os.path.join(workdir, 'results', 'sigmas_{}.pdf'.format(c_stat)))

    else:
        print('Reading sigmas from {}'.format(workdir))
        sigmas = np.loadtxt(os.path.join(workdir, 'results', 'sigmas_{}.txt'.format(c_stat)))

    ### Calculate Monte Carlo simulation for sigmas, with E2E
    if run_monte_carlo_modes:
        print('\nRunning Monte Carlo simulation for modes')
        # Keep track of time
        start_monte_carlo_modes = time.time()

        all_contr_rand_modes = []
        all_random_weight_sets = []
        for rep in range(n_repeat):
            print('Mode realization {}/{}'.format(rep + 1, n_repeat))
            random_weights, one_contrast_mode = calc_random_mode_configurations(pmodes, luvoir, sigmas, dh_mask)
            all_random_weight_sets.append(random_weights)
            all_contr_rand_modes.append(one_contrast_mode)

        # Mean of the distribution
        print('Mean of the Monte Carlo result modes: {}'.format(np.mean(all_contr_rand_modes)))
        end_monte_carlo_modes = time.time()

        # Save Monte Carlo simulation
        np.savetxt(os.path.join(workdir, 'results', 'random_weights_{}.txt'.format(c_stat)), all_random_weight_sets)
        np.savetxt(os.path.join(workdir, 'results', 'random_contrasts_modes_{}.txt'.format(c_stat)), all_contr_rand_modes)

        plt.figure(figsize=(16, 10))
        n, bins, patches = plt.hist(all_contr_rand_modes, int(n_repeat/10))
        plt.title('E2E raw contrast', size=20)
        plt.xlabel('Mean contrast in DH', size=20)
        plt.ylabel('PDF', size=20)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
        plt.savefig(os.path.join(workdir, 'results', 'random_sigma_distribution_{}.pdf'.format(c_stat)))

    ###  Calculate cumulative contrast plot with E2E simulator and matrix product
    if calc_cumulative_contrast:
        print('Calculating cumulative contrast plot')
        cumulative_e2e = cumulative_contrast_e2e(pmodes, sigmas, luvoir, dh_mask)
        cumulative_pastis = cumulative_contrast_matrix(pmodes, sigmas, matrix, coro_floor)

        np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_e2e_{}.txt'.format(c_stat)), cumulative_e2e)
        np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_pastis_{}.txt'.format(c_stat)), cumulative_pastis)

        # Plot the cumulative contrast from E2E simulator and matrix
        plt.figure(figsize=(16, 10))
        plt.plot(cumulative_e2e, label='E2E simulator')
        plt.plot(cumulative_pastis, label='PASTIS')
        plt.title('E2E cumulative contrast for target $C$ = {}'.format(c_stat), size=15)
        plt.xlabel('Mode number', size=15)
        plt.ylabel('Constrast', size=15)
        plt.legend()
        plt.savefig(os.path.join(workdir, 'results', 'cumulative_contrast_plot_{}.pdf'.format(c_stat)))

    ### Calculate segment-based static constraints
    if calculate_mus:
        print('Calculating static segment-based constraints')
        mus = calculate_segment_constraints(pmodes, matrix, c_stat, coro_floor)
        np.savetxt(os.path.join(workdir, 'results', 'mus_{}.txt'.format(c_stat)), mus)

        # Put mus on SM and plot
        wf_constraints = apply_mode_to_sm(mus, sm, wf_aper)

        plt.figure()
        hc.imshow_field(wf_constraints.phase / wf_constraints.wavenumber, cmap='Blues')  # in meters
        plt.title('Static segment constraints $\mu_p$ for C = {}'.format(c_stat), size=20)
        plt.colorbar()
        plt.savefig(os.path.join(workdir, 'results', 'static_constraints_{}.pdf'.format(c_stat)))

    else:
        print('Reading mus from {}'.format(workdir))
        mus = np.loadtxt(os.path.join(workdir, 'results', 'mus_{}.txt'.format(c_stat)))

    ### Calculate Monte Carlo confirmation for segments, with E2E
    if run_monte_carlo_segments:
        print('\nRunning Monte Carlo simulation for segments')
        # Keep track of time
        start_monte_carlo_seg = time.time()

        all_contr_rand_seg = []
        all_random_maps = []
        for rep in range(n_repeat):
            print('Segment realization {}/{}'.format(rep + 1, n_repeat))
            random_map, one_contrast_seg = calc_random_segment_configuration(luvoir, mus, dh_mask)
            all_random_maps.append(random_map)
            all_contr_rand_seg.append(one_contrast_seg)

        # Empirical mean and standard deviation of the distribution
        print('Mean of the Monte Carlo result segments: {}'.format(np.mean(all_contr_rand_seg)))
        print('Standard deviation of the Monte Carlo result segments: {}'.format(np.std(all_contr_rand_seg)))
        end_monte_carlo_seg = time.time()

        print('\nRuntimes:')
        print('Monte Carlo on segments with {} iterations: {} sec = {} min = {} h'.format(n_repeat, end_monte_carlo_seg - start_monte_carlo_seg,
                                                                                          (end_monte_carlo_seg - start_monte_carlo_seg) / 60,
                                                                                          (end_monte_carlo_seg - start_monte_carlo_seg) / 3600))

        # Save Monte Carlo simulation
        np.savetxt(os.path.join(workdir, 'results', 'random_maps_{}.txt'.format(c_stat)), all_random_maps)   # in m
        np.savetxt(os.path.join(workdir, 'results', 'random_contrasts_segments_{}.txt'.format(c_stat)), all_contr_rand_seg)

        # Plot histogram
        plt.figure(figsize=(16, 10))
        n, bins, patches = plt.hist(all_contr_rand_seg, int(n_repeat/10))
        plt.title('E2E raw contrast, {} iterations'.format(n_repeat), size=20)
        plt.xlabel('Mean contrast in DH', size=20)
        plt.ylabel('PDF', size=20)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
        plt.savefig(os.path.join(workdir, 'results', 'random_mu_distribution_{}.pdf'.format(c_stat)))

    ### Apply mu map and run through E2E simulator
    mus *= u.nm
    luvoir.flatten()
    for seg, mu in enumerate(mus):
        luvoir.set_segment(seg+1, mu.to(u.m).value/2, 0, 0)
    psf, ref = luvoir.calc_psf(ref=True, display_intermediate=True)
    contrast_mu = util.dh_mean(psf/ref.max(), dh_mask)
    print('Contrast with mu-map: {}'.format(contrast_mu))

    ### Calculate optimized error budget
    if optimized_error_budget:
        build_mode_based_error_budget(design, run_choice, c_stat, error_budget='optimized')
    else:
        print("No optimized error budget requested.")

    ###
    print('All saved in {}'.format(os.path.join(workdir, 'results')))
    print('\nGood job')


if __name__ == '__main__':

    coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
    run = CONFIG_INI.get('numerical', 'current_analysis')
    c_target = 1e-10
    mc_repeat = 100

    run_full_pastis_analysis_luvoir(coro_design, run_choice=run, c_stat=c_target, n_repeat=mc_repeat)
