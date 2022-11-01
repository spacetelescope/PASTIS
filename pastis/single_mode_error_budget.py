"""
This script calculates the sigmas and cumulative contrast plot of different mode-based error budgets.
The "single-mode error budget" is where one mode alone accounts for the entire contrast contribution on top on the
coronagraph floor to reach the target contrast.

The flat error budget gets calculated as a standard part of the main PASTIS analysis in "pastis_analysis.py".
"""

import os
import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirAPLC
from pastis.pastis_analysis import modes_from_file
import pastis.util

log = logging.getLogger(__name__)


def single_mode_sigma(c_target, c_floor, evalue):
    """Calculate the mode weight sigma assuming a single-mode error budget.

    Parameters
    ----------
    c_target : float
        overall target contrast
    c_floor : float
        coronagraph contrast floor
    evalue : float
        PASTIS eigenvalue of the mode weight to be calculated

    Returns
    -------
    sigma : float
        mode weight
    """

    sigma = np.sqrt((c_target - c_floor) / evalue)
    return sigma


def single_mode_contrasts(sigma, pmodes, single_mode, luvoir):
    """Calculate the contrast stemming from one weighted PASTIS mode.

    Parameters
    ----------
    sigma : float
        mode weight for the mode with index single_mode
    pmodes : ndarray
        all PASTIS modes
    single_mode : int
        mode index of mode to weight and calculate contrast for
    luvoir : LuvoirAPLC
        LuvoirAPLC simulator instance

    Returns
    -------
    float
        DH mean contrast for weighted PASTIS mode
    """

    # Calculate the OPD from scaling the mode by sigma
    opd = pmodes[:, single_mode - 1] * sigma

    # Put OPD on LUVOIR simulator
    luvoir.flatten()
    for seg, val in enumerate(opd):
        val *= u.nm
        luvoir.set_segment(seg + 1, val.to(u.m).value / 2, 0, 0)

    # Get PSF from putting this OPD on the simulator
    psf, ref = luvoir.calc_psf(ref=True)
    norm = ref.max()

    # Calculate the contrast from that PSF
    dh_intensity = psf / norm * luvoir.dh_mask
    contrast = np.mean(dh_intensity[np.where(luvoir.dh_mask != 0)])

    return contrast


def single_mode_error_budget(design, run_choice, c_target=1e-10, single_mode=None):
    """Calculate and plot single-mode error budget, for onde PASTIS mode.

    Calculate the mode weight and consecutive contrast for a range of target contrasts
    and plot the recovered contrasts against the target contrasts.

    Parameters
    ----------
    design : string
        "small", "medium" or "large" LUVOIR-A APLC design
    run_choice : string
        read path to data
    c_target : float, default 1e-1-
        target contrast
    single_mode : int, default None
        mode index for single mode error budget

    Returns
    -------

    """

    # Data directory
    workdir = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), run_choice)

    # Info
    log.info(f'Working on {design} coronagraph design.')

    # Instantiate LUVOIR-A
    optics_input = os.path.join(pastis.util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    luvoir = LuvoirAPLC(optics_input, design, sampling)
    luvoir.flatten()

    # Generate baseline contrast
    psf_unaber, ref = luvoir.calc_psf(ref=True)
    norm = ref.max()
    dh_intensity = psf_unaber / norm * luvoir.dh_mask
    coronagraph_floor = np.mean(dh_intensity[np.where(dh_intensity != 0)])
    log.info(f'coronagraph_floor: {coronagraph_floor}')

    # Load PASTIS modes and eigenvalues
    pmodes, svals = modes_from_file(workdir)

    log.info('Single mode error budget')

    # Calculate the mode weight
    single_sigma = single_mode_sigma(c_target, coronagraph_floor, svals[single_mode - 1])
    log.info(f'Eigenvalue: {svals[single_mode-1]}')
    log.info(f'single_sigma: {single_sigma}')

    single_contrast = single_mode_contrasts(single_sigma, pmodes, single_mode, luvoir)
    log.info(f'contrast: {single_contrast}')

    # Make array of target contrasts
    c_list = [5e-11, 8e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
    sigma_list = []

    # Calculate according sigmas
    for i, con in enumerate(c_list):
        sigma_list.append(single_mode_sigma(con, coronagraph_floor, svals[single_mode - 1]))

    # Calculate recovered contrasts
    c_recov = []
    for i, sig in enumerate(sigma_list):
        c_recov.append(single_mode_contrasts(sig, pmodes, single_mode, luvoir))

    log.info(f'c_recov: {c_recov}')
    np.savetxt(os.path.join(workdir, 'results', 'single_mode_target_contrasts.txt'), c_list)
    np.savetxt(os.path.join(workdir, 'results', f'single_mode_recovered_contrasts_mode{single_mode}.txt'), c_recov)

    plt.plot(c_list, c_recov)
    plt.title('Single-mode scaling')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('Target contrast $c_{target}$')
    plt.ylabel('Recovered contrast')
    plt.savefig(os.path.join(workdir, 'results', f'single_mode_scaled_mode{single_mode}.pdf'))


if __name__ == '__main__':

    coro_design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
    run = CONFIG_PASTIS.get('numerical', 'current_analysis')
    c_stat = 1e-10

    single_mode_error_budget(coro_design, run, c_stat, single_mode=69)
