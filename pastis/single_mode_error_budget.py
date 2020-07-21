"""
This script calculates the sigmas and cumulative contrast plot of different mode-based error budgets.
The "single-mode error budget" is where one mode alone accounts for the entire contrast contribution on top on the
coronagraph floor to reach the target contrast.

The flat error budget gets calculated as a standard part of the main PASTIS analysis in "modal_analysis.py".
"""

import os
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG_INI
from e2e_simulators.luvoir_imaging import LuvoirAPLC
from modal_analysis import modes_from_file


def single_mode_sigma(c_target, c_floor, evalue):
    """
    Calculate the mode weight sigma assuming a single-mode error budget.
    :param c_target: float, overall target contrast
    :param c_floor: float, coronagraph contrast floor
    :param evalue: float, PASTIS eigenvalue of the mode weight to be calculated
    :return: mode weight sigma
    """
    sigma = np.sqrt((c_target - c_floor) / evalue)
    return sigma


def single_mode_contrasts(sigma, pmodes, single_mode, luvoir):
    """
    Calculate the contrast stemming from one weighted PASTIS mode.
    :param sigma: mode weight for the mode with index single_mode
    :param pmodes: all PASTIS modes
    :param single_mode: mode index of mode to weight and calculate contrast for
    :param luvoir: LuvoirAPLC instance
    :return: float, DH mean contrast for weighted PASTIS mode
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
    """
    Calculate and plot single-mode error budget, for onde PASTIS mode.

    Calculate the mode weight and consecutive contrast for a range of target contrasts
    and plot the recovered contrasts against the target contrasts.

    :param design: str, "small", "medium" or "large" LUVOIR-A APLC design
    :param run_choice: str, path to data
    :param c_target: float, target contrast
    :param single_mode: int, mode index for single mode error budget
    :return:
    """

    # Data directory
    workdir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), run_choice)

    # Info
    print('Working on {} coronagraph design.'.format(design))

    # Instantiate LUVOIR-A
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    luvoir = LuvoirAPLC(optics_input, design, sampling)
    luvoir.flatten()

    # Generate baseline contrast
    psf_unaber, ref = luvoir.calc_psf(ref=True)
    norm = ref.max()
    dh_intensity = psf_unaber / norm * luvoir.dh_mask
    coronagraph_floor = np.mean(dh_intensity[np.where(dh_intensity != 0)])
    print('coronagraph_floor: {}'.format(coronagraph_floor))

    # Load PASTIS modes and eigenvalues
    pmodes, svals = modes_from_file(workdir)

    print('Single mode error budget')

    # Calculate the mode weight
    single_sigma = single_mode_sigma(c_target, coronagraph_floor, svals[single_mode-1])
    print('Eigenvalue: {}'.format(svals[single_mode-1]))
    print('single_sigma: {}'.format(single_sigma))

    single_contrast = single_mode_contrasts(single_sigma, pmodes, single_mode, luvoir)
    print('contrast: {}'.format(single_contrast))

    # Make array of target contrasts
    c_list = [5e-11, 8e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
    sigma_list = []

    # Calculate according sigmas
    for i, con in enumerate(c_list):
        sigma_list.append(single_mode_sigma(con, coronagraph_floor, svals[single_mode-1]))

    # Calculate recovered contrasts
    c_recov = []
    for i, sig in enumerate(sigma_list):
        c_recov.append(single_mode_contrasts(sig, pmodes, single_mode, luvoir))

    print('c_recov: {}'.format(c_recov))
    np.savetxt(os.path.join(workdir, 'results', 'single_mode_targets.txt'), c_list)
    np.savetxt(os.path.join(workdir, 'results', 'single_mode_recovered_mode{}.txt'.format(single_mode)), c_recov)

    plt.plot(c_list, c_recov)
    plt.title('Single-mode scaling')
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('Target contrast $c_{target}$')
    plt.ylabel('Recovered contrast')
    plt.savefig(os.path.join(workdir, 'results', 'single_mode_scaled_mode{}.pdf'.format(single_mode)))


if __name__ == '__main__':

    coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
    run = CONFIG_INI.get('numerical', 'current_analysis')
    c_stat = 1e-10

    single_mode_error_budget(coro_design, run, c_stat, single_mode=69)
