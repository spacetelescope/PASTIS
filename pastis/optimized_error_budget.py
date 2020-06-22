"""
This script calculates the sigmas and cumulative contrast plot of different mode-based error budgets.
The "single-mode error budget" is where one mode alone accounts for the entire contrast contribution on top on the
coronagraph floor to reach the target contrast.
The "optimized error" budget comes form the mu map and we also save the segment-space and mode-space covariance matrices.

The flat error budget gets calculated as a standard part of the main PASTIS analysis in "modal_analysis.py".
"""

import os
import astropy.units as u
import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt

from config import CONFIG_INI
import util_pastis as util
from modal_analysis import modes_from_file, cumulative_contrast_e2e
from e2e_simulators.luvoir_imaging import LuvoirAPLC


def single_mode_sigma(c_target, c_floor, evalue):
    """
    Calculate the mode weight sigma assuming a single-mode error budget.
    :param c_target: float, overall target contrast
    :param c_floor: float, coronagraph contrast floor
    :param evalue: float, PASTIS eigenvalue of the mode weight to be calcualted
    :return:
    """
    sigma = np.sqrt((c_target - c_floor) / evalue)
    return sigma


def single_mode_contrasts(sigma, pmodes, single_mode, luvoir):
    """
    Calculate the contrast stemming from one weighted PASTIS mode.
    :param sigma: mode weight for the mode with unmber single_mode
    :param pmodes: all PASTIS modes
    :param single_mode: mode number of mode to weight and calculate contrast for
    :param luvoir: LuvoirAPLC instance
    :return: float, DH mean contrast for weighted PASTIS mode
    """

    # Calculate the OPD from scaling the  mode by sigma
    opd = pmodes[:, single_mode - 1] * sigma

    # Put OPD on LUVOIR simulator
    luvoir.flatten()
    for seg, val in enumerate(opd):
        val *= u.nm
        luvoir.set_segment(seg + 1, val.to(u.m).value / 2, 0, 0)

    # Get PSF from putting this OPD on the SM
    psf, ref = luvoir.calc_psf(ref=True)
    norm = ref.max()

    # Calculate the contrast from that PSF
    dh_intensity = psf / norm * luvoir.dh_mask
    contrast = np.mean(dh_intensity[np.where(luvoir.dh_mask != 0)])

    return contrast


def build_mode_based_error_budget(design, run_choice, c_target=1e-10, error_budget='optimized', single_mode=None):
    """
    Calculate and plot optimized error budget, optimized for all PASTIS modes from the segment tolerances, or for a single mode.

    If error_budget='single_mode', calculate the mode weight and consecutive contrast for a range of target contrasts
    and plot the recovered contrasts against the target contrasts.

    If error_budget='optimized', calculate mode-space covariance matrix Cb from the mu map, extract mode weights
    (sigmas) and plot the sigmas, the contrast per mode, as well as an optimized cumulative contrast plot.

    :param design: str, "small", "medium" or "large" LUVOIR-A APLC design
    :param run_choice: str, path to data
    :param c_target: float, target contrast
    :param error_budget: str, "optimized" across all PASTIS modes from segment tolerances, or "single_mode"
    :param single_mode: int, optional, mode index for single mode error budget
    :return:
    """

    # Data directory
    workdir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), run_choice)

    # Info
    print('Working on {} coronagraph design.'.format(design))

    # Instantiate LUVOIR
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    luvoir = LuvoirAPLC(optics_input, design, sampling)
    luvoir.flatten()

    # Load PASTIS modes and eigenvalues
    pmodes, svals = modes_from_file(workdir)

    if error_budget == 'single_mode':
        print('Single mode error budget')

        # Generate baseline contrast
        psf_unaber, ref = luvoir.calc_psf(ref=True)
        norm = ref.max()
        dh_intensity = psf_unaber / norm * luvoir.dh_mask
        baseline_contrast = np.mean(dh_intensity[np.where(dh_intensity != 0)])
        print('baseline_contrast: {}'.format(baseline_contrast))

        # Calculate the mode weight
        single_sigma = single_mode_sigma(c_target, baseline_contrast, svals[single_mode-1])
        print('Eigenvalue: {}'.format(svals[single_mode-1]))
        print('single_sigma: {}'.format(single_sigma))

        single_contrast = single_mode_contrasts(single_sigma, pmodes, single_mode, luvoir)
        print('contrast: {}'.format(single_contrast))

        # Make array of target contrasts
        c_list = [5e-11, 8e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
        sigma_list = []

        # Calculate according sigmas
        for i, con in enumerate(c_list):
            sigma_list.append(single_mode_sigma(con, baseline_contrast, svals[single_mode-1]))

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

    if error_budget == 'optimized':
        print('Optimized error budget')

        # Load the mu map
        mus = np.loadtxt(os.path.join(workdir, 'results', 'mus_{}.txt'.format(c_target)))

        # Build segment-space covariance matrix Cy
        cy = np.diag(np.square(mus))
        hc.write_fits(cy, os.path.join(workdir, 'results', 'cy_{}.fits'.format(c_target)))

        # Calculate mode-space covariance matrix Cx
        cx = np.dot(np.transpose(pmodes), np.dot(cy, pmodes))
        hc.write_fits(cx, os.path.join(workdir, 'results', 'cx_{}.fits'.format(c_target)))

        # Extract optimized mode weights
        sigmas_opt = np.sqrt(np.diag(cx))
        np.savetxt(os.path.join(workdir, 'results', 'sigmas_opt_{}.txt'.format(c_target)), sigmas_opt)

        # Calculate contrast per mode
        per_mode_opt_e2e = cumulative_contrast_e2e(pmodes, sigmas_opt, luvoir, luvoir.dh_mask, individual=True)
        np.savetxt(os.path.join(workdir, 'results', 'per-mode_contrast_optimized_e2e_{}.txt'.format(c_target)),
                   per_mode_opt_e2e)

        # Calculate optimized cumulative contrast plot
        cumulative_opt_e2e = cumulative_contrast_e2e(pmodes, sigmas_opt, luvoir, luvoir.dh_mask)
        np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_optimized_e2e_{}.txt'.format(c_target)),
                   cumulative_opt_e2e)

        ### Plotting

        # Optimized mode weights
        plt.figure()
        plt.plot(sigmas_opt)
        plt.semilogy()
        plt.title('Constraints per mode', size=15)
        plt.xlabel('Mode', size=15)
        plt.ylabel('Max mode contribution $\sigma_p$ (nm)', size=15)
        plt.savefig(os.path.join(workdir, 'results', 'sigmas_opt_{}.pdf'.format(c_target)))

        # Segment-space covariance matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cy)
        plt.title('Segment-space covariance matrix $C_y$', size=20)
        plt.xlabel('segments', size=20)
        plt.ylabel('segments', size=20)
        plt.colorbar()
        plt.savefig(os.path.join(workdir, 'results', 'cy_{}.pdf'.format(c_target)))

        # Mode-space covariance matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cx)
        plt.title('Mode-space covariance matrix $C_x$', size=20)
        plt.xlabel('modes', size=20)
        plt.ylabel('modes', size=20)
        plt.colorbar()
        plt.savefig(os.path.join(workdir, 'results', 'cx_{}.pdf'.format(c_target)))

        # Contrast per mode from E2E simulator for optimized error budget
        plt.figure(figsize=(16, 10))
        plt.plot(per_mode_opt_e2e)
        plt.title('Optimized E2E contrast per mode for target $c$ = {}'.format(c_target), size=15)
        plt.xlabel('Mode number', size=15)
        plt.ylabel('Contrast', size=15)
        plt.savefig(os.path.join(workdir, 'results', 'per-mode_contrast_plot_optimized_{}.pdf'.format(c_target)))

        # Cumulative contrast from E2E simulator for optimized error budget
        plt.figure(figsize=(16, 10))
        plt.plot(cumulative_opt_e2e)
        plt.title('Optimized E2E cumulative contrast for target $c$ = {}'.format(c_target), size=15)
        plt.xlabel('Mode number', size=15)
        plt.ylabel('Contrast', size=15)
        plt.savefig(os.path.join(workdir, 'results', 'cumulative_contrast_plot_optimized_{}.pdf'.format(c_target)))


if __name__ == '__main__':

    coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
    run = CONFIG_INI.get('numerical', 'current_analysis')
    c_stat = 1e-10

    #build_mode_based_error_budget(coro_design, run, c_stat, error_budget='single_mode', single_mode=69)
    build_mode_based_error_budget(coro_design, run, c_stat, error_budget='optimized')
