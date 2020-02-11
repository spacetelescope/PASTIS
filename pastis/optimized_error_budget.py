"""
This script calculates the sigmas and cumulative contrast plot of the optimized error budget, coming form the mu map.
We also save the segment-space and mode-space covariance matrices.
"""

import os
import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt

from config import CONFIG_INI
import util_pastis as util
from modal_analysis import modes_from_file, cumulative_contrast_e2e
from e2e_simulators.luvoir_imaging import LuvoirAPLC


def build_optimized_error_budget(design, run_choice, c_target):

    # Data directory
    workdir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), run_choice)

    # Instantiate LUVOIR
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    # Generate reference PSF and coronagraph baseline
    luvoir.flatten()

    # Make dark hole mask
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    # Load PASTIS modes and eigenvalues
    pmodes, svals = modes_from_file(workdir)

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

    # Calculate optimized cumulative contrast plot
    cumulative_opt_e2e = cumulative_contrast_e2e(pmodes, sigmas_opt, luvoir, dh_mask)
    np.savetxt(os.path.join(workdir, 'results', 'cumulative_contrast_optimized_e2e_{}.txt'.format(c_target)), cumulative_opt_e2e)

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

    # Cumulative contrast from E2E simulator for optimized error budget
    plt.figure(figsize=(16, 10))
    plt.plot(cumulative_opt_e2e)
    plt.title('Optimized E2E cumulative contrast for target $C$ = {}'.format(c_target), size=15)
    plt.xlabel('Mode number', size=15)
    plt.ylabel('Contrast', size=15)
    plt.savefig(os.path.join(workdir, 'results', 'cumulative_contrast_plot_optimized_{}.pdf'.format(c_target)))


if __name__ == '__main__':

    coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
    run = CONFIG_INI.get('numerical', 'current_analysis')
    c_stat = 1e-10

    build_optimized_error_budget(coro_design, run, c_stat)
