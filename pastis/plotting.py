"""
Plotting functions for the PASTIS code.
"""
import os
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from e2e_simulators.luvoir_imaging import LuvoirAPLC
from modal_analysis import apply_mode_to_sm

cmap_brev = cm.get_cmap('Blues_r')


def plot_pastis_matrix(pastis_matrix, wvln, out_dir, design):

    plt.figure(figsize=(10, 10))
    plt.imshow(pastis_matrix / wvln)
    plt.title('Semi-analytical PASTIS matrix', size=30)
    cbar = plt.colorbar(fraction=0.046, pad=0.06)  # format='%.0e'
    cbar.ax.tick_params(labelsize=20)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
    cbar.ax.yaxis.offsetText.set(size=15)   # this changes the base of ten size on the colorbar
    cbar.set_label('contrast/wave$^2$', size=30)
    plt.xlabel('Segments', size=30)
    plt.ylabel('Segments', size=30)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'matrix_{design}.pdf'))


def plot_hockey_stick_curve(rms_range, pastis_matrix_contrasts, e2e_contrasts, wvln, out_dir, design, xlim=None, ylim=None):
    plt.figure(figsize=(12, 8))
    plt.title("Semi-analytical PASTIS vs. E2E", size=30)
    plt.plot(rms_range / wvln, pastis_matrix_contrasts, label="SA PASTIS", linewidth=4)
    plt.plot(rms_range / wvln, e2e_contrasts, label="E2E simulator", linewidth=4, linestyle='--')
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.semilogx()
    plt.semilogy()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel("OPD RMS (waves)", size=30)
    plt.ylabel("Contrast", size=30)
    plt.legend(prop={'size': 30})
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'hockeystick_{design}.pdf'))


def plot_eigenvalues(eigenvalues, nseg, wvln, out_dir, design):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, nseg + 1), eigenvalues / wvln, linewidth=3, color='red')
    plt.semilogy()
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.title('PASTIS matrix eigenvalues', size=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Eigenvalues $\lambda_p$ (c/wave$^{2})$', size=30)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'eigenvalues_{design}.pdf'))


def plot_mode_weights(sigmas, wvln, out_dir, labels=None):

    # Figure out how many sets of sigmas we have
    if isinstance(sigmas, tuple):
        sets = len(sigmas)
    elif isinstance(sigmas, np.array) and sigmas.ndim == 1:
        sets = 1

    plt.figure(figsize=(12, 8))
    if sets == 1:
        plt.plot(sigmas / wvln, linewidth=3, c='r', label=labels)
    else:
        for i in range(sets):
            plt.plot(sigmas[i] / wvln, linewidth=3, label=labels[i])
    plt.semilogy()
    plt.title('Uniform contrast allocation across modes', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Mode weights $\widetilde{b_p}$ (waves)', size=30)
    plt.legend()
    plt.tight_layout()

    plt.annotate(s='Low impact modes\n (high tolerance)', xy=(60, 2e-5), xytext=(67, 0.0024), color='black',
                 fontweight='bold', size=25)
    plt.annotate(s='High impact modes\n (low tolerance)', xy=(60, 2e-5), xytext=(3, 3.4e-5), color='black',
                 fontweight='bold', size=25)

    plt.savefig(os.path.join(out_dir, 'sigmas_flat_error_budget.pdf'))


def plot_cumulative_contrast(cumulative_c_pastis, cumulative_c_e2e, out_dir, design):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plt.plot(cumulative_c_pastis, label='SA PASTIS', linewidth=4)
    plt.plot(cumulative_c_e2e, label='E2E simulator', linewidth=4, linestyle='--')
    plt.title('Cumulative contrast', size=25)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Cumulative contrast', size=30)
    plt.legend(prop={'size': 30}, loc=(0.02, 0.52))
    plt.axhline(cumulative_c_e2e[0], linestyle='dashdot', c='dimgrey')  # coronagraph floor
    plt.axhline(cumulative_c_e2e[-1], linestyle='dashdot', c='dimgrey')  # target contrast
    plt.text(75, cumulative_c_e2e[0], "coronagraph floor", size=30)
    plt.text(15, cumulative_c_e2e[-1], "target contrast", size=30)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # set y-axis formatter to x10^{-10}
    ax.yaxis.offsetText.set_fontsize(30)  # fontsize for y-axis formatter

    plt.savefig(os.path.join(out_dir, f'cumulative_contrast_{design}.pdf'))


def plot_covariance_matrix(covariance_matrix, out_dir, design, segment_space=True):
    seg_or_mode = 'c_a' if segment_space else 'c_b'

    plt.figure(figsize=(10, 10))
    plt.imshow(covariance_matrix)
    if segment_space:
        plt.title('Segment-space covariance matrix $C_a$', size=25)
        plt.xlabel('segments', size=25)
        plt.ylabel('segments', size=25)
    else:
        plt.title('Mode-space covariance matrix $C_b$', size=25)
        plt.xlabel('Modes', size=25)
        plt.ylabel('Modes', size=25)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
    cbar = plt.colorbar(fraction=0.046, pad=0.06)  # format='%.0e'
    cbar.ax.tick_params(labelsize=15)

    plt.savefig(os.path.join(out_dir, f'{seg_or_mode}_{design}.pdf'))