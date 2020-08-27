"""
Plotting functions for the PASTIS code.
"""
import os
import hcipy
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from config import CONFIG_INI
from e2e_simulators.luvoir_imaging import LuvoirAPLC
from util_pastis import apply_mode_to_luvoir

cmap_brev = cm.get_cmap('Blues_r')


def plot_pastis_matrix(pastis_matrix, wvln, out_dir, fname_suffix='', save=False):
    """
    Plot a PASTIS matrix.
    :param pastis_matrix: array, PASTIS matrix
    :param wvln: float, wavelength at which the PASTIS matrix was generated
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'pastis_matrix'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    plt.figure(figsize=(10, 10))
    plt.imshow(pastis_matrix / wvln**2)
    plt.title('Semi-analytical PASTIS matrix', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
    cbar = plt.colorbar(fraction=0.046, pad=0.06)  # format='%.0e'
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.offsetText.set(size=15)   # this changes the base of ten size on the colorbar
    cbar.set_label('contrast/wave$^2$', size=30)
    plt.xlabel('Segments', size=30)
    plt.ylabel('Segments', size=30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_hockey_stick_curve(rms_range, pastis_matrix_contrasts, e2e_contrasts, wvln, out_dir, fname_suffix='', xlim=None, ylim=None, save=False):
    """
    Plot a hockeystick curve comparting the optical propagation between semi-analytical PASTIS and end-to-end simulator.
    :param rms_range: array or list of RMS values in nm
    :param pastis_matrix_contrasts: array or list, contrast values from SA PASTIS
    :param e2e_contrasts: array or list, contrast values from E2E simulator
    :param wvln: float, wavelength at which the PASTIS matrix was generated, in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param xlim: tuple, limits of x-axis, default None
    :param ylim:tuple, limits of y-axis, default None
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'hockeystick'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

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
    plt.xlabel("WFE RMS (waves)", size=30)
    plt.ylabel("Contrast", size=30)
    plt.legend(prop={'size': 30})
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_eigenvalues(eigenvalues, nseg, wvln, out_dir, fname_suffix='', save=False):
    """
    Plot PASTIS eigenvalues as function of PASTIS mode index.
    :param eigenvalues: array or list of eigenvalues of the PASTIS matrix
    :param nseg: int, number of segments/modes
    :param wvln: float, wavelength at which the PASTIS matrix was generated, in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'eigenvalues'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, nseg + 1), eigenvalues / wvln, linewidth=3, color='red')
    plt.semilogy()
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.title('PASTIS matrix eigenvalues', size=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Eigenvalues $\lambda_p$ (c/wave$^{2})$', size=30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_mode_weights_simple(sigmas, wvln, out_dir, c_target, fname_suffix='', labels=None, save=False):
    """
    Plot mode weights against mode index, with mode weights in units of waves.
    :param sigmas: array or list, or tuple of arrays or lists of mode weights, in nm
    :param wvln: float, wavelength at which the PASTIS matrix was generated, in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param labels: tuple, optional, labels for the different lists of sigmas provided
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'mode_requirements_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    # Figure out how many sets of sigmas we have
    if isinstance(sigmas, tuple):
        sets = len(sigmas)
        if labels is None:
            raise AttributeError('A tuple of labels needs to be defined when more than one set of sigmas is provided.')
    elif isinstance(sigmas, np.ndarray) and sigmas.ndim == 1:
        sets = 1
    else:
        raise AttributeError('sigmas must be an array of values, or a tuple of such arrays.')

    plt.figure(figsize=(12, 8))
    if sets == 1:
        plt.plot(sigmas / wvln, linewidth=3, c='r', label=labels)
    else:
        for i in range(sets):
            plt.plot(sigmas[i] / wvln, linewidth=3, label=labels[i])
    plt.semilogy()
    plt.title('Mode weights', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Mode weights $\sigma_p$ (waves)', size=30)
    if labels is not None:
        plt.legend(prop={'size': 20})
    plt.tight_layout()

    plt.annotate(s='Low impact modes\n (high tolerance)', xy=(60, 2e-5), xytext=(67, 0.0024), color='black',
                 fontweight='bold', size=25)
    plt.annotate(s='High impact modes\n (low tolerance)', xy=(60, 2e-5), xytext=(3, 3.4e-5), color='black',
                 fontweight='bold', size=25)

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_mode_weights_double_axis(sigmas, wvln, out_dir, c_target, fname_suffix='', labels=None, alphas=None, linestyles=None, colors=None, save=False):
    """
    Plot mode weights against mode index, both in units of nm and waves, on a double y-axis.
    :param sigmas: array or list, or tuple of arrays or lists of mode weights, in nm
    :param wvln: float, wavelength at which the PASTIS matrix was generated, in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param labels: tuple, optional, labels for the different lists of sigmas provided
    :param alphas: tuple, optional, transparency factors (0-1) for the different lists of sigmas provided
    :param linestyles: tuple, optional, matplotlib linestyles for the different lists of sigmas provided
    :param colors: tuple, optional, colors for the different lists of sigmas provided
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'mode_requirements_double_axis_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    # Figure out how many sets of sigmas we have
    if isinstance(sigmas, tuple):
        sets = len(sigmas)
        if labels is None:
            raise AttributeError('A tuple of labels needs to be defined when more than one set of sigmas is provided.')
        if alphas is None:
            alphas = [1] * sets
        if linestyles is None:
            linestyles = ['-'] * sets
        if colors is None:
            colors = [None] * sets
    elif isinstance(sigmas, np.ndarray) and sigmas.ndim == 1:
        sets = 1
    else:
        raise AttributeError('sigmas must be an array of values, or a tuple of such arrays.')

    # Adapted from https://matplotlib.org/gallery/subplots_axes_and_figures/fahrenheit_celsius_scales.html
    def nm2wave(wfe, wvln):
        """
        Returns WFE in waves given the wavelength.
        """
        return wfe / wvln

    def make_plot():
        # Define a closure function to register as a callback
        def convert_ax_wave_to_wave(ax_nm):
            """
            Update second axis according with first axis.
            """
            y1, y2 = ax_nm.get_ylim()
            ax_wave.set_ylim(nm2wave(y1, wvln), nm2wave(y2, wvln))
            ax_wave.figure.canvas.draw()

        fig, ax_nm = plt.subplots(figsize=(13, 8))
        ax_wave = ax_nm.twinx()

        # automatically update ylim of ax2 when ylim of ax1 changes.
        ax_nm.callbacks.connect("ylim_changed", convert_ax_wave_to_wave)

        if sets == 1:
            ax_nm.plot(sigmas / wvln, linewidth=3, c='r', label=labels)
        else:
            for i in range(sets):
                ax_nm.plot(sigmas[i] / wvln, linewidth=3, label=labels[i], alpha=alphas[i], ls=linestyles[i], c=colors[i])

        ax_nm.semilogy()
        ax_wave.semilogy()
        ax_nm.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
        ax_wave.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)

        ax_nm.set_title(f'Constraints per mode for $c_t = {c_target}$', size=30)
        ax_nm.set_ylabel('Mode weight $\sigma_p$ (nm)', size=30)
        ax_wave.set_ylabel('Mode weight $\sigma_p$ (waves)', size=30)
        ax_nm.set_xlabel('Mode index', size=30)
        if labels is not None:
            ax_nm.legend(prop={'size': 25})
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))

    make_plot()


def plot_cumulative_contrast_compare_accuracy(cumulative_c_pastis, cumulative_c_e2e, out_dir, c_target, fname_suffix='', save=False):
    """
    Plot cumulative contrast plot to verify accuracy between SA PASTIS propagation and E2E propagation.
    :param cumulative_c_pastis: array or list, contrast values from SA PASTIS
    :param cumulative_c_e2e: array or list, contrast values from E2E simulator
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'cumulative_contrast_accuracy_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

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
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_cumulative_contrast_compare_allocation(segment_based_cumulative_c, uniform_cumulative_c_e2e, out_dir, c_target, fname_suffix='', save=False):
    """
    Plot cumulative contrast plot, comparing segment-based and uniform error budget.
    :param segment_based_cumulative_c: array or list, contrast values from segment-based error budget
    :param uniform_cumulative_c_e2e: array or list, contrast values from uniform error budget
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'cumulative_contrast_allocation_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(segment_based_cumulative_c, label='Segment-driven error budget', linewidth=4)
    plt.plot(uniform_cumulative_c_e2e, label='Uniform', linewidth=4, linestyle='--', c='k', alpha=0.5)
    plt.title(f'Cumulative contrast, $c_t = {c_target}$', size=29)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Contrast', size=30)
    plt.text(0.2, 0.13, 'Uniform error budget', transform=ax.transAxes, fontsize=30, rotation=33, c='dimgrey')
    plt.text(0.06, 0.14, 'Segment-based error budget', transform=ax.transAxes, fontsize=30, rotation=40, c='C0')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # set y-axis formatter to x10^{-10}
    plt.gca().yaxis.offsetText.set_fontsize(30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_covariance_matrix(covariance_matrix, out_dir, c_target, segment_space=True, fname_suffix='', save=False):
    """
    Plot covariance matrix of a particular error budget and for a particular target contrast.
    :param covariance_matrix: array, covariance matrix in contrast/nm^2
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the covariance matrix has been calculated
    :param segment_space: bool, is this a segment-space covariance matrix or not, default is True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    seg_or_mode = 'segments_Ca' if segment_space else 'modes_Cb'

    fname = f'cov_matrix_{seg_or_mode}_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    plt.figure(figsize=(10, 10))
    plt.imshow(covariance_matrix)
    if segment_space:
        plt.title('Segment-space covariance matrix $C_a$', size=25)
        plt.xlabel('Segments', size=25)
        plt.ylabel('Segments', size=25)
    else:
        plt.title('Mode-space covariance matrix $C_b$', size=25)
        plt.xlabel('Modes', size=25)
        plt.ylabel('Modes', size=25)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
    cbar = plt.colorbar(fraction=0.046, pad=0.06)  # format='%.0e'
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('contrast/nm$^2$', size=30)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_segment_weights(mus, out_dir, c_target, labels=None, fname_suffix='', save=False):
    """
    Plot segment weights against segment index, in units of picometers (converted from input).
    :param mus: array or list, segment requirements in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param labels: tuple, optional, labels for the different lists of sigmas provided
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'segment_requirements_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    # Figure out how many sets of sigmas we have
    if isinstance(mus, tuple):
        sets = len(mus)
        if labels is None:
            raise AttributeError('A tuple of labels needs to be defined when more than one set of mus is provided.')
    elif isinstance(mus, np.ndarray) and mus.ndim == 1:
        sets = 1
    else:
        raise AttributeError('sigmas must be an array of values, or a tuple of such arrays.')

    plt.figure(figsize=(12, 8))
    if sets == 1:
        plt.plot(mus * 1e3, lw=3, label=labels)   # 1e3 to convert from nm to pm
    else:
        for i in range(sets):
            plt.plot(mus[i] * 1e3, lw=3, label=labels[i])
    plt.xlabel('Segment number', size=30)
    plt.ylabel('WFE requirements (pm)', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    if labels is not None:
        plt.legend(prop={'size': 25}, loc=(0.15, 0.73))
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_mu_map(instrument, mus, sim_instance, out_dir, design, c_target, limits=None, fname_suffix='', save=False):
    """
    Plot the segment requirement map for a specific target contrast.
    :param instrument: string, "LUVOIR" or "HiCAT"
    :param mus: array or list, segment requirements (standard deviations) in nm
    :param sim_instance: class instance of the simulator for "instrument"
    :param out_dir: str, output path to save the figure to if save=True
    :param design: str, "small", "medium", or "large" LUVOIR-A APLC design for which the mus have been calculated
    :param c_target: float, target contrast for which the segment requirements have been calculated
    :param limits: tuple, colorbar limirs, deault is None
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'segment_tolerance_map_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    if instrument == 'LUVOIR':
        wf_constraints = apply_mode_to_luvoir(mus, sim_instance)
        map_small = (wf_constraints.phase / wf_constraints.wavenumber * 1e12).shaped  # in picometers
    if instrument == 'HiCAT':
        for segnum in range(CONFIG_INI.getint(instrument, 'nb_subapertures')):
            sim_instance.iris_dm.set_actuator(segnum, mus[segnum] / 1e9, 0, 0)  # /1e9 converts to meters
        psf, inter = sim_instance.calc_psf(return_intermediates=True)
        wf_sm = inter[1].phase

        hicat_wavenumber = 2 * np.pi / (CONFIG_INI.getfloat('HiCAT', 'lambda') / 1e9)  # /1e9 converts to meters
        map_small = (wf_sm / hicat_wavenumber) * 1e12  # in picometers

    map_small = np.ma.masked_where(map_small == 0, map_small)
    cmap_brev.set_bad(color='black')

    plt.figure(figsize=(10, 10))
    plt.imshow(map_small, cmap=cmap_brev)
    cbar = plt.colorbar(fraction=0.046,
                        pad=0.04)  # no clue what these numbers mean but they did the job of adjusting the colorbar size to the actual plot size
    cbar.ax.tick_params(labelsize=30)  # this changes the numbers on the colorbar
    cbar.ax.yaxis.offsetText.set(size=25)  # this changes the base of ten on the colorbar
    cbar.set_label('picometers', size=30)
    if limits is not None:
        plt.clim(limits[0] * 1e3, limits[1] * 1e3)  # in pm
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=20)
    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def calculate_mode_phases(pastis_modes, design):
    """
    Calculate the phase maps in radians of a set of PASTIS modes.
    :param pastis_modes: array, PASTIS modes [seg, mode] in nm
    :param design: str, "small", "medium", or "large" LUVOIR-A APLC design
    :return: all_modes, array of phase pupil images
    """
    # Create luvoir instance
    sampling = CONFIG_INI.getfloat('LUVOIR', 'sampling')
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    # Calculate phases of all modes
    all_modes = []
    for mode in range(len(pastis_modes)):
        all_modes.append(apply_mode_to_luvoir(pastis_modes[:, mode], luvoir).phase)

    return all_modes


def plot_all_modes(pastis_modes, out_dir, design, fname_suffix='', save=False):
    """
    Plot all PATIS modes onto a grid.
    :param pastis_modes: array, PASTIS modes [seg, mode] in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param design: str, "small", "medium", or "large" LUVOIR-A APLC design
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'all_modes'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    # Calculate phases of all modes
    all_modes = calculate_mode_phases(pastis_modes, design)

    # Plot them
    fig, axs = plt.subplots(12, 10, figsize=(20, 24))
    for i, ax in enumerate(axs.flat):
        im = hcipy.imshow_field(all_modes[i], cmap='RdBu', ax=ax, vmin=-0.0045, vmax=0.0045)
        ax.axis('off')
        ax.annotate(f'{i + 1}', xy=(-6.8, -6.8), fontweight='roman', fontsize=13)
    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_single_mode(mode_nr, pastis_modes, out_dir, design, figsize=(8.5,8.5), vmin=None, vmax=None, fname_suffix='', save=False):
    """
    Plot a single PASTIS mode.
    :param mode_nr: int, mode index
    :param pastis_modes: array, PASTIS modes [seg, mode] in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param design: str, "small", "medium", or "large" LUVOIR-A APLC design
    :param figsize: tuple, size of figure, default=(8.5,8.5)
    :param vmin: matplotlib min extent of image, default is None
    :param vmax: matplotlib max extent of image, default is None
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'mode_{mode_nr}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    # Create luvoir instance
    sampling = CONFIG_INI.getfloat('LUVOIR', 'sampling')
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    plt.figure(figsize=figsize, constrained_layout=False)
    one_mode = apply_mode_to_luvoir(pastis_modes[:, mode_nr - 1], luvoir)
    hcipy.imshow_field(one_mode.phase, cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.annotate(f'{mode_nr}', xy=(-7.1, -6.9), fontweight='roman', fontsize=43)
    cbar = plt.colorbar(fraction=0.046,
                        pad=0.04)  # no clue what these numbers mean but it did the job of adjusting the colorbar size to the actual plot size
    cbar.ax.tick_params(labelsize=40)  # this changes the numbers on the colorbar
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_monte_carlo_simulation(random_contrasts, out_dir, c_target, segments=True, stddev=None, fname_suffix='', save=False):
    """
    Plot histogram of Monte Carlo simulation for contrasts.
    :param random_contrasts: array or list, contrasts calculated by random WFE realizations
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the Monte Carlo simulation was run
    :param segments: bool, whether run with segment or mode requirements, default is True
    :param stddev: float, standard deviation of the contrast distribution
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    mc_name = 'segments' if segments else 'modes'
    base_color = '#1f77b4' if segments else 'sandybrown'
    lines_color = 'darkorange' if segments else 'brown'

    fname = f'monte_carlo_{mc_name}_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.subplots()

    ans = np.ceil(np.log10(len(random_contrasts)))
    binsize = np.power(10, ans-1) if ans <= 3 else np.power(10, ans-2)

    n, bins, patches = plt.hist(random_contrasts, int(binsize), color=base_color)
    plt.title(f'Monte-Carlo simulation for {mc_name}', size=30)
    plt.xlabel('Mean contrast in dark hole', size=30)
    plt.ylabel('Frequency', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # set x-axis formatter to x10^{-10}
    ax1.xaxis.offsetText.set_fontsize(30)  # set x-axis formatter font size
    plt.axvline(c_target, c=lines_color, ls='-.', lw='3')
    if segments:
        plt.axvline(c_target + stddev, c=lines_color, ls=':', lw=4)
        plt.axvline(c_target - stddev, c=lines_color, ls=':', lw=4)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_contrast_per_mode(contrasts_per_mode, coro_floor, c_target, nmodes, out_dir, fname_suffix='', save=False):
    """
    Plot contrast per mode, after subtracting the coronagraph floor.
    :param contrasts_per_mode: array or list, contrast contribution per mode from optical propagation
    :param coro_floor: float, contrast floor in absence of aberrations
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param nmodes: int, number of modes/segments
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'contrast_per_mode_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.plot(contrasts_per_mode - coro_floor, linewidth=3)  # SUBTRACTING THE BASELINE CONTRAST!!
    plt.title(f'Contrast per mode, $c_t = {c_target}$', size=29)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel('Contrast', size=30)
    plt.axhline((c_target - coro_floor) / nmodes, ls='dashed', lw=3, c='dimgrey')
    plt.text(0.005, 0.55, 'Uniform error budget', transform=ax.transAxes, fontsize=30, c='dimgrey')
    plt.text(0.89, 0.85, 'Segment-based\nerror budget', transform=ax.transAxes, fontsize=30, c='C0', ha='right')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # set y-axis formatter to x10^{-10}
    plt.gca().yaxis.offsetText.set_fontsize(30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))
