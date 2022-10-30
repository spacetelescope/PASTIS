"""
Plotting and animation functions for the PASTIS code.
"""
import copy
import os
import glob
import progressbar
import re
import time

from astropy.io import fits
import hcipy
import matplotlib
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import norm

from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirAPLC
import pastis.simulators.webbpsf_imaging as webbpsf_imaging
import pastis.util

matplotlib.rc('image', origin='lower')    # Make sure image origin is always in lower left
cmap_brev = copy.copy(cm.get_cmap('Blues_r'))        # A blue colormap where white is zero, used for mu maps
cmap_brev.set_bad(color='black')
clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
blue_orange_divergent = LinearSegmentedColormap.from_list("custom_blue_orange", clist)    # diverging colormap for PASTIS matrix
# Define a normalization of diverging colormap so that it is centered on zero (depending on matrix, black or white)
norm_center_zero = matplotlib.colors.TwoSlopeNorm(vcenter=0)


def plot_direct_coro_dh(direct_psf, coro_psf, dh_mask, outpath):
    # Save direct PSF, unaberrated coro PSF and DH masked coro PSF as PDF
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Direct PSF")
    plt.imshow(direct_psf, norm=LogNorm())
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("Unaberrated coro PSF")
    plt.imshow(coro_psf, norm=LogNorm())
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Dark hole coro PSF")
    plt.imshow(np.ma.masked_where(~dh_mask, coro_psf), norm=LogNorm())
    plt.colorbar()
    plt.savefig(os.path.join(outpath, 'unaberrated_dh.pdf'))


def plot_pastis_matrix(pastis_matrix, wvln=None, out_dir='', fname_suffix='', save=False):
    """
    Plot a PASTIS matrix.
    :param pastis_matrix: array, PASTIS matrix in units of contrast/nm**2
    :param wvln: float, optional, wavelength at which the PASTIS matrix was generated in nm. If provided, converts
                 PASTIS matrix to units of contrast/wave^2, if None it stays in contrast/nm^2.
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'pastis_matrix'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    if wvln is not None:
        matrix_to_plot = pastis_matrix * wvln**2
        cbar_label = 'contrast/wave$^2$'
    else:
        matrix_to_plot = pastis_matrix
        cbar_label = 'contrast/nm$^2$'

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_to_plot, cmap=blue_orange_divergent, norm=norm_center_zero)
    plt.title('Semi-analytical PASTIS matrix', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
    cbar = plt.colorbar(fraction=0.046, pad=0.06)  # format='%.0e'
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.offsetText.set(size=15)   # this changes the base of ten size on the colorbar
    cbar.set_label(cbar_label, size=30)
    plt.xlabel('Segments', size=30)
    plt.ylabel('Segments', size=30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))
    else:
        plt.show()


def plot_hockey_stick_curve(rms_range, pastis_matrix_contrasts, e2e_contrasts, wvln=None, out_dir='', fname_suffix='', xlim=None, ylim=None, save=False):
    """
    Plot a hockeystick curve comparing the optical propagation between semi-analytical PASTIS and end-to-end simulator.
    :param rms_range: array or list of RMS values in nm
    :param pastis_matrix_contrasts: array or list, contrast values from SA PASTIS
    :param e2e_contrasts: array or list, contrast values from E2E simulator
    :param wvln: float, optional, wavelength at which the PASTIS matrix was generated in nm. If provided, converts
                 rms_range (x-axis) to units of waves, if None it stays in nm.
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

    if wvln is not None:
        rms_range_to_plot = rms_range / wvln
        rms_units = 'wave'
    else:
        rms_range_to_plot = rms_range
        rms_units = 'nm'

    plt.figure(figsize=(12, 8))
    plt.title("Semi-analytical PASTIS vs. E2E", size=30)
    plt.plot(rms_range_to_plot, pastis_matrix_contrasts, label="SA PASTIS", linewidth=4)
    plt.plot(rms_range_to_plot, e2e_contrasts, label="E2E simulator", linewidth=4, linestyle='--')
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.semilogx()
    plt.semilogy()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel(f"WFE RMS ({rms_units})", size=30)
    plt.ylabel("Contrast", size=30)
    plt.legend(prop={'size': 30})
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))
    else:
        plt.show()


def plot_eigenvalues(eigenvalues, nseg, wvln=None, out_dir='', fname_suffix='', save=False):
    """
    Plot PASTIS eigenvalues as function of PASTIS mode index.
    :param eigenvalues: array or list of eigenvalues of the PASTIS matrix, in units of contrast/nm**2
    :param nseg: int, number of segments/modes
    :param wvln: float, optional, wavelength at which the PASTIS matrix was generated in nm. If provided, converts
                 eiganvalues to units of contrast/wave^2, if None they stay in contrast/nm^2.
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'eigenvalues'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    if wvln is not None:
        evals_to_plot = eigenvalues * wvln**2
        evals_unit = 'c/wave$^{2}$'
    else:
        evals_to_plot = eigenvalues
        evals_unit = 'c/nm$^2$'

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, nseg + 1), evals_to_plot, linewidth=3, color='red')
    plt.semilogy()
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.title('PASTIS matrix eigenvalues', size=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel(f'Eigenvalues $\lambda_p$ ({evals_unit})', size=30)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))
    else:
        plt.show()


def plot_mode_weights_simple(sigmas, c_target, wvln=None, out_dir='', fname_suffix='', labels=None, save=False):
    """
    Plot mode weights against mode index, with mode weights in units of waves.
    :param sigmas: array or list, or tuple of arrays or lists of mode weights, in nm
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param wvln: float, optional, wavelength at which the PASTIS matrix was generated in nm. If provided, converts
                 mode weights (sigmas) to units of waves, if None they stay in nm.
    :param out_dir: str, output path to save the figure to if save=True
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param labels: tuple, optional, labels for the different lists of sigmas provided
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'mode_requirements_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    if wvln is not None:
        sigmas_to_plot = sigmas / wvln
        weights_units = 'waves'
    else:
        sigmas_to_plot = sigmas
        weights_units = 'nm'

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
        plt.plot(sigmas_to_plot, linewidth=3, c='r', label=labels)
    else:
        for i in range(sets):
            plt.plot(sigmas_to_plot[i], linewidth=3, label=labels[i])
    plt.semilogy()
    plt.title('Mode weights', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    plt.xlabel('Mode index', size=30)
    plt.ylabel(f'Mode weights $\sigma_p$ ({weights_units})', size=30)
    if labels is not None:
        plt.legend(prop={'size': 20})
    plt.tight_layout()

    plt.annotate(text='Low impact modes\n (high tolerance)', xy=(60, 2e-5), xytext=(67, 0.0024), color='black',
                 fontweight='bold', size=25)
    plt.annotate(text='High impact modes\n (low tolerance)', xy=(60, 2e-5), xytext=(3, 3.4e-5), color='black',
                 fontweight='bold', size=25)

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))
    else:
        plt.show()


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


def plot_cumulative_contrast_compare_accuracy(cumulative_c_pastis, cumulative_c_e2e, out_dir, coro_floor, c_target,
                                              fname_suffix='', save=False):
    """
    Plot cumulative contrast plot to verify accuracy between SA PASTIS propagation and E2E propagation.
    :param cumulative_c_pastis: array or list, contrast values from SA PASTIS
    :param cumulative_c_e2e: array or list, contrast values from E2E simulator
    :param out_dir: str, output path to save the figure to if save=True
    :param: coro_floor: float, contrast floor in absence of aberrations
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
    plt.axhline(coro_floor, linestyle='dashdot', c='dimgrey')  # coronagraph floor
    plt.axhline(c_target, linestyle='dashdot', c='dimgrey')  # target contrast
    plt.text(75, coro_floor, "coronagraph floor", size=30)
    plt.text(15, c_target, "target contrast", size=30)
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
    plt.imshow(covariance_matrix, cmap='seismic', norm=norm_center_zero)
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


def plot_segment_weights(mus, out_dir, c_target, labels=None, fname=None, save=False):
    """
    Plot segment weights against segment index, in units of picometers (converted from input).

    :param mus: array or list of arrays, segment requirements in nm
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the mode weights have been calculated
    :param labels: list, optional, labels for the different lists of sigmas provided
    :param fname: str, optional, file name to save plot to
    :param save: bool, whether to save to disk or not, default is False
    """
    if fname is None:
        fname = f'segment_requirements_{c_target:.2e}'

    # Figure out how many sets of mode coefficients per segment we have
    if isinstance(mus, list):
        sets = len(mus)
        if sets > 1:
            if labels is None:
                raise AttributeError('A list of labels needs to be defined when more than one set of mus is provided.')
        elif sets == 1:
            mus = mus[0]
    elif isinstance(mus, np.ndarray) and mus.ndim == 1:
        sets = 1
    else:
        raise AttributeError('Segment weights "mus" must be a 1d array of values, or a list of such arrays.')

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
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_mu_map(instrument, mus, sim_instance, out_dir, c_target, limits=None, fname_suffix='', save=False):
    """
    Plot the segment requirement map for a specific target contrast.
    :param instrument: string, "LUVOIR", "HiCAT" or "JWST"
    :param mus: array or list, segment requirements (standard deviations) in nm WFE
    :param sim_instance: class instance of the simulator for "instrument"
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the segment requirements have been calculated
    :param limits: tuple, colorbar limits, default is None
    :param fname_suffix: str, optional, suffix to add to the saved file name
    :param save: bool, whether to save to disk or not, default is False
    :return:
    """
    fname = f'segment_tolerance_map_{c_target}'
    if fname_suffix != '':
        fname += f'_{fname_suffix}'

    if instrument == 'LUVOIR':
        sim_instance.flatten()
        wf_constraints = pastis.util.apply_mode_to_luvoir(mus, sim_instance)[0]
        map_small = (wf_constraints.phase / wf_constraints.wavenumber * 1e12).shaped  # in picometers

    if instrument == 'HiCAT':
        sim_instance.iris_dm.flatten()
        for segnum in range(CONFIG_PASTIS.getint(instrument, 'nb_subapertures')):
            sim_instance.iris_dm.set_actuator(segnum, mus[segnum] / 1e9, 0, 0)  # /1e9 converts to meters
        psf, inter = sim_instance.calc_psf(return_intermediates=True)
        wf_sm = inter[1].phase

        hicat_wavenumber = 2 * np.pi / (CONFIG_PASTIS.getfloat('HiCAT', 'lambda') / 1e9)  # /1e9 converts to meters
        map_small = (wf_sm / hicat_wavenumber) * 1e12  # in picometers

    if instrument == 'JWST':
        sim_instance[1].zero()
        for segnum in range(CONFIG_PASTIS.getint(instrument, 'nb_subapertures')):  # TODO: there is probably a single function that puts the aberration on the OTE at once
            seg_name = webbpsf_imaging.WSS_SEGS[segnum].split('-')[0]
            sim_instance[1].move_seg_local(seg_name, piston=mus[segnum]/2, trans_unit='nm')    # this function works with physical motions, meaning the piston is in surface

        psf, inter = sim_instance[0].calc_psf(nlambda=1, return_intermediates=True)
        wf_sm = inter[1].phase

        jwst_wavenumber = 2 * np.pi / (CONFIG_PASTIS.getfloat('JWST', 'lambda') / 1e9)  # /1e9 converts to meters
        map_small = (wf_sm / jwst_wavenumber) * 1e12  # in picometers

    map_small = np.ma.masked_where(map_small == 0, map_small)

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
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    optics_input = os.path.join(pastis.util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    # Calculate phases of all modes
    all_modes = []
    for mode in range(len(pastis_modes)):
        all_modes.append(pastis.util.apply_mode_to_luvoir(pastis_modes[:, mode], luvoir)[0].phase)

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
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    optics_input = os.path.join(pastis.util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    plt.figure(figsize=figsize, constrained_layout=False)
    one_mode = pastis.util.apply_mode_to_luvoir(pastis_modes[:, mode_nr - 1], luvoir)[0]
    hcipy.imshow_field(one_mode.phase, cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.annotate(f'{mode_nr}', xy=(-7.1, -6.9), fontweight='roman', fontsize=43)
    cbar = plt.colorbar(fraction=0.046,
                        pad=0.04)  # no clue what these numbers mean but it did the job of adjusting the colorbar size to the actual plot size
    cbar.ax.tick_params(labelsize=40)  # this changes the numbers on the colorbar
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(out_dir, '.'.join([fname, 'pdf'])))


def plot_monte_carlo_simulation(random_contrasts, out_dir, c_target, segments=True, stddev=None, plot_empirical_stats=False, fname_suffix='', save=False):
    """
    Plot histogram of Monte Carlo simulation for contrasts.
    :param random_contrasts: array or list, contrasts calculated by random WFE realizations
    :param out_dir: str, output path to save the figure to if save=True
    :param c_target: float, target contrast for which the Monte Carlo simulation was run
    :param segments: bool, whether run with segment or mode requirements, default is True
    :param stddev: float, analytically calculated standard deviation of the contrast distribution
    :param plot_empirical_stats: bool, whether to plot the empirical mean and standard deviation from the data
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

    n, bins, patches = plt.hist(np.array(random_contrasts), int(binsize), color=base_color)
    plt.title(f'Monte-Carlo simulation for {mc_name}', size=30)
    plt.xlabel('Mean contrast in dark hole', size=30)
    plt.ylabel('Frequency', size=30)
    plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=30)
    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # set x-axis formatter to x10^{-10}
    ax1.xaxis.offsetText.set_fontsize(30)  # set x-axis formatter font size
    plt.axvline(c_target, c=lines_color, ls='-.', lw='3')
    # Add analytical mean and stddev
    if stddev:
        plt.axvline(c_target + stddev, c=lines_color, ls=':', lw=4, label='Analytical stddev')
        plt.axvline(c_target - stddev, c=lines_color, ls=':', lw=4)
    # Add empirical mean and stddev
    if plot_empirical_stats:
        empirical_mean = np.mean(random_contrasts)
        empirical_stddev = np.std(random_contrasts)
        plt.axvline(empirical_mean, c='maroon', ls='-.', lw='3')
        plt.axvline(empirical_mean + empirical_stddev, c='maroon', ls=':', lw=4, label='Empirical stddev')
        plt.axvline(empirical_mean - empirical_stddev, c='maroon', ls=':', lw=4)
    if stddev or plot_empirical_stats:
        plt.legend(prop={'size': 20})
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


def animate_contrast_matrix(data_path, instrument='LUVOIR', design='small', display_mode='stretch'):
    """
    Create animation of the contrast matrix generation and save to MP4 file.
    :param data_path: string, absolute path to main PASTIS directory containing all subdirs, e.g. "matrix_numerical"
    :param instrument: string, "LUVOIR" or "HiCAT"
    :param design: string, necessary if instrument='LUVOIR', defaults to "small" - LUVOIR APLC design choice
    :param display_mode: string, 'boxy' for two panels on top, one on bottom, 'stretch' for all three panels in one row
    """

    # Keep track of time
    start_time = time.time()

    # Load contrast matrix and OTE + PSF fits images
    contrast_matrix = fits.getdata(os.path.join(data_path, 'matrix_numerical', 'contrast_matrix.fits'))
    print('Reading OTE images...')
    all_ote_images = read_ote_fits_files(data_path)
    print('All OTE fits files read')
    print('Reading PSF images...')
    all_psf_images = read_psf_fits_files(data_path)
    print('All PSF fits files read')

    # Define some instrument specific parameters
    if instrument == 'LUVOIR':
        # Instantiate LUVOIR sim object (needed only for DH mask)
        sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
        optics_input = os.path.join(pastis.util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
        luvoir = LuvoirAPLC(optics_input, design, sampling)
        dh_mask = luvoir.dh_mask.shaped
        # Load LUVOIR aperture file
        aper_path_in_optics = CONFIG_PASTIS.get('LUVOIR', 'aperture_path_in_optics')
        aperture = fits.getdata(os.path.join(optics_input, aper_path_in_optics))
        # Calculate segment pair tuples
        seg_pair_tuples = list(pastis.util.segment_pairs_non_repeating(120))

        # Define plotting limits
        vmin_psfs = 1e-10
        vmax_psfs = 1e-7

    elif instrument == 'HiCAT':
        # Create HiCAT DH mask
        iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
        owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
        sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')
        dh_mask = pastis.util.create_dark_hole(all_psf_images[0], iwa, owa, sampling).astype('bool')
        # Load HiCAT aperture file
        aperture = np.ones_like(all_ote_images[0])    #TODO: load actual HiCAT aperture
        # Calculate segment pair tuples
        seg_pair_tuples = list(pastis.util.segment_pairs_non_repeating(37))

        # Define plotting limits
        vmin_psfs = 1e-8
        vmax_psfs = 1e-4

    else:
        raise ValueError("Only instruments 'LUVOIR' and 'HiCAT' are implemented for this animation.")

    matrix_anim = hcipy.FFMpegWriter('video.mp4', framerate=5)
    if display_mode == 'boxy':
        plt.figure(figsize=(15, 15))
    elif display_mode == 'stretch':
        plt.figure(figsize=(24, 8))

    cmap_matrix_anim = copy.copy(cm.get_cmap('Blues'))
    cmap_matrix_anim.set_bad(color='black')

    for i in progressbar.progressbar(range(len(seg_pair_tuples))):
        contrast_matrix_here = np.copy(contrast_matrix)

        plt.clf()

        if display_mode == 'boxy':
            plt.subplot(2, 2, 1)
        elif display_mode == 'stretch':
            plt.subplot(1, 3, 1)
        plt.title('Segmented mirror phase', fontsize=30)
        this_ote = np.ma.masked_where(aperture == 0, all_ote_images[i])    #TODO: add apodizer (and LS) to aperture
        plt.imshow(this_ote, cmap=cmap_matrix_anim)
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=30)

        if display_mode == 'boxy':
            plt.subplot(2, 2, 2)
        elif display_mode == 'stretch':
            plt.subplot(1, 3, 2)
        plt.title('Dark hole contrast', fontsize=30)
        plt.imshow(all_psf_images[i] * dh_mask, norm=LogNorm(), cmap='inferno', vmin=vmin_psfs, vmax=vmax_psfs)
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046,
                            pad=0.04)  # no clue what these numbers mean but it did the job of adjusting the colorbar size to the actual plot size
        cbar.ax.tick_params(labelsize=30)

        # I need only the matrix elements up to and including the current iteration
        # So I null the rest
        contrast_matrix_here[seg_pair_tuples[i][0] + 1:, :] = 0
        contrast_matrix_here[seg_pair_tuples[i][0]:, seg_pair_tuples[i][1] + 1:] = 0

        if display_mode == 'boxy':
            plt.subplot(2, 1, 2)
        elif display_mode == 'stretch':
            plt.subplot(1, 3, 3)
        plt.title('Contrast matrix', fontsize=30)
        plt.imshow(contrast_matrix_here, cmap='Greys')
        plt.xlabel('Segments', size=30)
        plt.ylabel('Segments', size=30)
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
        # cbar = plt.colorbar(fraction=0.046, pad=0.04)    # no clue what these numbers mean but it did the job of adjusting the colorbar size to the actual plot size
        # cbar.ax.tick_params(labelsize=30)
        # cbar.ax.yaxis.offsetText.set(size=25)   # this changes the base of ten on the colorbar
        #TODO: figure out whether to add colorbar to contrast matrix or not (above)

        plt.suptitle(instrument, fontsize=40)

        if i == 0:
            plt.tight_layout()

        matrix_anim.add_frame()

    plt.close()
    matrix_anim.close()

    # Tell us how long it took to finish.
    end_time = time.time()

    print(f'Runtime for animate_contrast_matrix(): {end_time - start_time}sec = {(end_time - start_time) / 60}min')
    print(f'Animation saved to {os.getcwd()}')


def animate_random_wfe_maps(data_path, c_target, instrument='LUVOIR', design='small', display_mode='stretch'):
    """
    Create animation of the drawing of a random WFE map following the my map, and save to MP4 file.
    :param data_path: string, absolute path to the directory that contains the segment requirements txt file
    :param c_target: float, target contrast the segment constraints were calculated for
    :param instrument: string, "LUVOIR"
    :param design: string, necessary if instrument='LUVOIR', defaults to "small" - LUVOIR APLC design choice
    :param display_mode: string, 'boxy' for two panels on top, one on bottom, 'stretch' for all three panels in one row
    """

    # Keep track of time
    start_time = time.time()

    # Load the mu map
    mu_map = np.loadtxt(os.path.join(data_path, f'segment_requirements_{c_target}.txt'))
    mu_min = np.min(mu_map)
    mu_max = np.max(mu_map)

    dist_mean = 0
    range_limits = mu_max + mu_max * 0.5
    wfe_range = np.linspace(-range_limits, range_limits, 1000)

    # Define some instrument specific parameters
    if instrument == 'LUVOIR':
        # Instantiate LUVOIR sim object
        sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
        optics_input = os.path.join(pastis.util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
        luvoir = LuvoirAPLC(optics_input, design, sampling)

    seg_weights_all = np.zeros_like(mu_map)
    wfe_maps_anim = hcipy.FFMpegWriter('video.mp4', framerate=5)
    plt.figure(figsize=(18, 6))

    for i in progressbar.progressbar(range(mu_map.shape[0])):

        plt.clf()

        # mu map
        partial_mu_map = np.copy(mu_map)
        partial_mu_map[i + 1:] = 0
        luvoir.flatten()
        wf_constraints = pastis.util.apply_mode_to_luvoir(partial_mu_map, luvoir)[0]
        map_small = (wf_constraints.phase / wf_constraints.wavenumber * 1e12).shaped  # in picometers
        map_small = np.ma.masked_where(map_small == 0, map_small)

        plt.subplot(1, 3, 1)
        plt.title('$\mu$ map', fontsize=30)
        plt.imshow(map_small, cmap=cmap_brev, norm=norm_center_zero)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=30)  # this changes the numbers on the colorbar
        cbar.ax.yaxis.offsetText.set(size=20)  # this changes the base of ten on the colorbar
        cbar.set_label('picometers', size=20)
        plt.clim(mu_min * 1e3, mu_max * 1e3)  # in pm
        plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=20)
        plt.axis('off')

        # Normal distribution
        dist_stddev = mu_map[i]
        pdf = norm.pdf(wfe_range, dist_mean, dist_stddev)

        plt.subplot(1, 3, 2)
        plt.title('$\mu_k$ as stddev', fontsize=30)
        plt.plot(wfe_range, pdf)
        plt.axvline(dist_mean, c='r', ls='-.', lw=3)
        plt.axvline(dist_mean + dist_stddev, c='darkorange', ls=':', lw=3)
        plt.axvline(dist_mean - dist_stddev, c='darkorange', ls=':', lw=3)

        # Random WFE map
        segments_random_state = np.random.RandomState()
        seg_weights_all[i] = segments_random_state.normal(0, dist_stddev)
        if i < 89:
            vmin = -4e-4
            vmax = 4e-4
        else:
            vmin = -0.0015
            vmax = 0.0015

        plt.subplot(1, 3, 3)
        plt.title('$a_k \sim \mathcal{N}(0,\mu_k)$', fontsize=30)
        one_mode = pastis.util.apply_mode_to_luvoir(seg_weights_all, luvoir)[0]
        hcipy.imshow_field(one_mode.phase, cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)  # this changes the numbers on the colorbar

        # plt.suptitle(instrument, fontsize=40)

        wfe_maps_anim.add_frame()

    plt.close()
    wfe_maps_anim.close()

    # Tell us how long it took to finish.
    end_time = time.time()

    print(f'Runtime for animate_contrast_matrix(): {end_time - start_time}sec = {(end_time - start_time) / 60}min')
    print(f'Animation saved to {os.getcwd()}')


def read_ote_fits_files(data_path):
    """
    Read OTE fits files from a PASTIS matrix calculation and return as list of arrays.
    :param data_path: string, path to PASTIS folder containing subdir "matrix_numerical" ff
    :return: all_ote_images, list of images arrays
    """
    all_ote_images = []
    try:
        all_filenames = glob.glob(os.path.join(data_path, 'matrix_numerical', 'OTE_images', 'fits', '*.fits'))
    except FileNotFoundError:
        raise FileNotFoundError('The OTE files either do not exist, or the directory structure is different than assumed.')

    # Sort the filenames by human numbering
    all_filenames.sort(key=natural_keys)

    # https://stackoverflow.com/a/55489469/10112569
    for filename in all_filenames:
        with fits.open(filename, memmap=False) as hdulist:
            all_ote_images.append(hdulist[0].data)
        del hdulist[0].data

    return all_ote_images


def read_psf_fits_files(data_path):
    """
    Read PSF fits files from a PASTIS matrix calculation and return as list of arrays.
    :param data_path: string, path to PASTIS folder containing subdir "matrix_numerical" ff
    :return: all_psf_images, list of images arrays
    """
    all_psf_images = []
    try:
        all_filenames = glob.glob(os.path.join(data_path, 'matrix_numerical', 'psfs', '*.fits'))
    except FileNotFoundError:
        raise FileNotFoundError('The OTE files either do not exist, or the directory structure is different than assumed.')

    # Sort the filenames by human numbering
    all_filenames.sort(key=natural_keys)

    # https://stackoverflow.com/a/55489469/10112569
    for filename in all_filenames:
        with fits.open(filename, memmap=False) as hdulist:
            all_psf_images.append(hdulist[0].data)
        del hdulist[0].data

    return all_psf_images


def atoi(text):
    # Taken from jost-package
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    # Taken from jost-package
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (from stack overflow:
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def plot_multimode_surface_maps(tel, mus, num_modes, mirror, cmin, cmax, data_dir=None, fname=None):
    """
    Creates surface deformation maps (not WFE) for localized wavefront aberrations.

    The input mode coefficients 'mus' are in units of *WFE* and need to be grouped by segment, meaning the array holds
    the mode coefficients as:
        mode1 on seg1, mode2 on seg1, ..., mode'nmodes' on seg1, mode1 on seg2, mode2 on seg2 and so on.

    Parameters:
    -----------
    tel : class instance of internal simulator
        the simulator to plot the surface maps for
    mus : 1d array
        1d array of standard deviations for all modes on each segment, in nm WFE
    num_modes : int
        number of local modes used to poke each segment
    mirror : str
        'harris_seg_mirror' or 'seg_mirror', segmented mirror of simulator 'tel' to use for plotting
    cmin : float
        minimum value for colorbar
    cmax : float
        maximum value for colorbar
    data_dir : str, default None
        path to save the plots; if None, then not saved to disk
    fname : str, default None
        file name for surface maps saved to disk
    """
    if fname is None:
        fname = f'surface_on_{mirror}'

    coeffs_mumaps = pastis.util.sort_1d_mus_per_actuator(mus, num_modes, tel.nseg)  # in nm

    mu_maps = []
    for mode in range(num_modes):
        coeffs = coeffs_mumaps[mode]
        if mirror == 'harris_seg_mirror':
            tel.harris_sm.actuators = coeffs / 2
            mu_maps.append(tel.harris_sm.surface)  # in m
        if mirror == 'seg_mirror':
            tel.sm.actuators = coeffs / 2
            mu_maps.append(tel.sm.surface)  # in m

    plot_norm = TwoSlopeNorm(vcenter=0, vmin=cmin, vmax=cmax)
    for i in range(num_modes):
        plt.figure(figsize=(7, 5))
        hcipy.imshow_field((mu_maps[i]) * 1e12, norm=plot_norm, cmap='RdBu')
        plt.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Surface (pm)", fontsize=10)
        plt.tight_layout()

        if data_dir is not None:
            fname += f'_mode_{i}.pdf'
            plt.savefig(os.path.join(data_dir, 'mu_maps', fname))
