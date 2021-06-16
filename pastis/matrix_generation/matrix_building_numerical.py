"""
This module contains functions that construct the matrix M for PASTIS *NUMERICALLY FROM THE RESPECTIVE E2E SIMULATOR*
and saves it.

 Currently supported:
 JWST
 LUVOIR
 HiCAT
"""

from abc import ABC, abstractmethod
import os
import time
import functools
import shutil
import logging
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import hcipy

from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.e2e_simulators.hicat_imaging import set_up_hicat
from pastis.e2e_simulators.luvoir_imaging import LuvoirAPLC
import pastis.e2e_simulators.webbpsf_imaging as webbpsf_imaging
import pastis.plotting as ppl

log = logging.getLogger()
matplotlib.rc('image', origin='lower')
matplotlib.rc('pdf', fonttype=42)


class PastisMatrix(ABC):
    instrument = None

    def __init__(self, design=None, initial_path=''):

        # General telescope parameters
        self.design = design
        self.nb_seg = CONFIG_PASTIS.getint(self.instrument, 'nb_subapertures')
        self.seglist = util.get_segment_list(self.instrument)
        self.wvln = CONFIG_PASTIS.getfloat(self.instrument, 'lambda') * 1e-9  # m
        self.wfe_aber = CONFIG_PASTIS.getfloat(self.instrument, 'calibration_aberration') * 1e-9  # m

        # Create directory names
        tel_suffix = f'{self.instrument.lower()}'
        if self.instrument == 'LUVOIR':
            if design is None:
                design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
            tel_suffix += f'-{design}'
        self.overall_dir = util.create_data_path(initial_path, telescope=tel_suffix)
        os.makedirs(self.overall_dir, exist_ok=True)
        self.resDir = os.path.join(self.overall_dir, 'matrix_numerical')

        # Create necessary directories if they don't exist yet
        os.makedirs(self.resDir, exist_ok=True)
        os.makedirs(os.path.join(self.resDir, 'OTE_images'), exist_ok=True)

        # Set up logger
        util.setup_pastis_logging(self.resDir, f'pastis_matrix_{tel_suffix}')
        log.info(f'Building numerical matrix for {tel_suffix}\n')

        # Record some of the defined parameters
        log.info(f'Instrument: {tel_suffix}')
        log.info(f'Wavelength: {self.wvln} m')
        log.info(f'Number of segments: {self.nb_seg}')
        log.info(f'Segment list: {self.seglist}')
        log.info(f'wfe_aber: {self.wfe_aber} m')

        # Copy configfile to resulting matrix directory
        util.copy_config(self.resDir)

    @abstractmethod
    def calc(self):
        """ This is the main method that should be called to calculate a PASTIS matrix. """


class PastisMatrixIntensities(PastisMatrix):
    instrument = None

    def __init__(self, design=None, initial_path='', savepsfs=True, saveopds=True):
        super().__init__(design=design, initial_path=initial_path)

        self.savepsfs = savepsfs
        self.saveopds = saveopds
        self.calculate_matrix_pair = None

        os.makedirs(os.path.join(self.resDir, 'psfs'), exist_ok=True)
        log.info(f'Total number of actuator pairs in {self.instrument} pupil: {len(list(util.segment_pairs_all(self.nb_seg)))}')
        log.info(
            f'Non-repeating pairs in {self.instrument} pupil calculated here: {len(list(util.segment_pairs_non_repeating(self.nb_seg)))}')

    def calc(self):
        start_time = time.time()

        # Calculate coronagraph floor, and normalization factor from direct image
        self.calculate_ref_image()
        self.setup_one_pair_function()
        self.calculate_contrast_matrix()
        self.calculate_pastis_from_contrast_matrix()

        end_time = time.time()
        log.info(
            f'Runtime for PastisMatrixIntensities().calc(): {end_time - start_time}sec = {(end_time - start_time) / 60}min')
        log.info(f'Data saved to {self.resDir}')

    def calculate_contrast_matrix(self):

        # Figure out how many processes is optimal and create a Pool.
        # Assume we're the only one on the machine so we can hog all the resources.
        # We expect numpy to use multithreaded math via the Intel MKL library, so
        # we check how many threads MKL will use, and create enough processes so
        # as to use 100% of the CPU cores.
        # You might think we should divide number of cores by 2 to get physical cores
        # to account for hyperthreading, however empirical testing on telserv3 shows that
        # it is slightly more performant on telserv3 to use all logical cores
        num_cpu = multiprocessing.cpu_count()
        # try:
        #     import mkl
        #     num_core_per_process = mkl.get_max_threads()
        # except ImportError:
        #     # typically this is 4, so use that as default
        #     log.info("Couldn't import MKL; guessing default value of 4 cores per process")
        #     num_core_per_process = 4

        num_core_per_process = 1  # NOTE: this was changed by Scott Will in HiCAT and makes more sense, somehow
        num_processes = int(num_cpu // num_core_per_process)
        log.info(
            f"Multiprocess PASTIS matrix for {self.instrument} will use {num_processes} processes (with {num_core_per_process} threads per process)")

        # Iterate over all segment pairs via a multiprocess pool
        mypool = multiprocessing.Pool(num_processes)
        t_start = time.time()
        results = mypool.map(self.calculate_matrix_pair,
                             util.segment_pairs_non_repeating(self.nb_seg))  # this util function returns a generator
        t_stop = time.time()

        log.info(f"Multiprocess calculation complete in {t_stop - t_start}sec = {(t_stop - t_start) / 60}min")

        # Unscramble results
        # results is a list of tuples that contain the return from the partial function, in this case: result[i] = (c, (seg1, seg2))
        self.contrast_matrix = np.zeros([self.nb_seg, self.nb_seg])  # Generate empty matrix
        for i in range(len(results)):
            # Fill according entry in the contrast matrix
            self.contrast_matrix[results[i][1][0], results[i][1][1]] = results[i][0]
        mypool.close()

        # Save all contrasts to disk, WITHOUT subtraction of coronagraph floor
        hcipy.write_fits(self.contrast_matrix, os.path.join(self.resDir, 'contrast_matrix.fits'))
        plt.figure(figsize=(10, 10))
        plt.imshow(self.contrast_matrix)
        plt.colorbar()
        plt.savefig(os.path.join(self.resDir, 'contrast_matrix.pdf'))

    def calculate_pastis_from_contrast_matrix(self):

        # Calculate the PASTIS matrix from the contrast matrix: analytical matrix element calculation and normalization
        self.matrix_pastis = pastis_from_contrast_matrix(self.contrast_matrix, self.seglist, self.wfe_aber, float(self.contrast_floor))

        # Save matrix to file
        filename_matrix = f'pastis_matrix'
        hcipy.write_fits(self.matrix_pastis, os.path.join(self.resDir, filename_matrix + '.fits'))
        ppl.plot_pastis_matrix(self.matrix_pastis, self.wvln * 1e9, out_dir=self.resDir, save=True)  # convert wavelength to nm
        log.info(f'PASTIS matrix saved to: {os.path.join(self.resDir, filename_matrix + ".fits")}')

    @abstractmethod
    def calculate_ref_image(self):
        """This method needs to create the attributes self.norm, self.contrast_floor and self.coro_simulator."""

    @abstractmethod
    def setup_one_pair_function(self):
        """This needs to create an attribute that is the partial function that can calculate the contrast from one
        aberrated segment/actuator pair. This needs to create self.calculate_matrix_pair."""


def calculate_unaberrated_contrast_and_normalization(instrument, design=None, return_coro_simulator=True, save_coro_floor=False, save_psfs=False, outpath=''):
    """
    Calculate the direct PSF peak and unaberrated coronagraph floor of an instrument.
    :param instrument: string, 'LUVOIR', 'HiCAT', 'RST' or 'JWST'
    :param design: str, optional, default=None, which means we read from the configfile: what coronagraph design
                   to use - 'small', 'medium' or 'large'
    :param return_coro_simulator: bool, whether to return the coronagraphic simulator as third return, default True
    :param save_coro_floor: bool, if True, will save coro floor value to txt file, default False
    :param save_psfs: bool, if True, will save direct and coro PSF images to disk, default False
    :param outpath: string, where to save outputs to if save=True
    :return: contrast floor and PSF normalization factor, and optionally (by default) the simulator in coron mode
    """

    if instrument == 'LUVOIR':
        # Instantiate LuvoirAPLC class
        sampling = CONFIG_PASTIS.getfloat(instrument, 'sampling')
        optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
        if design is None:
            design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
        luvoir = LuvoirAPLC(optics_input, design, sampling)

        # Calculate reference images for contrast normalization and coronagraph floor
        unaberrated_coro_psf, direct = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=None)
        norm = np.max(direct)
        direct_psf = direct.shaped
        coro_psf = unaberrated_coro_psf.shaped / norm

        # Return the coronagraphic simulator and DH mask
        coro_simulator = luvoir
        dh_mask = luvoir.dh_mask.shaped

    if instrument == 'HiCAT':
        # Set up HiCAT simulator in correct state
        hicat_sim = set_up_hicat(apply_continuous_dm_maps=True)

        # Calculate direct reference images for contrast normalization
        hicat_sim.include_fpm = False
        direct = hicat_sim.calc_psf()
        direct_psf = direct[0].data
        norm = direct_psf.max()

        # Calculate unaberrated coronagraph image for contrast floor
        hicat_sim.include_fpm = True
        coro_image = hicat_sim.calc_psf()
        coro_psf = coro_image[0].data / norm

        iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
        owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
        sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')
        dh_mask = util.create_dark_hole(coro_psf, iwa, owa, sampling).astype('bool')

        # Return the coronagraphic simulator
        coro_simulator = hicat_sim

    if instrument == 'JWST':

        # Instantiate NIRCAM object
        jwst_sim = webbpsf_imaging.set_up_nircam()  # this returns a tuple of two: jwst_sim[0] is the nircam object, jwst_sim[1] its ote

        # Calculate direct reference images for contrast normalization
        jwst_sim[0].image_mask = None
        direct = jwst_sim[0].calc_psf(nlambda=1)
        direct_psf = direct[0].data
        norm = direct_psf.max()

        # Calculate unaberrated coronagraph image for contrast floor
        jwst_sim[0].image_mask = CONFIG_PASTIS.get('JWST', 'focal_plane_mask')
        coro_image = jwst_sim[0].calc_psf(nlambda=1)
        coro_psf = coro_image[0].data / norm

        iwa = CONFIG_PASTIS.getfloat('JWST', 'IWA')
        owa = CONFIG_PASTIS.getfloat('JWST', 'OWA')
        sampling = CONFIG_PASTIS.getfloat('JWST', 'sampling')
        dh_mask = util.create_dark_hole(coro_psf, iwa, owa, sampling).astype('bool')

        # Return the coronagraphic simulator (a tuple in the JWST case!)
        coro_simulator = jwst_sim

    if instrument == 'RST':

        # Instantiate CGI object
        rst_sim = webbpsf_imaging.set_up_cgi()

        # Calculate direct reference images for contrast normalization
        rst_direct = rst_sim.raw_PSF()
        direct = rst_direct.calc_psf(nlambda=1, fov_arcsec=1.6)
        direct_psf = direct[0].data
        norm = direct_psf.max()

        # Calculate unaberrated coronagraph image for contrast floor
        coro_image = rst_sim.calc_psf(nlambda=1, fov_arcsec=1.6)
        coro_psf = coro_image[0].data / norm

        iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
        owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
        rst_sim.working_area(im=coro_psf, inner_rad=iwa, outer_rad=owa)
        dh_mask = rst_sim.WA

        # Return the coronagraphic simulator (a tuple in the RST case!)
        coro_simulator = rst_sim

    # Calculate coronagraph floor in dark hole
    contrast_floor = util.dh_mean(coro_psf, dh_mask)
    log.info(f'contrast floor: {contrast_floor}')

    if save_coro_floor:
        # Save contrast floor to text file
        with open(os.path.join(outpath, 'coronagraph_floor.txt'), 'w') as file:
            file.write(f'Coronagraph floor: {contrast_floor}')

    if save_psfs:
        ppl.plot_direct_coro_dh(direct_psf, coro_psf, dh_mask, outpath)

    if return_coro_simulator:
        return contrast_floor, norm, coro_simulator
    else:
        return contrast_floor, norm


def _jwst_matrix_one_pair(norm, wfe_aber, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate JWST mean contrast of one aberrated segment pair in NIRCam; for PastisMatrixIntensities().
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how JWST segments start numbering at 0 just because that's python indexing, with 0 being
                         the segment A1.
    :return: contrast as float, and segment pair as tuple
    """

    # Set up JWST simulator in coronagraphic state
    jwst_instrument, jwst_ote = webbpsf_imaging.set_up_nircam()
    jwst_instrument.image_mask = CONFIG_PASTIS.get('JWST', 'focal_plane_mask')

    # Put aberration on correct segments. If i=j, apply only once!
    log.info(f'PAIR: {segment_pair[0]}-{segment_pair[1]}')

    # Identify the correct JWST segments
    seg_i = webbpsf_imaging.WSS_SEGS[segment_pair[0]].split('-')[0]
    seg_j = webbpsf_imaging.WSS_SEGS[segment_pair[1]].split('-')[0]

    # Put aberration on correct segments. If i=j, apply only once!
    jwst_ote.zero()
    jwst_ote.move_seg_local(seg_i, piston=wfe_aber, trans_unit='m')
    if segment_pair[0] != segment_pair[1]:
        jwst_ote.move_seg_local(seg_j, piston=wfe_aber, trans_unit='m')

    log.info('Calculating coro image...')
    image = jwst_instrument.calc_psf(nlambda=1)
    psf = image[0].data / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        plt.clf()
        plt.figure(figsize=(8, 8))
        ax2 = plt.subplot(111)
        jwst_ote.display_opd(ax=ax2, vmax=500, colorbar_orientation='horizontal', title='Aberrated segment pair')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    iwa = CONFIG_PASTIS.getfloat('JWST', 'IWA')
    owa = CONFIG_PASTIS.getfloat('JWST', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('JWST', 'sampling')
    dh_mask = util.create_dark_hole(psf, iwa, owa, sampling)
    contrast = util.dh_mean(psf, dh_mask)

    return contrast, segment_pair


def _luvoir_matrix_one_pair(design, norm, wfe_aber, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate LVUOIR-A mean contrast of one aberrated segment pair; for PastisMatrixIntensities().
    :param design: str, what coronagraph design to use - 'small', 'medium' or 'large'
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: float, calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how LUVOIR segments start numbering at 1, with 0 being the center segment that doesn't exist.
    :return: contrast as float, and segment pair as tuple
    """

    # Instantiate LUVOIR object
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    luv = LuvoirAPLC(optics_input, design, sampling)

    log.info(f'PAIR: {segment_pair[0]+1}-{segment_pair[1]+1}')

    # Put aberration on correct segments. If i=j, apply only once!
    luv.flatten()
    luv.set_segment(segment_pair[0]+1, wfe_aber / 2, 0, 0)
    if segment_pair[0] != segment_pair[1]:
        luv.set_segment(segment_pair[1]+1, wfe_aber / 2, 0, 0)

    log.info('Calculating coro image...')
    image, inter = luv.calc_psf(ref=False, display_intermediate=False, return_intermediate='intensity')
    # Normalize PSF by reference image
    psf = image / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_segs_{segment_pair[0]+1}-{segment_pair[1]+1}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_segs_{segment_pair[0]+1}-{segment_pair[1]+1}'
        plt.clf()
        hcipy.imshow_field(inter['seg_mirror'], grid=luv.aperture.grid, mask=luv.aperture, cmap='RdBu')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    dh_intensity = psf * luv.dh_mask
    contrast = np.mean(dh_intensity[np.where(luv.dh_mask != 0)])
    log.info(f'contrast: {float(contrast)}')    # contrast is a Field, here casting to normal float

    return float(contrast), segment_pair


def _hicat_matrix_one_pair(norm, wfe_aber, resDir, savepsfs, saveopds, segment_pair):
    """
    Function to calculate HiCAT mean contrast of one aberrated segment pair; for PastisMatrixIntensities().
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param segment_pair: tuple, pair of segments to aberrate, 0-indexed. If same segment gets passed in both tuple
                         entries, the segment will be aberrated only once.
                         Note how HiCAT segments start numbering at 0, with 0 being the center segment.
    :return: contrast as float, and segment pair as tuple
    """

    # Set up HiCAT simulator in correct state
    hicat_sim = set_up_hicat(apply_continuous_dm_maps=True)
    hicat_sim.include_fpm = True

    # Put aberration on correct segments. If i=j, apply only once!
    log.info(f'PAIR: {segment_pair[0]}-{segment_pair[1]}')
    hicat_sim.iris_dm.flatten()
    hicat_sim.iris_dm.set_actuator(segment_pair[0], wfe_aber, 0, 0)
    if segment_pair[0] != segment_pair[1]:
        hicat_sim.iris_dm.set_actuator(segment_pair[1], wfe_aber, 0, 0)

    log.info('Calculating coro image...')
    image, inter = hicat_sim.calc_psf(display=False, return_intermediates=True)
    psf = image[0].data / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot segmented mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_piston_Noll1_segs_{segment_pair[0]}-{segment_pair[1]}'
        plt.clf()
        plt.imshow(inter[1].phase)
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    iwa = CONFIG_PASTIS.getfloat('HiCAT', 'IWA')
    owa = CONFIG_PASTIS.getfloat('HiCAT', 'OWA')
    sampling = CONFIG_PASTIS.getfloat('HiCAT', 'sampling')
    dh_mask = util.create_dark_hole(psf, iwa, owa, sampling)
    contrast = util.dh_mean(psf, dh_mask)

    return contrast, segment_pair


def _rst_matrix_one_pair(norm, wfe_aber, resDir, savepsfs, saveopds, actuator_pair):
    """
    Function to calculate RST mean contrast of one DM actuator pair in CGI.
    :param norm: float, direct PSF normalization factor (peak pixel of direct PSF)
    :param wfe_aber: calibration aberration per segment in m
    :param resDir: str, directory for matrix calculations
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :param actuator_pair:
    :return: contrast as float, and segment pair as tuple
    """

    # Set up RST simulator in coronagraphic state
    rst_cgi = webbpsf_imaging.set_up_cgi()

    # Put aberration on correct segments. If i=j, apply only once!
    log.info(f'PAIR: {actuator_pair[0]}-{actuator_pair[1]}')

    # Transform single-actuator index to x|y coordinate on DM
    nb_actu = rst_cgi.nbactuator
    actu_i_x, actu_i_y = util.seg_to_dm_xy(nb_actu, actuator_pair[0])
    actu_j_x, actu_j_y = util.seg_to_dm_xy(nb_actu, actuator_pair[1])

    # Put aberration on correct segments. If i=j, apply only once!
    rst_cgi.dm1.flatten()
    rst_cgi.dm1.set_actuator(actu_i_x, actu_i_y, wfe_aber)
    if actuator_pair[0] != actuator_pair[1]:
        rst_cgi.dm1.set_actuator(actu_j_x, actu_j_y, wfe_aber)

    log.info('Calculating coro image...')
    image = rst_cgi.calc_psf(nlambda=1, fov_arcsec=1.6)   # fov number taken from: https://github.com/spacetelescope/webbpsf/blob/5cdd41ef9643e1ef42ea6232890ce740515fb896/notebooks/roman_cgi_demo.ipynb#L289
    psf = image[0].data / norm

    # Save PSF image to disk
    if savepsfs:
        filename_psf = f'psf_actuator_{actuator_pair[0]}-{actuator_pair[1]}'
        hcipy.write_fits(psf, os.path.join(resDir, 'psfs', filename_psf + '.fits'))

    # Plot deformable mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_actuator_{actuator_pair[0]}-{actuator_pair[1]}'
        plt.clf()
        plt.figure(figsize=(8, 8))
        rst_cgi.dm1.display(what='opd', opd_vmax=wfe_aber, colorbar_orientation='horizontal', title='Aberrated actuator pair')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    log.info('Calculating mean contrast in dark hole')
    iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
    owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
    rst_cgi.working_area(im=psf, inner_rad=iwa, outer_rad=owa)
    dh_mask = rst_cgi.WA
    contrast = util.dh_mean(psf, dh_mask)

    return contrast, actuator_pair


def pastis_from_contrast_matrix(contrast_matrix, seglist, wfe_aber, coro_floor):
    """
    Calculate the final PASTIS matrix from the input contrast matrix (only half filled).

    The contrast matrix is a nseg x nseg matrix where only half of it is filled, including the diagonal, and the other
    half is filled with zeros. It holds the DH mean contrast values of each aberrated segment pair, with a WFE
    amplitude of the calibration aberration wfe_aber, in m. Hence, the input contrast matrix is not normalized to the
    desired units yet. The coronagraph floor is NOT subtracted from it at this point yet, but also passed in.
    This function first normalizes the contrast matrix and coro floor by the calibration aberration to get a matrix with
    units of contrast / nm^2. Then, it calculates the PASTIS matrix, and it adapts to whether we assume a static
    coronagraph floor (passed in as a single float) or a drifting coronagraph floor (passed in as a 2D array, per
    segment pair measurement). Finally, it symmetrizes the matrix to output the full PASTIS matrix.

    :param contrast_matrix: nd.array, nseg x nseg matrix holding DH mean contast values of all aberrated segment pairs;
                            only half of the matrix has non-zero values, contrast floor NOT SUBTRACTED yet.
    :param seglist: list of segment indices (e.g. 0, 1, 2, ...36 [HiCAT]; or 1, 2, ..., 120 [LUVOIR])
    :param wfe_aber: float, calibration aberration in m, this is the aberration that was used to generate contrast_matrix
    :param coro_floor: float, or nd.array of same dims like contrast_matrix. In simulations we usually assume a static
                       coronagraph floor across the measurements for the PASTIS matrix and can use a single number.
                       When we have a drifting coro floor though, we need one contrast floor number per measurement.
    :return: the finalized PASTIS matrix in units of contrast per nanometers squared, ndarray of nseg x nseg
    """

    # Normalize contrast matrix, and coronagraph floor, to the input aberration - this defines what units the PASTIS
    # matrix will be in. The PASTIS matrix propagation function (util.pastis_contrast()) then needs to take in the
    # aberration vector in these same units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to
    # units of nanometers (contrast / nanometers squared).
    log.info('Normalizing contrast matrix and coronagraph floor')
    contrast_matrix /= np.square(wfe_aber * 1e9)  # 1e9 converts the calibration aberration back to nanometers
    coro_floor /= np.square(wfe_aber * 1e9)

    # Calculate the semi-analytical PASTIS matrix from the contrast matrix
    log.info('Calculating the semi-analytical PASTIS matrix from the contrast matrix')
    matrix_pastis_half = calculate_semi_analytic_pastis_from_contrast(contrast_matrix, seglist, coro_floor)

    # Symmetrize the half-PASTIS matrix
    log.info('Symmetrizing PASTIS matrix')
    matrix_pastis = util.symmetrize(matrix_pastis_half)

    return matrix_pastis


def calculate_semi_analytic_pastis_from_contrast(contrast_matrix, seglist, coro_floor):
    """
    Perform the semi-analytical calculation to go from contrast matrix to PASTIS matrix, depending on assumption for
    coronagraph floor drift.

    This function calculates the elements of the (half !) PASTIS matrix from the contrast matrix in which the
    coronagraph has not been subtracted yet. The calculation is only performed on the half-PASTIS matrix, so it will
    need to be symmetrized after this step.
    The calculation is slightly different for the two cases in which the coronagraph is either assumed to be constant
    across all pair-wise aberrated measurements, or is drifting.

    :param contrast_matrix: nd.array, nseg x nseg matrix holding DH mean contrast values of all aberrated segment pairs
    :param seglist: list of segment indices (e.g. 0, 1, 2, ...36 [HiCAT]; or 1, 2, ..., 120 [LUVOIR])
    :param coro_floor: float or nd.array of same dims like contrast_matrix. In simulations we usually assume a static
                       coronagraph floor across the measurements for the PASTIS matrix and can use a single number.
                       When we have a drifting coro floor though, we need one contrast floor number per measurement.
    :return: half-PASTIS matrix, nd.array of nseg x nseg where one of its matrix triangles will be all zeros
    """

    # Create future (half filled) PASTIS matrix
    matrix_pastis_half = np.zeros_like(contrast_matrix)

    # Assuming constant coronagraph floor across all pair-aberrated measurements
    if isinstance(coro_floor, float):
        log.info('coro_floor is a constant --> float')

        # First calculate the on-axis elements, which just need to have the coronagraph floor subtracted
        np.fill_diagonal(matrix_pastis_half, np.diag(contrast_matrix) - coro_floor)
        log.info('On-axis elements of PASTIS matrix calculated')

        # Calculate the off-axis elements in the (half) PASTIS matrix
        for pair in util.segment_pairs_non_repeating(contrast_matrix.shape[0]):    # this util function returns a generator
            if pair[0] != pair[1]:    # exclude diagonal elements
                matrix_off_val = (contrast_matrix[pair[0], pair[1]] + coro_floor - contrast_matrix[pair[0], pair[0]] - contrast_matrix[pair[1], pair[1]]) / 2.
                matrix_pastis_half[pair[0], pair[1]] = matrix_off_val
                log.info(f'Off-axis for i{seglist[pair[0]]}-j{seglist[pair[1]]}: {matrix_off_val}')

    # Assuming drifting coronagraph floor across all pair-aberrated measurements  #TODO: this is untested
    elif isinstance(coro_floor, np.ndarray):
        log.info('coro_floor is drifting --> np.ndarray')

        # Check that the coro_floor array has same dimensions like the contrast_matrix array
        if coro_floor.shape != contrast_matrix.shape:
            raise ValueError('coro_floor needs to have same dimensions like contrast_matrix')

        # First calculate the on-axis elements, which just need to have the coronagraph floor subtracted
        np.fill_diagonal(matrix_pastis_half, np.diag(contrast_matrix) - np.diag(coro_floor))
        log.info('On-axis elements of PASTIS matrix calculated')

        for pair in util.segment_pairs_non_repeating(contrast_matrix.shape[0]):    # this util function returns a generator

            # Then calculate the off-axis elements
            if pair[0] != pair[1]:    # exclude diagonal elements
                matrix_off_val = (contrast_matrix[pair[0], pair[1]] - coro_floor[pair[0], pair[1]] - matrix_pastis_half[pair[0], pair[0]] - matrix_pastis_half[pair[1], pair[1]]) / 2.
                matrix_pastis_half[pair[0], pair[1]] = matrix_off_val
                log.info(f'Off-axis for i{seglist[pair[0]]}-j{seglist[pair[1]]}: {matrix_off_val}')

    else:
        raise TypeError('"coro_floor" needs to be either a float, if working with a constant coronagraph floor, or an '
                        'ndarray, if working with a drifting coronagraph floor.')

    return matrix_pastis_half


def num_matrix_multiprocess(instrument, design=None, initial_path='', savepsfs=True, saveopds=True):
    """
    Generate a numerical/semi-analytical PASTIS matrix.

    -- DEPRECATED !! -- This function is deprecated, use the class PastisMatrixIntensities instead.

    Multiprocessed script to calculate PASTIS matrix. Implementation adapted from
    hicat.scripts.stroke_minimization.calculate_jacobian
    :param instrument: str, what instrument (LUVOIR, HiCAT, JWST) to generate the PASTIS matrix for
    :param design: str, optional, default=None, which means we read from the configfile: what coronagraph design
                   to use - 'small', 'medium' or 'large'
    :param initial_path: str, path to save results directory to
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files.
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk as PDF
    :return: overall_dir: string, experiment directory
    """

    # Keep track of time
    start_time = time.time()

    # Create directory names
    tel_suffix = f'{instrument.lower()}'
    if instrument == 'LUVOIR':
        if design is None:
            design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
        tel_suffix += f'-{design}'
    overall_dir = util.create_data_path(initial_path, telescope=tel_suffix)
    os.makedirs(overall_dir, exist_ok=True)
    resDir = os.path.join(overall_dir, 'matrix_numerical')

    # Create necessary directories if they don't exist yet
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(os.path.join(resDir, 'OTE_images'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'psfs'), exist_ok=True)

    # Set up logger
    util.setup_pastis_logging(resDir, f'pastis_matrix_{tel_suffix}')
    log.info(f'Building numerical matrix for {tel_suffix}\n')

    # General telescope parameters
    nb_seg = CONFIG_PASTIS.getint(instrument, 'nb_subapertures')
    seglist = util.get_segment_list(instrument)
    wvln = CONFIG_PASTIS.getfloat(instrument, 'lambda') * 1e-9  # m
    wfe_aber = CONFIG_PASTIS.getfloat(instrument, 'calibration_aberration') * 1e-9   # m

    # Record some of the defined parameters
    log.info(f'Instrument: {tel_suffix}')
    log.info(f'Wavelength: {wvln} m')
    log.info(f'Number of segments: {nb_seg}')
    log.info(f'Segment list: {seglist}')
    log.info(f'wfe_aber: {wfe_aber} m')
    log.info(f'Total number of segment pairs in {instrument} pupil: {len(list(util.segment_pairs_all(nb_seg)))}')
    log.info(f'Non-repeating pairs in {instrument} pupil calculated here: {len(list(util.segment_pairs_non_repeating(nb_seg)))}')

    #  Copy configfile to resulting matrix directory
    util.copy_config(resDir)

    # Calculate coronagraph floor, and normalization factor from direct image
    contrast_floor, norm = calculate_unaberrated_contrast_and_normalization(instrument, design, return_coro_simulator=False,
                                                                            save_coro_floor=True, save_psfs=False, outpath=overall_dir)

    # Figure out how many processes is optimal and create a Pool.
    # Assume we're the only one on the machine so we can hog all the resources.
    # We expect numpy to use multithreaded math via the Intel MKL library, so
    # we check how many threads MKL will use, and create enough processes so
    # as to use 100% of the CPU cores.
    # You might think we should divide number of cores by 2 to get physical cores
    # to account for hyperthreading, however empirical testing on telserv3 shows that
    # it is slightly more performant on telserv3 to use all logical cores
    num_cpu = multiprocessing.cpu_count()
    # try:
    #     import mkl
    #     num_core_per_process = mkl.get_max_threads()
    # except ImportError:
    #     # typically this is 4, so use that as default
    #     log.info("Couldn't import MKL; guessing default value of 4 cores per process")
    #     num_core_per_process = 4

    num_core_per_process = 1   # NOTE: this was changed by Scott Will in HiCAT and makes more sense, somehow
    num_processes = int(num_cpu // num_core_per_process)
    log.info(f"Multiprocess PASTIS matrix for {instrument} will use {num_processes} processes (with {num_core_per_process} threads per process)")

    # Set up a function with all arguments fixed except for the last one, which is the segment pair tuple
    if instrument == 'LUVOIR':
        calculate_matrix_pair = functools.partial(_luvoir_matrix_one_pair, design, norm, wfe_aber, resDir,
                                                  savepsfs, saveopds)

    if instrument == 'HiCAT':
        # Copy used BostonDM maps to matrix folder
        shutil.copytree(CONFIG_PASTIS.get('HiCAT', 'dm_maps_path'), os.path.join(resDir, 'hicat_boston_dm_commands'))

        calculate_matrix_pair = functools.partial(_hicat_matrix_one_pair, norm, wfe_aber, resDir, savepsfs, saveopds)

    if instrument == 'JWST':
        calculate_matrix_pair = functools.partial(_jwst_matrix_one_pair, norm, wfe_aber, resDir, savepsfs, saveopds)

    if instrument == 'RST':
        calculate_matrix_pair = functools.partial(_rst_matrix_one_pair, norm, wfe_aber, resDir, savepsfs, saveopds)

    # Iterate over all segment pairs via a multiprocess pool
    mypool = multiprocessing.Pool(num_processes)
    t_start = time.time()
    results = mypool.map(calculate_matrix_pair, util.segment_pairs_non_repeating(nb_seg))    # this util function returns a generator
    t_stop = time.time()

    log.info(f"Multiprocess calculation complete in {t_stop-t_start}sec = {(t_stop-t_start)/60}min")

    # Unscramble results
    # results is a list of tuples that contain the return from the partial function, in this case: result[i] = (c, (seg1, seg2))
    contrast_matrix = np.zeros([nb_seg, nb_seg])  # Generate empty matrix
    for i in range(len(results)):
        # Fill according entry in the contrast matrix
        contrast_matrix[results[i][1][0], results[i][1][1]] = results[i][0]
    mypool.close()

    # Save all contrasts to disk, WITHOUT subtraction of coronagraph floor
    hcipy.write_fits(contrast_matrix, os.path.join(resDir, 'contrast_matrix.fits'))
    plt.figure(figsize=(10, 10))
    plt.imshow(contrast_matrix)
    plt.colorbar()
    plt.savefig(os.path.join(resDir, 'contrast_matrix.pdf'))

    # Calculate the PASTIS matrix from the contrast matrix: analytical matrix element calculation and normalization
    matrix_pastis = pastis_from_contrast_matrix(contrast_matrix, seglist, wfe_aber, float(contrast_floor))

    # Save matrix to file
    filename_matrix = 'pastis_matrix'
    hcipy.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    ppl.plot_pastis_matrix(matrix_pastis, wvln*1e9, out_dir=resDir, save=True)    # convert wavelength to nm
    log.info(f'PASTIS matrix saved to: {os.path.join(resDir, filename_matrix + ".fits")}')

    # Tell us how long it took to finish.
    end_time = time.time()
    log.info(f'Runtime for matrix_building_numerical.py/multiprocess: {end_time - start_time}sec = {(end_time - start_time)/60}min')
    log.info(f'Data saved to {resDir}')

    return overall_dir


class MatrixIntensityLuvoirA(PastisMatrixIntensities):
    instrument = 'LUVOIR'

    def __int__(self, design='small', initial_path='', savepsfs=True, saveopds=True):
        super().__init__(design=design, savepsfs=savepsfs, saveopds=saveopds)

    def setup_one_pair_function(self):
        self.calculate_matrix_pair = functools.partial(_luvoir_matrix_one_pair, self.design, self.norm, self.wfe_aber,
                                                       self.resDir, self.savepsfs, self.saveopds)

    def calculate_ref_image(self, save_coro_floor=True, save_psfs=True):
        self.contrast_floor, self.norm, self.coro_simulator = calculate_unaberrated_contrast_and_normalization('LUVOIR',
                                                                                                               self.design,
                                                                                                               return_coro_simulator=True,
                                                                                                               save_coro_floor=save_coro_floor,
                                                                                                               save_psfs=save_psfs,
                                                                                                               outpath=self.overall_dir)


class MatrixIntensityHicat(PastisMatrixIntensities):
    instrument = 'HiCAT'

    def __int__(self, initial_path='', savepsfs=True, saveopds=True):
        super().__init__(design=None, savepsfs=savepsfs, saveopds=saveopds)

    def setup_one_pair_function(self):
        # Copy used BostonDM maps to matrix folder
        shutil.copytree(CONFIG_PASTIS.get('HiCAT', 'dm_maps_path'),
                        os.path.join(self.resDir, 'hicat_boston_dm_commands'))
        self.calculate_matrix_pair = functools.partial(_hicat_matrix_one_pair, self.norm, self.wfe_aber, self.resDir,
                                                       self.savepsfs, self.saveopds)

    def calculate_ref_image(self, save_coro_floor=True, save_psfs=True):
        self.contrast_floor, self.norm, self.coro_simulator = calculate_unaberrated_contrast_and_normalization('HiCAT',
                                                                                                               return_coro_simulator=True,
                                                                                                               save_coro_floor=save_coro_floor,
                                                                                                               save_psfs=save_psfs,
                                                                                                               outpath=self.overall_dir)


class MatrixIntensityJWST(PastisMatrixIntensities):
    instrument = 'JWST'

    def __int__(self, initial_path='', savepsfs=True, saveopds=True):
        super().__init__(design=None, savepsfs=savepsfs, saveopds=saveopds)

    def setup_one_pair_function(self):
        self.calculate_matrix_pair = functools.partial(_jwst_matrix_one_pair, self.norm, self.wfe_aber, self.resDir,
                                                       self.savepsfs, self.saveopds)

    def calculate_ref_image(self, save_coro_floor=True, save_psfs=True):
        self.contrast_floor, self.norm, self.coro_simulator = calculate_unaberrated_contrast_and_normalization('JWST',
                                                                                                               return_coro_simulator=True,
                                                                                                               save_coro_floor=save_coro_floor,
                                                                                                               save_psfs=save_psfs,
                                                                                                               outpath=self.overall_dir)


class MatrixIntensityRST(PastisMatrixIntensities):
    instrument = 'RST'

    def __int__(self, initial_path='', savepsfs=True, saveopds=True):
        super().__init__(design=None, savepsfs=savepsfs, saveopds=saveopds)

    def setup_one_pair_function(self):
        self.calculate_matrix_pair = functools.partial(_rst_matrix_one_pair, self.norm, self.wfe_aber, self.resDir,
                                                       self.savepsfs, self.saveopds)

    def calculate_ref_image(self, save_coro_floor=False, save_psfs=False):
        self.contrast_floor, self.norm, self.coro_simulator = calculate_unaberrated_contrast_and_normalization('RST',
                                                                                                               return_coro_simulator=True,
                                                                                                               save_coro_floor=save_coro_floor,
                                                                                                               save_psfs=save_psfs,
                                                                                                               outpath=self.overall_dir)


if __name__ == '__main__':

        MatrixIntensityLuvoirA(design='small', initial_path=CONFIG_PASTIS.get('local', 'local_data_path')).calc()
        #MatrixIntensityHicat(initial_path=CONFIG_PASTIS.get('local', 'local_data_path')).calc()
