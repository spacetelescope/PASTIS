from abc import abstractmethod
import os
import time
import functools
import logging
import hcipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.e2e_simulators.luvoir_imaging import LuvoirA_APLC
import pastis.e2e_simulators.webbpsf_imaging as webbpsf_imaging
from pastis.matrix_generation.matrix_building_numerical import PastisMatrix
import pastis.plotting as ppl
import pastis.util as util
import pastis.e2e_simulators.telescopes

log = logging.getLogger()
matplotlib.rc('image', origin='lower')
matplotlib.rc('pdf', fonttype=42)


class PastisMatrixEfields(PastisMatrix):
    instrument = None
    """ Main class for PASTIS matrix calculations from individually 'poked' modes. """

    def __init__(self, design=None, initial_path='', param=None):
        """
        Parameters:
        ----------
        design: string
            Default None; if instrument=='LUVOIR', need to pass "small", "medium" or "large"
        initial_path: string
            Path to top-level directory where result folder should be saved to.
        saveefields: bool
            Whether to save E-fields as fits file to disk or not
        saveopds: bool
            Whether to save images of pair-wise aberrated pupils to disk or not
        """
        super().__init__(design=design, initial_path=initial_path , param=param)

        self.param = param
        self.save_efields = param.saveefields
        self.saveopds = self.param.saveopds
        self.calculate_one_mode = None
        self.efields_per_mode = []

        os.makedirs(os.path.join(self.resDir, 'efields'), exist_ok=True)

    def calc(self):
        """ Main method that calculates the PASTIS matrix """

        start_time = time.time()

        self.calculate_ref_efield()
        self.setup_deformable_mirror()
        self.setup_single_mode_function()
        self.calculate_efields()
        self.calculate_pastis_matrix_from_efields()

        end_time = time.time()
        log.info(
            f'Runtime for PastisMatrixEfields().calc(): {end_time - start_time}sec = {(end_time - start_time) / 60}min')
        log.info(f'Data saved to {self.resDir}')

    def calculate_efields(self):
        """ Poke each mode individually and calculate the resulting focal plane E-field. """

        for i in range(self.number_all_modes):
            self.efields_per_mode.append(self.calculate_one_mode(i))
        self.efields_per_mode = np.array(self.efields_per_mode)

    def calculate_pastis_matrix_from_efields(self):
        """ Use the individual-mode E-fields to calculate the PASTIS matrix from it. """

        self.matrix_pastis = pastis_matrix_from_efields(self.efields_per_mode, self.efield_ref, self.telescope.norm,
                                                        self.telescope.dh_mask, self.wfe_aber)

        # Save matrix to file
        filename_matrix = f'pastis_matrix'
        hcipy.write_fits(self.matrix_pastis, os.path.join(self.resDir, filename_matrix + '.fits'))
        ppl.plot_pastis_matrix(self.matrix_pastis, self.wvln * 1e9, out_dir=self.resDir, save=True)  # convert wavelength to nm
        log.info(f'PASTIS matrix saved to: {os.path.join(self.resDir, filename_matrix + ".fits")}')

    @abstractmethod
    def calculate_ref_efield(self):
        """ Create the attributes self.norm, self.dh_mask, self.coro_simulator and self.efield_ref. """
        pass

    @abstractmethod
    def setup_deformable_mirror(self):
        """ Set up the deformable mirror for the modes you're using, if necessary, and define the total number of mode actuators. """
        pass

    @abstractmethod
    def setup_single_mode_function(self):
        """ Create an attribute that is the partial function that can calculate the focal plane E-field from one
        aberrated mode. This needs to create self.calculate_one_mode. """
        pass


def pastis_matrix_from_efields(electric_fields, efield_ref, direct_norm, dh_mask, wfe_aber):
    """ Calculate the semi-analytical PASTIS matrix from the individual E-fields
    :param electric_fields: list, items of same type as "efield_ref", individually poked mode E-fields
    :param efield_ref: same type as items in "electric_fields", reference E-field of an unaberrated system
    :param direct_norm: float, normalization factor - peak pixel of a direct PSF
    :param dh_mask: array, dark hole mask
    :param wfe_aber: float, calibration aberration in meters
    :return: full, normalized PASTIS matrix
    """

    matrix_pastis_half = calculate_semi_analytic_pastis_from_efields(electric_fields, efield_ref, direct_norm, dh_mask)

    # Symmetrize the half-PASTIS matrix
    log.info('Symmetrizing PASTIS matrix')
    matrix_pastis = util.symmetrize(matrix_pastis_half)

    # Normalize PASTIS matrix by input aberration
    matrix_pastis /= np.square(wfe_aber * 1e9)

    return matrix_pastis


def calculate_semi_analytic_pastis_from_efields(efields, efield_ref, direct_norm, dh_mask):
    """
    Perform the semi-analytical calculation to go from list of E-fields to PASTIS matrix.

    This function calculates the elements of the (half !) PASTIS matrix from the E-field responses of the individually
    poked modes, in which the reference E-field has not been subtracted yet. The calculation is only performed on the
    half-PASTIS matrix, so it will need to be symmetrized after this step.
    :param efields: list, items of same type as "efield_ref", individually poked mode E-fields
    :param efield_ref: same type as items in "efields", reference E-field of an unaberrated system
    :param direct_norm: float, normalization factor - peak pixel of a direct PSF
    :param dh_mask: array, dark hole mask
    :return: half-PASTIS matrix, where one of its matrix triangles will be all zeros
    """

    # Create empty matrix
    nb_modes = efields.shape[0]
    matrix_pastis_half = np.zeros([nb_modes, nb_modes])

    for pair in util.segment_pairs_non_repeating(nb_modes):
        intensity_im = np.real((efields[pair[0]] - efield_ref) * np.conj(efields[pair[1]] - efield_ref))
        contrast = util.dh_mean(intensity_im / direct_norm, dh_mask)
        matrix_pastis_half[pair[0], pair[1]] = contrast
        log.info(f'Calculated contrast for pair {pair[0]}-{pair[1]}: {contrast}')

    return matrix_pastis_half


class MatrixEfieldLuvoirA(PastisMatrixEfields):
    instrument = 'LUVOIR'
    """ Calculate a PASTIS matrix for LUVOIR-A, using E-fields. """

    def __init__(self, which_dm, dm_spec, design='small', initial_path='', saveefields=True, saveopds=True):
        """
        :param which_dm: string, which DM to calculate the matrix for - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
        :param dm_spec: tuple or int, specification for the used DM -
                        for seg_mirror: int, number of local Zernike modes on each segment
                        for harris_seg_mirror: tuple (string, array, bool, bool, bool), absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets
                        for zernike_mirror: int, number of global Zernikes
        :param design: str, what coronagraph design to use - 'small', 'medium' or 'large'
        :param initial_path: string, path to top-level directory where result folder should be saved to.
        :param saveefields: bool, whether to save E-fields as fits file to disk or not
        :param saveopds: bool, whether to save images of pair-wise aberrated pupils to disk or not
        """
        super().__init__(design=design, initial_path=initial_path, saveefields=saveefields, saveopds=saveopds)
        self.which_dm = which_dm
        self.dm_spec = dm_spec

    def calculate_ref_efield(self):
        """Instantiate the simulator object and calculate the reference E-field, DH mask, and direct PSF norm factor."""

        optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
        sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
        self.luvoir = LuvoirA_APLC(optics_input, self.design, sampling)
        self.dh_mask = self.luvoir.dh_mask

        # Calculate contrast normalization factor from direct PSF (intensity)
        _unaberrated_coro_psf, direct = self.luvoir.calc_psf(ref=True)
        self.norm = np.max(direct)

        # Calculate reference E-field in focal plane, without any aberrations applied
        unaberrated_ref_efield, _inter = self.luvoir.calc_psf(return_intermediate='efield')
        self.efield_ref = unaberrated_ref_efield.electric_field

    def setup_deformable_mirror(self):
        """ Set up the deformable mirror for the modes you're using and define the total number of mode actuators. """

        log.info('Setting up deformable mirror...')
        if self.which_dm == 'seg_mirror':
            n_modes_segs = self.dm_spec
            log.info(f'Creating segmented mirror with {n_modes_segs} local modes on each segment...')
            self.luvoir.create_segmented_mirror(n_modes_segs)
            self.number_all_modes = self.luvoir.sm.num_actuators

        elif self.which_dm == 'harris_seg_mirror':
            fpath, pad_orientations, therm, mech, other = self.dm_spec
            log.info(f'Reading Harris spreadsheet from {fpath}')
            log.info(f'Using pad orientations: {pad_orientations}')
            self.luvoir.create_segmented_harris_mirror(fpath, pad_orientations, therm, mech, other)
            self.number_all_modes = self.luvoir.harris_sm.num_actuators

        elif self.which_dm == 'zernike_mirror':
            n_modes_zernikes = self.dm_spec
            log.info(f'Creating global Zernike mirror with {n_modes_zernikes} global modes...')
            self.luvoir.create_global_zernike_mirror(n_modes_zernikes)
            self.number_all_modes = self.luvoir.zernike_mirror.num_actuators

        else:
            raise ValueError(f'DM with name "{self.which_dm}" not recognized.')

        log.info(f'Total number of modes: {self.number_all_modes}')

    def setup_single_mode_function(self):
        """ Create the partial function that returns the E-field of a single aberrated mode. """

        self.calculate_one_mode = functools.partial(_luvoir_matrix_single_mode, self.which_dm, self.number_all_modes,
                                                    self.wfe_aber, self.luvoir, self.resDir, self.save_efields,
                                                    self.saveopds)


class MatrixEfieldRST(PastisMatrixEfields):
    """
    Class to calculate the PASTIS matrix from E-fields of RST CGI.
    """
    instrument = 'RST'

    def __init__(self, initial_path='', saveefields=True, saveopds=True):
        super().__init__(initial_path=initial_path, saveefields=saveefields, saveopds=saveopds)

    def calculate_ref_efield(self):
        iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
        owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
        self.rst_cgi = webbpsf_imaging.set_up_cgi()

        # Calculate direct reference images for contrast normalization
        rst_direct = self.rst_cgi.raw_coronagraph()
        direct = rst_direct.calc_psf(nlambda=1, fov_arcsec=1.6)
        direct_psf = direct[0].data
        self.norm = direct_psf.max()

        # Calculate dark hole mask
        self.rst_cgi.working_area(im=direct_psf, inner_rad=iwa, outer_rad=owa)
        self.dh_mask = self.rst_cgi.WA

        # Calculate reference E-field in focal plane, without any aberrations applied
        _trash, inter = self.rst_cgi.calc_psf(nlambda=1, fov_arcsec=1.6, return_intermediates=True)
        self.efield_ref = inter[6].wavefront    # [6] is the last optic = detector

    def setup_deformable_mirror(self):
        """DM setup not needed for RST, just define number of total modes"""
        self.number_all_modes = CONFIG_PASTIS.getint('RST', 'nb_subapertures')

    def setup_single_mode_function(self):
        self.calculate_one_mode = functools.partial(_rst_matrix_single_mode, self.wfe_aber,
                                                    self.rst_cgi, self.resDir, self.save_efields, self.saveopds)


class MatrixEfield(PastisMatrixEfields):
    """
    Class to calculate the PASTIS matrix from E-fields.
    """
    instrument = CONFIG_PASTIS.get('telescope', 'name')

    def __int__(self, initial_path='', param=None):
        super().__init__(design=None, param=param)

    def calculate_ref_efield(self):
        # Calculate direct reference images for contrast normalization
        self.telescope = self.param.def_telescope()
        self.telescope.normalization_and_dark_hole()
        self.telescope.calculate_unaberrated_contrast()

        log.info(f'contrast floor: {self.telescope.contrast_floor}')

        # Calculate reference E-field in focal plane, without any aberrations applied
        self.efield_ref = self.telescope.imaging_efield()

    def setup_deformable_mirror(self):
        """DM setup not needed for RST, just define number of total modes"""
        self.number_all_modes = self.param.telescope.number_all_modes

    def setup_single_mode_function(self):
        self.calculate_one_mode = functools.partial(general_matrix_single_mode, self.telescope, self.wfe_aber,
                                                    self.resDir, self.save_efields, self.saveopds)


def _luvoir_matrix_single_mode(which_dm, number_all_modes, wfe_aber, luvoir_sim, resDir, saveefields, saveopds, mode_no):
    """
    Calculate the LUVOIR-A mean E-field of one aberrated mode; for PastisMatrixEfields().
    :param which_dm: string, which DM - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    :param number_all_modes: int, total number of all modes
    :param wfe_aber: float, calibration aberration in meters
    :param luvoir_sim: instance of LUVOIR simulator
    :param resDir: str, directory for matrix calculation results
    :param saveefields: bool, Whether to save E-fields as fits file to disk or not
    :param saveopds: bool, Whether to save images of pair-wise aberrated pupils to disk or not
    :param mode_no: int, which mode index to calculate the E-field for
    :return: complex array, resulting focal plane E-field
    """

    log.info(f'MODE NUMBER: {mode_no}')

    # Apply calibration aberration to used mode
    all_modes = np.zeros(number_all_modes)
    all_modes[mode_no] = wfe_aber / 2    # LUVOIR simulator takes aberrations in surface  #TODO: check that this is true for all the DMs
    if which_dm == 'seg_mirror':
        luvoir_sim.sm.actuators = all_modes
    elif which_dm == 'harris_seg_mirror':
        luvoir_sim.harris_sm.actuators = all_modes
    elif which_dm == 'zernike_mirror':
        luvoir_sim.zernike_mirror.actuators = all_modes
    else:
        raise ValueError(f'DM with name "{which_dm}" not recognized.')

    # Calculate coronagraphic E-field
    efield_focal_plane, inter = luvoir_sim.calc_psf(return_intermediate='efield')

    if saveefields:
        fname_real = f'efield_real_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.real, os.path.join(resDir, 'efields', fname_real + '.fits'))
        fname_imag = f'efield_imag_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.imag, os.path.join(resDir, 'efields', fname_imag + '.fits'))

    if saveopds:
        opd_map = inter[which_dm].phase
        opd_name = f'opd_mode_{mode_no}'
        plt.clf()
        hcipy.imshow_field(opd_map, grid=luvoir_sim.aperture.grid, mask=luvoir_sim.aperture, cmap='RdBu')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    return efield_focal_plane.electric_field


def _rst_matrix_single_mode(wfe_aber, rst_sim, resDir, saveefields, saveopds, mode_no):
    """
    Function to calculate RST Electrical field (E_field) of one DM actuator in CGI.
    :param wfe_aber: float, calibration aberration per actuator in m
    :param rst_sim: instance of CGI simulator
    :param resDir: str, directory for matrix calculations
    :param saveefields: bool, if True, all E_field will be saved to disk individually, as fits files
    :param savepods: bool, if True, all pupil surface maps of aberrated actuators pairs will be saved to disk as PDF
    :param mode_no: int, which aberrated actuator to calculate the E-field for
    """

    log.info(f'ACTUATOR NUMBER: {mode_no}')

    # Apply calibration aberration to used mode
    rst_sim.dm1.flatten()

    nb_actu = rst_sim.nbactuator
    actu_x, actu_y = util.seg_to_dm_xy(nb_actu, mode_no)
    rst_sim.dm1.set_actuator(actu_x, actu_y, wfe_aber)

    # Calculate coronagraphic E-field
    _psf, inter = rst_sim.calc_psf(nlambda=1, fov_arcsec=1.6, return_intermediates=True)
    efield_focal_plane = inter[6]    # [6] is the last optic = detector

    # Save E field image to disk
    if saveefields:
        fname_real = f'efield_real_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.wavefront.real, os.path.join(resDir, 'efields', fname_real + '.fits'))
        fname_imag = f'efield_imag_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.wavefront.imag, os.path.join(resDir, 'efields', fname_imag + '.fits'))

    # Plot deformable mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_actuator_{mode_no}'
        plt.clf()
        plt.figure(figsize=(nb_actu, nb_actu))
        rst_sim.dm1.display(what='opd', opd_vmax=wfe_aber, colorbar_orientation='horizontal',
                            title='Aberrated actuator pair')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    return efield_focal_plane.wavefront


def general_matrix_single_mode(telescope, wfe_aber, resDir, saveefields, saveopds, mode_no):
    """
    Function to calculate Electrical field (E_field) of one mode.
    :param telescope:
    :param wfe_aber: float, calibration aberration per actuator in m
    :param resDir: str, directory for matrix calculations
    :param saveefields: bool, if True, all E_field will be saved to disk individually, as fits files
    :param savepods: bool, if True, all pupil surface maps of aberrated actuators pairs will be saved to disk as PDF
    :param mode_no: int, which aberrated actuator to calculate the E-field for
    """

    log.info(f'MODE NUMBER: {mode_no}')

    # Apply calibration aberration to used mode
    telescope.flatten()
    telescope.push_mode(mode_no, wfe_aber)

    # Calculate coronagraphic E-field
    efield_focal_plane = telescope.imaging_efield()

    # Save E field image to disk
    if saveefields:
        fname_real = f'efield_real_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.real, os.path.join(resDir, 'efields', fname_real + '.fits'))
        fname_imag = f'efield_imag_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.imag, os.path.join(resDir, 'efields', fname_imag + '.fits'))

    # Plot deformable mirror WFE and save to disk
    if saveopds:
        opd_name = f'opd_actuator_{mode_no}'
        plt.clf()
        telescope.display_opd()
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    return efield_focal_plane