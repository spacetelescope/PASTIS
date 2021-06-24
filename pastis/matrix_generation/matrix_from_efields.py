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

log = logging.getLogger()
matplotlib.rc('image', origin='lower')
matplotlib.rc('pdf', fonttype=42)


class PastisMatrixEfields(PastisMatrix):

    def __init__(self, design=None, initial_path='', saveefields=True, saveopds=True):
        super().__init__(design=design, initial_path=initial_path)

        self.instrument = CONFIG_PASTIS.get('telescope', 'name')

        self.save_efields = saveefields
        self.saveopds = saveopds
        self.calculate_one_mode = None
        self.efields_per_mode = []

        os.makedirs(os.path.join(self.resDir, 'efields'), exist_ok=True)

    def calc(self):
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
        for i in range(self.number_all_modes):
            self.efields_per_mode.append(self.calculate_one_mode(i))
        self.efields_per_mode = np.array(self.efields_per_mode)

    def calculate_pastis_matrix_from_efields(self):
        self.matrix_pastis = pastis_matrix_from_efields(self.instrument, self.efields_per_mode, self.efield_ref, self.norm, self.dh_mask, self.wfe_aber)

        # Save matrix to file
        filename_matrix = f'pastis_matrix'
        hcipy.write_fits(self.matrix_pastis, os.path.join(self.resDir, filename_matrix + '.fits'))
        ppl.plot_pastis_matrix(self.matrix_pastis, self.wvln * 1e9, out_dir=self.resDir, save=True)  # convert wavelength to nm
        log.info(f'PASTIS matrix saved to: {os.path.join(self.resDir, filename_matrix + ".fits")}')

    @abstractmethod
    def calculate_ref_efield(self):
        pass

    @abstractmethod
    def setup_deformable_mirror(self):
        pass

    @abstractmethod
    def setup_single_mode_function(self):
        pass


def pastis_matrix_from_efields(instrument, electric_fields, efield_ref, direct_norm, dh_mask, wfe_aber):

    # Calculate the semi-analytical PASTIS matrix from the individual E-fields
    matrix_pastis_half = calculate_semi_analytic_pastis_from_efields(instrument, electric_fields, efield_ref, direct_norm, dh_mask)

    # Symmetrize the half-PASTIS matrix
    log.info('Symmetrizing PASTIS matrix')
    matrix_pastis = util.symmetrize(matrix_pastis_half)

    # Normalize PASTIS matrix by input aberration
    matrix_pastis /= np.square(wfe_aber * 1e9)

    return matrix_pastis


def calculate_semi_analytic_pastis_from_efields(instrument, efields, efield_ref, direct_norm, dh_mask):

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

    def __init__(self, design='small', max_local_zernike=3, initial_path='', saveefields=True, saveopds=True):
        super().__init__(design=design, initial_path=initial_path, saveefields=saveefields, saveopds=saveopds)
        self.max_local_zernike = max_local_zernike

    def calculate_ref_efield(self):
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
        log.info(f'Creating segmented mirror with {self.max_local_zernike} local modes each...')
        self.luvoir.create_segmented_mirror(self.max_local_zernike)
        self.number_all_modes = self.luvoir.sm.num_actuators
        log.info(f'Total number of modes: {self.number_all_modes}')

    def setup_single_mode_function(self):
        self.calculate_one_mode = functools.partial(_luvoir_matrix_single_mode, self.number_all_modes, self.wfe_aber,
                                                    self.luvoir, self.resDir, self.save_efields, self.saveopds)


class MatrixEfieldRST(PastisMatrixEfields):
    """
    Class to calculate RST Electrical field of RST CGI.
    """
    instrument = 'RST'

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
        rst_direct = self.rst_cgi.raw_PSF()
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


def _luvoir_matrix_single_mode(number_all_modes, wfe_aber, luvoir_sim, resDir, saveefields, saveopds, mode_no):

    log.info(f'MODE NUMBER: {mode_no}')

    # Apply calibration aberration to used mode
    all_modes = np.zeros(number_all_modes)
    all_modes[mode_no] = wfe_aber / 2
    luvoir_sim.sm.actuators = all_modes

    # Calculate coronagraphic E-field
    efield_focal_plane, inter = luvoir_sim.calc_psf(return_intermediate='efield')

    if saveefields:
        fname_real = f'efield_real_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.real, os.path.join(resDir, 'efields', fname_real + '.fits'))
        fname_imag = f'efield_imag_mode{mode_no}'
        hcipy.write_fits(efield_focal_plane.imag, os.path.join(resDir, 'efields', fname_imag + '.fits'))

    if saveopds:
        opd_name = f'opd_mode_{mode_no}'
        plt.clf()
        hcipy.imshow_field(inter['seg_mirror'].phase, grid=luvoir_sim.aperture.grid, mask=luvoir_sim.aperture, cmap='RdBu')
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
        plt.figure(figsize=(8, 8))
        rst_sim.dm1.display(what='opd', opd_vmax=wfe_aber, colorbar_orientation='horizontal',
                            title='Aberrated actuator pair')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    return efield_focal_plane.wavefront
