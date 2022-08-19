import os
import time
import functools
import logging
import hcipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
from pastis.simulators.scda_telescopes import HexRingAPLC
import pastis.simulators.webbpsf_imaging as webbpsf_imaging
from pastis.matrix_generation.matrix_building_numerical import PastisMatrix
import pastis.plotting as ppl
import pastis.util as util

log = logging.getLogger()
matplotlib.rc('image', origin='lower')
matplotlib.rc('pdf', fonttype=42)


class PastisMatrixEfields(PastisMatrix):
    instrument = None
    """ Main class for PASTIS matrix calculations from individually 'poked' modes. """

    def __init__(self, nb_seg, seglist, calc_science, calc_wfs,
                 initial_path='', saveefields=True, saveopds=True, norm_one_photon=True):
        """
        Parameters:
        ----------
        nb_seg : int
            Number of segments in the segmented aperture.
        seglist : list or array
            List of all segment indices, as given in the indexed aperture file.
        calc_science : bool
            Whether to calculate the Efields in the science focal plane.
        calc_wfs : bool
            Whether to calculate the Efields in the out-of-band Zernike WFS plane.
        initial_path: string
            Path to top-level directory where result folder should be saved to.
        saveefields: bool
            Whether to save E-fields both at focal and wfs plane as fits file to disk or not
        saveopds : bool
            Whether to save images of pair-wise aberrated pupils to disk or not
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.
        """
        super().__init__(nb_seg=nb_seg, seglist=seglist, save_path=initial_path)
        self.calc_science = calc_science
        self.calc_wfs = calc_wfs

        self.save_efields = saveefields
        self.saveopds = saveopds
        self.calculate_one_mode = None
        self.efields_per_mode = []
        self.efields_per_mode_wfs = []
        self.norm_one_photon = norm_one_photon

        os.makedirs(os.path.join(self.resDir, 'efields'), exist_ok=True)
        os.makedirs(os.path.join(self.resDir, 'efields_wfs'), exist_ok=True)

    def calc(self):
        """ Main method that calculates the PASTIS matrix """

        start_time = time.time()

        self.calculate_ref_efield()
        self.calculate_ref_efield_wfs()
        self.setup_deformable_mirror()
        self.setup_single_mode_function()
        self.calculate_efields()
        if self.calc_science:
            self.calculate_pastis_matrix_from_efields()

        end_time = time.time()
        log.info(
            f'Runtime for {self.__class__.__name__}.calc(): {end_time - start_time}sec = {(end_time - start_time) / 60}min')
        log.info(f'Data saved to {self.resDir}')

    def calculate_efields(self):
        """ Poke each mode individually and calculate the resulting focal plane E-field. """

        for i in range(self.number_all_modes):
            efields = self.calculate_one_mode(i)
            if self.calc_science:
                self.efields_per_mode.append(efields['efield_science_plane'])
            if self.calc_wfs:
                self.efields_per_mode_wfs.append(efields['efield_wfs_plane'])
        self.efields_per_mode = np.array(self.efields_per_mode)
        self.efields_per_mode_wfs = np.array(self.efields_per_mode_wfs)

    def calculate_pastis_matrix_from_efields(self):
        """ Use the individual-mode E-fields to calculate the PASTIS matrix from it. """

        self.matrix_pastis = pastis_matrix_from_efields(self.efields_per_mode, self.efield_ref, self.norm, self.dh_mask, self.wfe_aber)

        # Save matrix to file
        filename_matrix = f'pastis_matrix'
        hcipy.write_fits(self.matrix_pastis, os.path.join(self.resDir, filename_matrix + '.fits'))
        ppl.plot_pastis_matrix(self.matrix_pastis, self.wvln * 1e9, out_dir=self.resDir, save=True)  # convert wavelength to nm
        log.info(f'PASTIS matrix saved to: {os.path.join(self.resDir, filename_matrix + ".fits")}')

    def calculate_ref_efield(self):
        """ Create the attributes self.norm, self.dh_mask, self.coro_simulator and self.efield_ref. """
        raise NotImplementedError()

    def calculate_ref_efield_wfs(self):
        """ Create the attributes self.norm, self.dh_mask, self.wfs_simulator and self.efield_ref_wfs. """
        raise NotImplementedError()

    def setup_deformable_mirror(self):
        """ Set up the deformable mirror for the modes you're using, if necessary, and define the total number of mode actuators. """
        raise NotImplementedError()

    def setup_single_mode_function(self):
        """ Create an attribute that is the partial function that can calculate the focal plane E-field from one
        aberrated mode. This needs to create self.calculate_one_mode. """
        raise NotImplementedError()


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


class MatrixEfieldInternalSimulator(PastisMatrixEfields):
    """ Calculate a PASTIS matrix for one of the package-internal simulators, using E-fields. """
    def __init__(self, which_dm, dm_spec, nb_seg, seglist, calc_science, calc_wfs,
                 initial_path='', saveefields=True, saveopds=True, norm_one_photon=True):
        """
        :param which_dm: string, which DM to calculate the matrix for - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
        :param dm_spec: tuple or int, specification for the used DM -
                        for seg_mirror: int, number of local Zernike modes on each segment
                        for harris_seg_mirror: tuple (string, array, bool, bool, bool), absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets
                        for zernike_mirror: int, number of global Zernikes
        :param nb_seg: int, Number of segments in the segmented aperture.
        :param seglist: list or array, list of all segment indices, as given in the indexed aperture file.
        :param calc_science: bool, whether to calculate the Efields in the science focal plane.
        :param calc_wfs: bool, whether to calculate the Efields in the out-of-band Zernike WFS plane.
        :param initial_path: string, path to top-level directory where result folder should be saved to.
        :param saveefields: bool, whether to save E-fields as fits file to disk or not
        :param saveopds: bool, whether to save images of pair-wise aberrated pupils to disk or not
        :param norm_one_photon: bool, whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.
        """
        super().__init__(nb_seg=nb_seg, seglist=seglist, calc_science=calc_science, calc_wfs=calc_wfs,
                         initial_path=initial_path, saveefields=saveefields, saveopds=saveopds, norm_one_photon=norm_one_photon)
        self.which_dm = which_dm
        self.dm_spec = dm_spec

        self.instantiate_simulator()

    def instantiate_simulator(self):
        """ Create a simulator object and save to self.simulator """
        raise NotImplementedError()

    def calculate_ref_efield(self):
        """Calculate the reference E-field, DH mask, and direct PSF norm factor."""
        self.dh_mask = self.simulator.dh_mask

        # Calculate contrast normalization factor from direct PSF (intensity)
        unaberrated_coro_psf, direct = self.simulator.calc_psf(ref=True, norm_one_photon=self.norm_one_photon)
        self.norm = np.max(direct)
        hcipy.write_fits(unaberrated_coro_psf/self.norm, os.path.join(self.overall_dir, 'unaberrated_coro_psf.fits'))

        npx = unaberrated_coro_psf.shaped.shape[0]
        im_lamd = npx/2 / self.simulator.sampling
        plt.figure(figsize=(10, 10))
        plt.imshow(np.log10(unaberrated_coro_psf.shaped/self.norm), cmap='inferno', extent=[-im_lamd, im_lamd, -im_lamd, im_lamd])
        plt.xlabel('$\lambda/D$', size=30)
        plt.ylabel('$\lambda/D$', size=30)
        plt.tick_params(axis='both', length=6, width=2, labelsize=30)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('log contrast', fontsize=30, weight='bold', rotation=270, labelpad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.overall_dir, 'unaberrated_coro_psf.pdf'))

        # Calculate reference E-field in focal plane, without any aberrations applied
        unaberrated_ref_efield, _inter = self.simulator.calc_psf(return_intermediate='efield', norm_one_photon=self.norm_one_photon)
        self.efield_ref = unaberrated_ref_efield.electric_field
        print("npx defined in function:", npx)
    def calculate_ref_efield_wfs(self):
        """Calculate the reference E-field at the wavefront sensor plane."""
        unaberrated_ref_efield_wfs = self.simulator.calc_out_of_band_wfs(norm_one_photon=self.norm_one_photon)
        self.efield_ref_wfs = unaberrated_ref_efield_wfs

    def setup_deformable_mirror(self):
        """ Set up the deformable mirror for the modes you're using and define the total number of mode actuators. """

        log.info('Setting up deformable mirror...')
        if self.which_dm == 'seg_mirror':
            n_modes_segs = self.dm_spec
            log.info(f'Creating segmented mirror with {n_modes_segs} local modes on each segment...')
            self.simulator.create_segmented_mirror(n_modes_segs)
            self.number_all_modes = self.simulator.sm.num_actuators

        elif self.which_dm == 'harris_seg_mirror':
            fpath, pad_orientations, therm, mech, other = self.dm_spec
            log.info(f'Reading Harris spreadsheet from {fpath}')
            log.info(f'Using pad orientations: {pad_orientations}')
            self.simulator.create_segmented_harris_mirror(fpath, pad_orientations, therm, mech, other)
            self.number_all_modes = self.simulator.harris_sm.num_actuators

        elif self.which_dm == 'zernike_mirror':
            n_modes_zernikes = self.dm_spec
            log.info(f'Creating global Zernike mirror with {n_modes_zernikes} global modes...')
            self.simulator.create_global_zernike_mirror(n_modes_zernikes)
            self.number_all_modes = self.simulator.zernike_mirror.num_actuators

        else:
            raise ValueError(f'DM with name "{self.which_dm}" not recognized.')

        log.info(f'Total number of modes: {self.number_all_modes}')

    def setup_single_mode_function(self):
        """ Create the partial function that returns the E-field of a single aberrated mode. """

        self.calculate_one_mode = functools.partial(_simulator_matrix_single_mode, self.which_dm, self.number_all_modes,
                                                    self.wfe_aber, self.simulator, self.calc_science, self.calc_wfs,
                                                    self.norm_one_photon, self.resDir, self.save_efields, self.saveopds)


class MatrixEfieldLuvoirA(MatrixEfieldInternalSimulator):
    """ Calculate a PASTIS matrix for LUVOIR-A, using E-fields. """
    instrument = 'LUVOIR'

    def __init__(self, which_dm, dm_spec, design='small', calc_science=True, calc_wfs=False,
                 initial_path='', saveefields=True, saveopds=True, norm_one_photon=True):
        """
        :param which_dm: string, which DM to calculate the matrix for - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
        :param dm_spec: tuple or int, specification for the used DM -
                        for seg_mirror: int, number of local Zernike modes on each segment
                        for harris_seg_mirror: tuple (string, array, bool, bool, bool), absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets
                        for zernike_mirror: int, number of global Zernikes
        :param design: str, what coronagraph design to use - 'small', 'medium' or 'large'
        :param calc_science: bool, whether to calculate the Efields in the science focal plane.
        :param calc_wfs: bool, whether to calculate the Efields in the out-of-band Zernike WFS plane.
        :param initial_path: string, path to top-level directory where result folder should be saved to.
        :param saveefields: bool, whether to save E-fields as fits file to disk or not
        :param saveopds: bool, whether to save images of pair-wise aberrated pupils to disk or not
        :param norm_one_photon: bool, whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.
        """
        nb_seg = CONFIG_PASTIS.getint(self.instrument, 'nb_subapertures')
        seglist = util.get_segment_list(self.instrument)
        self.design = design
        super().__init__(which_dm=which_dm, dm_spec=dm_spec, nb_seg=nb_seg, seglist=seglist, calc_science=calc_science, calc_wfs=calc_wfs,
                         initial_path=initial_path, saveefields=saveefields, saveopds=saveopds, norm_one_photon=norm_one_photon)

    def instantiate_simulator(self):
        optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
        sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
        self.simulator = LuvoirA_APLC(optics_input, self.design, sampling)


class MatrixEfieldHex(MatrixEfieldInternalSimulator):
    """ Calculate a PASTIS matrix for a SCDA Hex aperture with 1-5 segment rings, using E-fields. """
    instrument = 'HexRingTelescope'

    def __init__(self, which_dm, dm_spec, num_rings=1, calc_science=True, calc_wfs=False,
                 initial_path='', saveefields=True, saveopds=True, norm_one_photon=True):
        """
        :param which_dm: string, which DM to calculate the matrix for - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
        :param dm_spec: tuple or int, specification for the used DM -
                        for seg_mirror: int, number of local Zernike modes on each segment
                        for harris_seg_mirror: tuple (string, array, bool, bool, bool), absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets
                        for zernike_mirror: int, number of global Zernikes
        :param num_rings: int, number of hexagonal segment rings
        :param calc_science: bool, whether to calculate the Efields in the science focal plane.
        :param calc_wfs: bool, whether to calculate the Efields in the out-of-band Zernike WFS plane.
        :param initial_path: string, path to top-level directory where result folder should be saved to.
        :param saveefields: bool, whether to save E-fields as fits file to disk or not
        :param saveopds: bool, whether to save images of pair-wise aberrated pupils to disk or not
        :param norm_one_photon: bool, whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.
        """
        nb_seg = 3 * num_rings * (num_rings + 1) + 1
        seglist = np.arange(nb_seg) + 1
        self.num_rings = num_rings
        super().__init__(which_dm=which_dm, dm_spec=dm_spec, nb_seg=nb_seg, seglist=seglist, calc_science=calc_science, calc_wfs=calc_wfs,
                         initial_path=initial_path, saveefields=saveefields, saveopds=saveopds, norm_one_photon=norm_one_photon)

    def instantiate_simulator(self):
        optics_input = os.path.join(util.find_repo_location(), 'data', 'SCDA')
        sampling = CONFIG_PASTIS.getfloat('HexRingTelescope', 'sampling')
        self.simulator = HexRingAPLC(optics_input, self.num_rings, sampling)


class MatrixEfieldRST(PastisMatrixEfields):
    """
    Class to calculate the PASTIS matrix from E-fields of RST CGI.
    """
    instrument = 'RST'

    def __init__(self, initial_path='', saveefields=True, saveopds=True):
        nb_seg = CONFIG_PASTIS.getint(self.instrument, 'nb_subapertures')
        seglist = util.get_segment_list(self.instrument)
        super().__init__(nb_seg=nb_seg, seglist=seglist, calc_science=True, calc_wfs=False,
                         initial_path=initial_path, saveefields=saveefields, saveopds=saveopds)

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


def _simulator_matrix_single_mode(which_dm, number_all_modes, wfe_aber, simulator, calc_science, calc_wfs,
                                  norm_one_photon, resDir, saveefields, saveopds, mode_no):
    """
    Calculate the mean E-field of one aberrated mode on one of the internal simulator instances; for PastisMatrixEfields().
    :param which_dm: string, which DM - "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    :param number_all_modes: int, total number of all modes
    :param wfe_aber: float, calibration aberration in meters
    :param simulator: instance of segmented telescope simulator
    :param calc_science: bool, whether to calculate the Efields in the science focal plane.
    :param calc_wfs: bool, whether to calculate the Efields in the out-of-band Zernike WFS plane.
    :param norm_one_photon: bool, whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.
    :param resDir: str, directory for matrix calculation results
    :param saveefields: bool, Whether to save E-fields as fits file to disk or not
    :param saveopds: bool, Whether to save images of pair-wise aberrated pupils to disk or not
    :param mode_no: int, which mode index to calculate the E-field for
    :return: dict, resulting E-fields
    """

    log.info(f'MODE NUMBER: {mode_no}')
    efield_focal_plane = None
    efield_wfs_plane = None

    # Apply calibration aberration to used mode
    all_modes = np.zeros(number_all_modes)
    all_modes[mode_no] = wfe_aber / 2    # simulator takes aberrations in surface  #TODO: check that this is true for all the DMs
    if which_dm == 'seg_mirror':
        simulator.sm.actuators = all_modes
    elif which_dm == 'harris_seg_mirror':
        simulator.harris_sm.actuators = all_modes
    elif which_dm == 'zernike_mirror':
        simulator.zernike_mirror.actuators = all_modes
    else:
        raise ValueError(f'DM with name "{which_dm}" not recognized.')

    # Calculate coronagraphic E-field
    efield_focal_plane, inter = simulator.calc_psf(return_intermediate='efield', norm_one_photon=norm_one_photon)

    # Calculate WFS plane E-field
    if calc_wfs:
        # Purposefully do not use `simulator.calc_out_of_band_wfs()` because it would recalculate all intermediate planes,
        # as is already done with `simulator.calc_psf()`, so we can use the output from there.
        if simulator.zwfs is None:
            simulator.create_zernike_wfs()
        efield_wfs_plane = simulator.zwfs(inter['active_pupil'])

    if saveefields:
        # Save focal plane Efields
        if calc_science:
            fname_real_focal = f'focal_real_mode{mode_no}'
            hcipy.write_fits(efield_focal_plane.real, os.path.join(resDir, 'efields', fname_real_focal + '.fits'))
            fname_imag_focal = f'focal_imag_mode{mode_no}'
            hcipy.write_fits(efield_focal_plane.imag, os.path.join(resDir, 'efields', fname_imag_focal + '.fits'))

        # Save wfs plane Efields
        if calc_wfs:
            fname_real_wfs = f'wfs_real_mode{mode_no}'
            hcipy.write_fits(efield_wfs_plane.real, os.path.join(resDir, 'efields_wfs', fname_real_wfs + '.fits'))
            fname_imag_wfs = f'wfs_imag_mode{mode_no}'
            hcipy.write_fits(efield_wfs_plane.imag, os.path.join(resDir, 'efields_wfs', fname_imag_wfs + '.fits'))

    if saveopds:
        opd_map = inter[which_dm].phase
        opd_name = f'opd_mode_{mode_no}'
        plt.clf()
        hcipy.imshow_field(opd_map, grid=simulator.aperture.grid, mask=simulator.aperture, cmap='RdBu')
        plt.savefig(os.path.join(resDir, 'OTE_images', opd_name + '.pdf'))

    # Format returned Efields
    efields = {'efield_science_plane': efield_focal_plane.electric_field,
               'efield_wfs_plane': efield_wfs_plane}

    return efields


def _rst_matrix_single_mode(wfe_aber, rst_sim, resDir, saveefields, saveopds, mode_no):
    """
    Function to calculate RST Electrical field (E_field) of one DM actuator in CGI.
    :param wfe_aber: float, calibration aberration per actuator in m
    :param rst_sim: instance of CGI simulator
    :param resDir: str, directory for matrix calculations
    :param saveefields: bool, if True, all E_field will be saved to disk individually, as fits files
    :param savepods: bool, if True, all pupil surface maps of aberrated actuators pairs will be saved to disk as PDF
    :param mode_no: int, which aberrated actuator to calculate the E-field for
    :return: dict, resulting E-fields
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

    # Format returned Efields
    efields = {'efield_science_plane': efield_focal_plane.wavefront}

    return efields
