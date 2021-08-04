"""
This module contains class and all necessary requierement to work RST
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import hcipy

from pastis.config import CONFIG_PASTIS
import pastis.util as util
import pastis.launchers.parameters
import pastis.e2e_simulators.webbpsf_imaging as webbpsf_imaging
from pastis.e2e_simulators.luvoir_imaging import LuvoirAPLC

log = logging.getLogger()

class RST():

        def __init__(self, initial_path=''):
                self.sim = webbpsf_imaging.set_up_cgi()
                self.nb_actu = self.sim.nbactuator
                self.norm = 1

                self.wfe_aber = CONFIG_PASTIS.getfloat('RST', 'calibration_aberration') * 1e-9   # m
                self.number_all_modes = CONFIG_PASTIS.getint('RST', 'nb_subapertures')

        def normalization_and_dark_hole(self):
                # Calculate direct reference images for contrast normalization
                self.flatten()
                rst_direct = self.sim.raw_coronagraph()
                self.direct_psf = self.imaging_psf(inst=rst_direct)
                self.norm = self.direct_psf.max()

                self.iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
                self.owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
                self.sim.working_area(im=self.direct_psf, inner_rad=self.iwa, outer_rad=self.owa)
                self.dh_mask = self.sim.WA

        def calculate_unaberrated_contrast(self):
                '''Calcul an underrated contrast, usually needs one execution of normalization_and_dark_hole'''
                self.flatten()
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
                ''' Flat all actuators'''
                self.sim.dm1.flatten()
                return []

        def push_mode(self, mode, amplitude):
                actu_x, actu_y = util.seg_to_dm_xy(self.nb_actu, mode)
                self.sim.dm1.set_actuator(actu_x, actu_y, amplitude)

        def imaging_psf(self, inst=None):
                if inst == None :
                        inst = self.sim
                fit_psf = inst.calc_psf(nlambda=1, fov_arcsec=1.6)
                self.psf = fit_psf[0].data/self.norm
                return self.psf

        def imaging_efield(self):
                _psf , inter = self.sim.calc_psf(nlambda=1, fov_arcsec=1.6, return_intermediates=True)
                self.efield = inter[-1].wavefront
                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def setup_deformable_mirror(self):
                pass

        def display_opd(self):
                plt.figure(figsize=(self.nb_actu, self.nb_actu))
                self.sim.dm1.display(what='opd', opd_vmax=self.wfe_aber, colorbar_orientation='horizontal',
                     title='Aberrated actuator pair')



class LUVOIRA():

        instrument = 'LUVOIR'


        def __init__(self, initial_path=''):
                # General telescope parameters
                self.design = CONFIG_PASTIS.get('LUVOIR', 'design')
                self.nb_seg = CONFIG_PASTIS.getint(self.instrument, 'nb_subapertures')
                self.seglist = util.get_segment_list(self.instrument)
                self.wvln = CONFIG_PASTIS.getfloat(self.instrument, 'lambda') * 1e-9  # m
                self.wfe_aber = CONFIG_PASTIS.getfloat(self.instrument, 'calibration_aberration') * 1e-9  # m
                self.sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
                self.optics_input = os.path.join(util.find_repo_location(),
                                            CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))

                self.sim = LuvoirAPLC(self.optics_input, self.design, self.sampling)
                self.parameters = pastis.launchers.parameters.parameters()
                self.parameters.def_saves()

        def normalization_and_dark_hole(self):
                # Calculate direct reference images for contrast normalization
                self.flatten()
                self.dh_mask = self.sim.dh_mask.shaped

                # Calculate contrast normalization factor from direct PSF (intensity)
                _unaberrated_coro_psf, psf = self.sim.calc_psf(ref=True)
                self.direct_psf = psf.shaped
                self.norm = self.direct_psf.max()

        def calculate_unaberrated_contrast(self):
                '''Calcul an underrated contrast, usually needs one execution of normalization_and_dark_hole'''
                self.flatten()
                self.sim.calc_psf(ref=True)
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
                ''' Flat all modes'''
                self.sim.flatten()
                return []

        def push_mode(self, mode, amplitude):
                self.seg.set_segment(mode+1, amplitude / 2, 0, 0)

        def imaging_psf(self, inst=None):

                if inst == None :
                        inst = self.sim
                if self.parameters.saveopds:
                        psf, self.inter = inst.calc_psf(ref=False, display_intermediate=False,
                                                    return_intermediate='intensity')
                else:
                        psf = inst.calc_psf(ref=False, display_intermediate=False,
                                                           return_intermediate=None)
                self.psf = (psf/self.norm).shaped
                return self.psf

        def imaging_efield(self):
                efield_focal_plane , self.inter = self.sim.calc_psf(return_intermediate='efield')
                self.efield = efield_focal_plane.electric_field
                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def setup_deformable_mirror(self):
                """ Set up the deformable mirror for the modes you're using and define the total number of mode actuators. """

                #DM config
                self.which_dm = CONFIG_PASTIS.get('LUVOIR', 'DM')

                log.info('Setting up deformable mirror...')
                if self.which_dm == 'seg_mirror':
                        log.info(f'Creating segmented mirror with {self.nb_seg} local modes on each segment...')
                        self.sim.create_segmented_mirror(self.nb_seg)
                        self.seg = self.sim.sm.set_segment()
                elif self.which_dm == 'harris_seg_mirror':
                        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
                        pad_orientations = np.pi / 2 * np.ones(120)
                        therm = False
                        mech = True
                        other = False
                        log.info(f'Reading Harris spreadsheet from {fpath}')
                        log.info(f'Using pad orientations: {pad_orientations}')
                        self.sim.create_segmented_harris_mirror(fpath, pad_orientations, therm, mech, other)
                        self.number_all_modes = self.sim.harris_sm.num_actuators
                        self.seg = self.sim.harris_sm.set_segment()
                elif self.which_dm == 'zernike_mirror':
                        n_modes_zernikes = CONFIG_PASTIS.getin('zernikes','max_zern')
                        log.info(f'Creating global Zernike mirror with {n_modes_zernikes} global modes...')
                        self.sim.create_global_zernike_mirror(n_modes_zernikes)
                        self.number_all_modes = self.sim.zernike_mirror.num_actuators
                        self.seg = self.sim.zernike_mirror.set_segment()
                else:
                        raise ValueError(f'DM with name "{self.which_dm}" not recognized.')

                log.info(f'Total number of modes: {self.number_all_modes}')

        def display_opd(self):
                hcipy.imshow_field(self.inter['seg_mirror'], grid=self.sim.aperture.grid, mask=self.sim.aperture, cmap='RdBu')

class MODEL():
        '''This is minial template for a new telescope implementation all parameter inside are necessary replace every #message#'''

        def __init__(self, initial_path=''):
                self.sim #= MODEL simulation start#
                self.norm = 1

                self.wfe_aber = CONFIG_PASTIS.getfloat('#MODEL#', 'calibration_aberration') * 1e-9   # m
                self.number_all_modes = CONFIG_PASTIS.getint('#MODEL#', 'nb_subapertures')

        def normalization_and_dark_hole(self):
                '''Calcul an underrated contrast, usually needs one execution of normalization_and_dark_hole'''

                # Calculate direct reference images for contrast normalization
                self.flatten()
                rst_direct = self.sim.raw_coronagraph()
                self.direct_psf = self.imaging_psf(inst=rst_direct)
                self.norm = self.direct_psf.max()

                self.iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
                self.owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
                self.sim.working_area(im=self.direct_psf, inner_rad=self.iwa, outer_rad=self.owa)
                self.dh_mask = self.sim.WA

        def calculate_unaberrated_contrast(self):
                self.flatten()
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
                ''' Flat all modes'''
                ##
                return []

        def push_mode(self, mode, amplitude):
                actu_x, actu_y = util.seg_to_dm_xy(self.nb_actu, mode)
                self.sim.dm1.set_actuator(actu_x, actu_y, amplitude)

        def imaging_psf(self, inst=None):
                if inst == None :
                        inst = self.sim
                fit_psf = inst.calc_psf(nlambda=1, fov_arcsec=1.6)
                self.psf = fit_psf[0].data/self.norm
                return self.psf

        def imaging_efield(self):
                _psf , inter = self.sim.calc_psf(nlambda=1, fov_arcsec=1.6, return_intermediates=True)
                self.efield = inter[-1].wavefront
                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def setup_deformable_mirror(self):
                pass

        def display_opd(self):
                line = int(np.sqrt(self.number_all_modes))
                plt.figure(figsize=(self.nb_actu, self.nb_actu))
                self.sim.dm1.display(what='opd', opd_vmax=self.wfe_aber, colorbar_orientation='horizontal',
                     title='Aberrated actuator pair')

