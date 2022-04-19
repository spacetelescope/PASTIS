"""
This module contains class and all necessary requierement to work RST
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import hcipy
import re

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
                self.unaberrated = self.imaging_psf()

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
                self.number_all_modes = CONFIG_PASTIS.getint(self.instrument, 'nb_subapertures')
                self.seglist = util.get_segment_list(self.instrument)
                self.wvln = CONFIG_PASTIS.getfloat(self.instrument, 'lambda') * 1e-9  # m
                self.wfe_aber = CONFIG_PASTIS.getfloat(self.instrument, 'calibration_aberration') * 1e-9  # m
                self.sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
                self.optics_input = os.path.join(util.find_repo_location(),
                                            CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))

                self.dm_modes_list = [0]
                self.dm_modes_max = 1

                self.sim = LuvoirAPLC(self.optics_input, self.design, self.sampling)
                self.parameters = pastis.launchers.parameters.parameters()
                self.parameters.def_saves()
                self.parameters.def_analysis()

                if self.design is None and self.parameters.hockeystick_curve :
                        raise ValueError(
                                'Need to specify apodizer_choice when woant plot hockeystick curve with LUVOIR instrument.')

        def normalization_and_dark_hole(self):
                # Calculate direct reference images for contrast normalization
                self.flatten()
                self.dh_mask = self.sim.dh_mask.shaped

                # Calculate contrast normalization factor from direct PSF (intensity)
                psf, direct = self.sim.calc_psf(ref=True)
                self.norm = direct.max()
                self.direct_psf = psf.shaped

        def calculate_unaberrated_contrast(self):
                '''Calcul an underrated contrast, usually needs one execution of normalization_and_dark_hole'''
                self.flatten()
                self.sim.calc_psf(ref=True)
                self.unaberrated = self.imaging_psf()
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
                ''' Flat all modes'''
                self.sim.flatten()
                return []

        def push_mode(self, mode, amplitude):
                mode_type = (mode % self.dm_modes_max)+1
                if mode_type in self.dm_modes_list or 0 in self.dm_modes_list:
                        if self.which_dm == 'default':
                                if mode_type ==  1 :
                                        self.dm_mode.set_segment(mode%self.dm_modes_max+1, amplitude /2, 0, 0)
                                elif mode_type == 2 :
                                        self.dm_mode.set_segment(mode%self.dm_modes_max+1, 0, amplitude /2, 0)
                                elif mode_type == 3 :
                                        self.dm_mode.set_segment(mode%self.dm_modes_max+1, 0, 0, amplitude /2)
                        else:
                                all_modes = np.zeros(self.number_all_modes)
                                all_modes[mode] = amplitude / 2
                                self.dm_mode.actuators = all_modes

        def imaging_psf(self, inst=None):

                if inst == None :
                        inst = self.sim
                if self.parameters.saveopds:
                        psf, inter = inst.calc_psf(ref=False, display_intermediate=False,
                                                    return_intermediate='intensity')
                        self.inter = inter['seg_mirror']
                else:
                        psf = inst.calc_psf(ref=False, display_intermediate=False,
                                                           return_intermediate=None)
                self.psf = (psf.shaped/self.norm)
                return self.psf

        def imaging_efield(self):
                efield_focal_plane , inter = self.sim.calc_psf(return_intermediate='efield')
                self.efield = efield_focal_plane.electric_field.shaped
                self.inter = inter['seg_mirror'].phase
                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def setup_deformable_mirror(self):
                """ Set up the deformable mirror for the modes you're using and define the total number of mode actuators. """

                #DM config
                self.which_dm = CONFIG_PASTIS.get('LUVOIR', 'DM')
                self.dm_modes_max = CONFIG_PASTIS.getint('LUVOIR', 'DM_mode')

                dm_modes_list = CONFIG_PASTIS.get('LUVOIR', 'DM_mode_select')
                dm_modes_list = map(int, re.findall(r'\d+', dm_modes_list)) #Find integer
                self.dm_modes_list = list(set(dm_modes_list)) #return unique items

                if np.max(self.dm_modes_list) > self.dm_modes_max :
                        error_msg = "one or more DM_mode_select in LUVOIR section inside config file is/are higher than DM_mode!"
                        log.error(error_msg)
                        raise ValueError(error_msg)

                log.info('Setting up deformable mirror...')
                if self.which_dm == 'seg_mirror':
                        log.info(f'Creating segmented mirror with {self.dm_modes_max} local modes on each segment...')
                        if np.max(self.dm_modes_max) > 3:
                                error_msg = f"DM_mode={self.dm_modes_max} in LUVOIR section inside config file is higher than 3!"
                                log.error(error_msg)
                                raise ValueError(error_msg)
                        self.sim.create_segmented_mirror(self.dm_modes_max)
                        self.number_all_modes = self.sim.sm.num_actuators
                        self.dm_mode = self.sim.sm
                elif self.which_dm == 'harris_seg_mirror': #TODO Test it
                        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
                        pad_orientations = np.pi / 2 * np.ones(120)
                        therm = CONFIG_PASTIS.getboolean('LUVOIR', 'therm')
                        mech = CONFIG_PASTIS.getboolean('LUVOIR', 'mech')
                        other = CONFIG_PASTIS.getboolean('LUVOIR', 'other')
                        log.info(f'Reading Harris spreadsheet from {fpath}')
                        log.info(f'Using pad orientations: {pad_orientations}')
                        self.sim.create_segmented_harris_mirror(fpath, pad_orientations, therm, mech, other)
                        self.number_all_modes = self.sim.harris_sm.num_actuators
                        self.dm_mode = self.sim.harris_sm
                        self.dm_modes_max = 1
                elif self.which_dm == 'zernike_mirror':
                        log.info(f'Creating global Zernike mirror with {self.dm_modes_max} global modes...')
                        self.sim.create_global_zernike_mirror(self.dm_modes_max)
                        self.number_all_modes = self.sim.zernike_mirror.num_actuators
                        self.dm_mode = self.sim.zernike_mirror
                else:
                        self.dm_mode = self.sim
                        if self.which_dm != 'default':
                                log.info(f'DM with name "{self.which_dm}" not recognized.')
                                self.which_dm = 'default'
                        log.info('Default mirror is on')
                        if np.max(self.dm_modes_max) > 3:
                                error_msg = f"DM_mode={self.dm_modes_max} in LUVOIR section inside config file is higher than 3!"
                                log.error(error_msg)
                                raise ValueError(error_msg)
                        self.number_all_modes *= self.dm_modes_max

                log.info(f'Total number of modes: {self.number_all_modes}')

        def display_opd(self):
                hcipy.imshow_field(self.inter, grid=self.sim.aperture.grid, mask=self.sim.aperture, cmap='RdBu')

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
                self.direct_psf
                self.norm

                self.dh_mask

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
                pass

        def imaging_psf(self, inst=None):

                return self.psf

        def imaging_efield(self):

                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def setup_deformable_mirror(self):
                pass

        def display_opd(self):
                ""
                pass


