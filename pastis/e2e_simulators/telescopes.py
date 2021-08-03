"""
This module contains class and all necessary requierement to work RST
"""

import matplotlib.pyplot as plt
import hcipy

from pastis.config import CONFIG_PASTIS
import pastis.util as util
import pastis.e2e_simulators.webbpsf_imaging as webbpsf_imaging


class RST():

        def __init__(self):
                self.sim = webbpsf_imaging.set_up_cgi()
                self.number_all_modes = CONFIG_PASTIS.getint('RST', 'nb_subapertures')
                self.nb_actu = self.sim.nbactuator
                self.norm = 1

                self.wfe_aber = CONFIG_PASTIS.getfloat('RST', 'calibration_aberration') * 1e-9   # m

        def normalization_and_dark_hole(self):
                # Calculate direct reference images for contrast normalization
                rst_direct = self.sim.raw_coronagraph()
                self.direct_psf = self.imaging_psf(inst=rst_direct)
                self.norm = self.direct_psf.max()

                # Calculate unaberrated coronagraph image for contrast floor
                self.coro_psf = self.imaging_psf()

                self.iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
                self.owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
                self.sim.working_area(im=self.coro_psf, inner_rad=self.iwa, outer_rad=self.owa)
                self.dh_mask = self.sim.WA

        def calculate_unaberrated_contrast(self):
                self.sim.raw_coronagraph()
                # Return the coronagraphic simulator (a tuple in the RST case!)
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
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

        def display_opd(self):
                plt.figure(figsize=(self.nb_actu, self.nb_actu))
                self.sim.dm1.display(what='opd', opd_vmax=self.wfe_aber, colorbar_orientation='horizontal',
                     title='Aberrated actuator pair')
