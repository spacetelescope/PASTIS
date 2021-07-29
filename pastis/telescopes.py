"""
This module contains class and all necessary requierement to work RST
"""

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
import pastis.e2e_simulators.webbpsf_imaging as webbpsf_imaging



class RST():
        def __init__(self):
                self.sim = webbpsf_imaging.set_up_cgi()
                self.nb_actu = self.sim.nbactuator

                self.wfe_aber = CONFIG_PASTIS.getfloat('RST', 'calibration_aberration') * 1e-9   # m

        def calculate_unaberrated_contrast_and_normalization(self):
                # Calculate direct reference images for contrast normalization
                rst_direct = self.sim.raw_coronagraph()
                direct = rst_direct.imaging_psf()
                self.direct_psf = direct[0].data
                self.norm = direct_psf.max()

                # Calculate unaberrated coronagraph image for contrast floor
                coro_image = self.imaging_psf()
                coro_psf = coro_image[0].data / self.norm

                self.iwa = CONFIG_PASTIS.getfloat('RST', 'IWA')
                self.owa = CONFIG_PASTIS.getfloat('RST', 'OWA')
                self.sim.working_area(im=coro_psf, inner_rad=self.iwa, outer_rad=self.owa)
                self.dh_mask = self.sim.WA

                # Return the coronagraphic simulator (a tuple in the RST case!)
                self.coro_simulator = self.sim

                # Calculate coronagraph floor in dark hole
                self.contrast_floor = self.contrast()

        def flatten(self):
                self.sim.dm1.flatten()
                return []

        def push_seg(self, seg, amplitude):
                actu_x, actu_y = util.seg_to_dm_xy(self.nb_actu, seg)
                self.sim.dm1.set_actuator(actu_x, actu_y, amplitude)

        def imaging_psf(self):
                self.psf = self.sim.calc_psf(nlambda=1, fov_arcsec=1.6)
                return self.psf

        def imaging_efielf(self):
                _psf , inter = self.sim.calc_psf(nlambda=1, fov_arcsec=1.6, return_intermediates=True)
                self.efield = inter[-1].wavefront
                return self.efield

        def contrast(self):
                self.imaging_psf()
                return util.dh_mean(self.psf, self.dh_mask)

        def display_opd(self):
                self.dm1.display(what='opd', opd_vmax=self.wfe_aber, colorbar_orientation='horizontal',
                     title='Aberrated actuator pair')