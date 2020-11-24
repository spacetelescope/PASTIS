"""
This module contains useful functions to interface with the HiCAT simulator.
"""

import logging
from astropy.io import fits
import os

from pastis.config import CONFIG_PASTIS

log = logging.getLogger()

try:
    import hicat.simulators
except ImportError:
    log.info('HiCAT simulator not imported.')


def set_up_hicat(apply_continuous_dm_maps=False):
    """
    Return a configured instance of the HiCAT simulator.

    Sets the pupil mask, whether the IrisAO is in or out, apodizer, Lyot stop and detector. Optionally, loads DM maps
    onto the two continuous face-sheet Boston DMs.
    :param apply_continuous_dm_maps: bool, whether to load BostonDM maps from path specified in configfile, default False
    :return: instance of HICAT_Sim()
    """

    hicat_sim = hicat.simulators.hicat_sim.HICAT_Sim()

    hicat_sim.pupil_maskmask = CONFIG_PASTIS.get('HiCAT', 'pupil_mask')  # I will likely have to implement a new pupil mask
    hicat_sim.iris_ao = CONFIG_PASTIS.get('HiCAT', 'iris_ao')
    hicat_sim.apodizer = CONFIG_PASTIS.get('HiCAT', 'apodizer')
    hicat_sim.lyot_stop = CONFIG_PASTIS.get('HiCAT', 'lyot_stop')
    hicat_sim.detector = 'imager'

    log.info(hicat_sim.describe())

    # Load Boston DM maps into HiCAT simulator
    if apply_continuous_dm_maps:
        path_to_dh_solution = CONFIG_PASTIS.get('HiCAT', 'dm_maps_path')
        dm1_surface, dm2_surface = read_continuous_dm_maps_hicat(path_to_dh_solution)
        hicat_sim.dm1.set_surface(dm1_surface)
        hicat_sim.dm2.set_surface(dm2_surface)

        log.info(f'BostonDM maps applied from {path_to_dh_solution}.')

    return hicat_sim


def read_continuous_dm_maps_hicat(path_to_dm_maps):
    """
    Read Boston DM maps from disk and return as one list per DM.
    Hijacked partially from StrokeMinimizatoin.restore_last_strokemin_dm_shapes()
    :param path_to_dm_maps: string, absolute path to folder containing DM maps to load
    :return: DM1 actuator map array, DM2 actuator map array; in m
    """

    surfaces = []
    for dmnum in [1, 2]:
        actuators_2d = fits.getdata(os.path.join(path_to_dm_maps, f'dm{dmnum}_command_2d_noflat.fits'))
        surfaces.append(actuators_2d)

    return surfaces[0], surfaces[1]
