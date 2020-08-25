"""
This module contains useful functions to interface with the HiCAT simulator.
"""

import os
import hicat.simulators
from config import CONFIG_INI


def set_up_hicat():

    hicat_sim = hicat.simulators.hicat_sim.HICAT_Sim()

    hicat_sim.pupil_maskmask = 'circular'  # I will likely have to implement a new pupil mask
    hicat_sim.iris_ao = 'iris_ao'
    hicat_sim.apodizer = 'no_apodizer'
    hicat_sim.lyot_stop = 'circular'
    hicat_sim.detector = 'imager'

    return hicat_sim