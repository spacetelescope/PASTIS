import os
os.chdir("/Users/asahoo/repos/PASTIS")
import time
from shutil import copy
from astropy.io import fits
import astropy.units as u
import hcipy
import numpy as np
import pastis.util as util
from pastis.config import CONFIG_PASTIS
from pastis.e2e_simulators.luvoir_imaging import LuvoirA_APLC
from pastis.e2e_simulators.generic_segmented_telescopes import SegmentedAPLC
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import exoscene.image
import exoscene.star
import exoscene.planet
from exoscene.planet import Planet
from astropy.io import fits as pf
from pastis.analytical_pastis.temporal_analysis import req_closedloop_calc_batch
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from matplotlib.colors import TwoSlopeNorm
from pastis.pastis_analysis import calculate_sigma, calculate_segment_constraints
from pastis.plotting import plot_mu_map


APLC_DESIGN =  CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
DM = 'harris_seg_mirror'
fpath = "/Users/asahoo/Documents/ultra/Sensitivities2.xlsx"
nb_seg = CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures')
pad_orientations = np.pi/2*np.ones(nb_seg)
DM_SPEC = (fpath, pad_orientations, True, False, False)
run_matrix = MatrixEfieldLuvoirA(which_dm=DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))

run_matrix.calc()
dir_run = run_matrix.overall_dir
print(f'All saved to {dir_run}.')

pastis_matrix = fits.getdata(os.path.join(dir_run, "matrix_numerical","pastis_matrix.fits"))
evecs, evals = modes_from_matrix('LUVOIR', dir_run)
mu_map = calculate_segment_constraints(pastis_matrix, 1e-11, 0.0)

optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
luvoir = LuvoirA_APLC(optics_input, APLC_DESIGN, sampling)

N_harris = 5  # number of harris modes, (thermal only)
harris_coeffs_numaps = np.zeros([N_harris, 600])
harris_modes_std = mu_map


###plot multi mode
for qq in range(N_harris):
    harris_coeffs_tmp = np.zeros(600)
    for kk in range(nb_seg):
        harris_coeffs_tmp[qq + (kk) * N_harris] = harris_modes_std[qq + (kk) * N_harris]
    harris_coeffs_numaps[qq] = harris_coeffs_tmp  # arranged 600 elements into 5*120 elements

harris_coeffs_table = np.zeros([N_harris, nb_seg])
for qq in range(N_harris):
    for kk in range(nb_seg):
        harris_coeffs_table[qq, kk] = mu_map[qq + (kk) * N_harris]  # numpy ndarray 120

nu_maps = []
for qq in range(N_harris):
    harris_coeffs = harris_coeffs_numaps[qq]
    luvoir.harris_sm.actuators = harris_coeffs * 1e-9 / 2  # m
    nu_maps.append(luvoir.harris_sm.surface)

plot1 = plot_mu_map("LUVOIR", mu_map, luvoir, dir_run, 1e-11, limits=None, fname_suffix='', save=False )