"""
This script computes the contrast for a random IrisAO mislignment on the HiCAT simulator.
"""

import sys
import os
import time
import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import poppy
import hicat.simulators

from config import CONFIG_INI
import util_pastis as util

def contrast_hicat_num(dir, matrix_mode='hicat', rms=1*u.nm):

    nb_seg = 36

    # Keep track of time
    start_time = time.time()   # runtime currently is around 12 min

    # Import numerical PASTIS matrix for HiCAT sim
    filename = 'PASTISmatrix_num_HiCAT_piston_Noll1'
    matrix_pastis = fits.getdata(os.path.join(dir, filename + '.fits'))

    # Create random aberration coefficients
    aber = np.random.random([nb_seg])   # piston values in input units
    print('PISTON ABERRATIONS:', aber)

    # Normalize to the RMS value I want
    rms_init = util.rms(aber)
    aber *= rms.value / rms_init
    calc_rms = util.rms(aber) * u.nm
    aber *= u.nm    # making sure the aberration has the correct units
    print("Calculated RMS:", calc_rms)

    # Remove global piston
    aber -= np.mean(aber)

    ### BASELINE PSF - NO ABERRATIONS, NO CORONAGRAPH
    print('Generating baseline PSF from E2E - no coronagraph, no aberrations')
    hc = hicat.simulators.hicat_sim.HICAT_Sim()
    hc.iris_ao = 'iris_ao'
    hc.apodizer = 'cnt1_apodizer'
    hc.lyot_stop = 'cnt1_apodizer_lyot_stop'
    hc.include_fpm = False

    psf_perfect = hc.calc_psf(display=False, return_intermediates=False)
    normp = np.max(psf_perfect[0].data)
    #psf_perfect = psf_perfect[0].data / normp   don't actually need the perfect PSF

    ### HiCAT sim
    start_e2e = time.time()
    # Set up the HiCAT simulator, get PSF
    hc.apodizer = 'cnt1_apodizer'
    hc.lyot_stop = 'cnt1_apodizer_lyot_stop'
    hc.include_fpm = True

    # Calculate coro PSF without aberrations
    psf_coro = hc.calc_psf(display=False, return_intermediates=False)
    psf_coro = psf_coro[0].data / normp


    print('Calculating E2E contrast...')
    # Put aberration on Iris AO
    for nseg in range(nb_seg):
        hc.iris_dm.set_actuator(nseg+1, aber[nseg], 0, 0)

    psf_hicat = hc.calc_psf(display=False, return_intermediates=False)
    psf_hicat = psf_hicat[0].data / normp

    # Create DH
    dh_mask = util.create_dark_hole(psf_hicat, iwa=5, owa=12, samp=13 / 4)
    # Get the mean contrast
    hicat_dh_psf = psf_hicat * dh_mask
    contrast_hicat = np.mean(hicat_dh_psf[np.where(hicat_dh_psf != 0)])
    end_e2e = time.time()

    ###
    # Calculate baseline contrast
    baseline_dh = psf_coro * dh_mask
    contrast_base = np.mean(baseline_dh[np.where(baseline_dh != 0)])

    ## MATRIX PASTIS
    print('Generating contrast from matrix-PASTIS')
    start_matrixpastis = time.time()
    # Get mean contrast from matrix PASTIS
    contrast_matrix = util.pastis_contrast(aber, matrix_pastis) + contrast_base   # calculating contrast with PASTIS matrix model
    end_matrixpastis = time.time()

    # Outputs
    print('\n--- CONTRASTS: ---')
    print('Mean contrast from E2E:', contrast_hicat)
    print('Contrast from matrix PASTIS:', contrast_matrix)

    print('\n--- RUNTIMES: ---')
    print('E2E: ', end_e2e-start_e2e, 'sec =', (end_e2e-start_e2e)/60, 'min')
    print('Matrix PASTIS: ', end_matrixpastis-start_matrixpastis, 'sec =', (end_matrixpastis-start_matrixpastis)/60, 'min')

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for contrast_calculation_simple.py: {} sec = {} min'.format(runtime, runtime/60))

    return contrast_hicat, contrast_matrix


if __name__ == '__main__':

    c_e2e, c_matrix = contrast_hicat_num(dir='/Users/ilaginja/Documents/Git/PASTIS/Jupyter Notebooks/HiCAT', rms=10*u.nm)
