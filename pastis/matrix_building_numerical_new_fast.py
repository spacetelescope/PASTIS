"""
This module contains functions that construct the matrix M for PASTIS *NUMERICALLY FROM THE RESPECTIVE E2E SIMULATOR*
 and saves it.

 Currently supported:
 JWST
 LUVOIR
 #TODO: HiCAT (already exists in notebook HiCAT/4)
 """

import os
import time
import functools
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import hcipy as hc

from config import CONFIG_INI
import util_pastis as util
from e2e_simulators.luvoir_imaging_new import LuvoirAPLC


def num_matrix_luvoir(design, savepsfs=True, saveopds=True):
    """
    Generate a numerical PASTIS matrix for a LUVOIR A coronagraph.

    All inputs are read from the (local) configfile and saved to the specified output directory.
    The LUVOIR STDT delivery in May 2018 included three different apodizers
    we can work with, so I will implement an easy way of making a choice between them.
    :param design: string, what coronagraph design to use - 'small', 'medium' or 'large'
    :param savepsfs: bool, if True, all PSFs will be saved to disk individually, as fits files, additionally to the
                     total PSF cube. If False, the total cube will still get saved at the very end of the script.
    :param saveopds: bool, if True, all pupil surface maps of aberrated segment pairs will be saved to disk
    """

    # Keep track of time
    start_time = time.time()   # runtime is currently around 150 minutes
    print('Building numerical matrix for LUVOIR\n')

    ### Parameters

    # System parameters
    overall_dir = util.create_data_path(CONFIG_INI.get('local', 'local_data_path'), telescope = 'luvoir-'+design)
    os.makedirs(overall_dir, exist_ok=True)
    resDir = os.path.join(overall_dir, 'matrix_numerical')
    max_zern_number = CONFIG_INI.getint('calibration', 'maxzernike')

    # General telescope parameters
    nb_seg = CONFIG_INI.getint('LUVOIR', 'nb_subapertures')
    wvln = CONFIG_INI.getfloat('LUVOIR', 'lambda') * 1e-9  # m
    diam = CONFIG_INI.getfloat('LUVOIR', 'diameter')  # m
    nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration') * 1e-9   # m

    # Image system parameters
    im_lamD = CONFIG_INI.getfloat('numerical', 'im_size_lamD_hcipy')  # image size in lambda/D
    sampling = CONFIG_INI.getfloat('numerical', 'sampling')

    # Print some of the defined parameters
    print('LUVOIR apodizer design: {}'.format(design))
    print()
    print('Wavelength: {} m'.format(wvln))
    print('Telescope diameter: {} m'.format(diam))
    print('Number of segments: {}'.format(nb_seg))
    print()
    print('Image size: {} lambda/D'.format(im_lamD))
    print('Sampling: {} px per lambda/D'.format(sampling))

    # Create necessary directories if they don't exist yet
    os.makedirs(resDir, exist_ok=True)
    os.makedirs(os.path.join(resDir, 'OTE_images'), exist_ok=True)
    os.makedirs(os.path.join(resDir, 'psfs'), exist_ok=True)

    #  Copy configfile to resulting matrix directory
    util.copy_config(resDir)

    ### Instantiate Luvoir telescope with chosen apodizer design
    optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
    luvoir = LuvoirAPLC(optics_input, design, sampling)

    ### Instantiate the primary segmented mirror as a DMs a la HCIPy
    luvoir.make_segment_zernike_primary(max_zern_number)


    ### Dark hole mask
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    ### Reference images for contrast normalization and coronagraph floor
    unaberrated_coro_psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    norm = np.max(ref)

    dh_intensity = (unaberrated_coro_psf / norm) * dh_mask
    contrast_floor = np.mean(dh_intensity[np.where(dh_mask != 0)])
    print('contrast floor: {}'.format(contrast_floor))
    nonaberrated_coro_psf, ref,inter_ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate='efield')
    Efield_ref = inter_ref['at_science_focus'].electric_field

    ### Generating the PASTIS matrix and a list for all contrasts
    number_of_modes = luvoir.sm.num_actuators
    print('number of modes'.format(number_of_modes))
    matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    all_psfs = []
    all_contrasts = []

    print('nm_aber: {} m'.format(nm_aber))

    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0, number_of_modes):
        print(pp)
        zernike_coeffs = np.zeros([luvoir.sm.num_actuators])
        zernike_coeffs[pp] = nm_aber / 2
        luvoir.sm.actuators = zernike_coeffs
        luvoir.sm(luvoir.wf_aper)
        aberrated_coro_psf, ref, inter = luvoir.calc_psf(ref=True, display_intermediate=False,
                                                            return_intermediate='efield')
        focus_field1 = inter['at_science_focus']
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    mat_fast = np.zeros([number_of_modes, number_of_modes])
    for i in range(0, number_of_modes):
        for j in range(0, number_of_modes):
            test = np.real(
                (focus_fieldS[i].electric_field - Efield_ref) * np.conj(focus_fieldS[j].electric_field - Efield_ref))
            dh_test = (test / norm) * dh_mask
            contrast = np.mean(dh_test[np.where(dh_mask != 0)])
            mat_fast[i, j] = contrast




    #
    #
    # # Transform saved lists to arrays
    # all_psfs = np.array(all_psfs)
    # all_contrasts = np.array(all_contrasts)
    #
    # # Save the PSF image *cube* as well (as opposed to each one individually)
    # hc.write_fits(all_psfs, os.path.join(resDir, 'psfs', 'psf_cube' + '.fits'),)
    # np.savetxt(os.path.join(resDir, 'contrasts.txt'), all_contrasts, fmt='%e')

    # Filling the off-axis elements
    matrix_two_N = np.copy(matrix_direct)      # This is just an intermediary copy so that I don't mix things up.
    matrix_pastis = np.copy(mat_fast)     # This will be the final PASTIS matrix.

    # for i in range(number_of_modes):
    #     for j in range(number_of_modes):
    #         if i != j:
    #             matrix_off_val = (matrix_two_N[i,j] - matrix_two_N[i,i] - matrix_two_N[j,j]) / 2.
    #             matrix_pastis[i,j] = matrix_off_val
    #             print('Off-axis for i{}-j{}: {}'.format(i+1, j+1, matrix_off_val))



    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    matrix_pastis /= np.square(nm_aber * 1e9)    #  1e9 converts the calibration aberration back to nanometers

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_Multiple_Modes_MaxZer_' + str(max_zern_number)
    hc.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Re_matrix_num_Multiple_Modes_MaxZer_' + str(max_zern_number)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Im_matrix_num_Multiple_Modes_MaxZer_' + str(max_zern_number)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))


    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for matrix_building.py:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))
    
    return overall_dir


if __name__ == '__main__':

        # Pick the function of the telescope you want to run
        #num_matrix_jwst()

        coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
        num_matrix_luvoir(design=coro_design)
