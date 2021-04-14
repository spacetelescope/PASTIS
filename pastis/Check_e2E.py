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
from e2e_simulators.luvoir_imaging_onephot import LuvoirAPLC


def num_fields_luvoir(design, savepsfs=True, saveopds=True):
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

    print('Setting everything up\n')

    ### Parameters

    # System parameters
    overall_dir = util.create_data_path(CONFIG_INI.get('local', 'local_data_path'), telescope = 'luvoir-'+design)
    os.makedirs(overall_dir, exist_ok=True)
    resDir = os.path.join(overall_dir, 'matrix_numerical')

    # Moving parts parameters
    max_LO = CONFIG_INI.getint('calibration', 'max_LO')
    max_MID = CONFIG_INI.getint('calibration', 'max_MID')
    max_HI = CONFIG_INI.getint('calibration', 'max_HI')
    num_DM_act = CONFIG_INI.getint('calibration', 'num_DM_act')

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

    ### Instantiate the moving parts as a DMs a la HCIPy
    luvoir.make_LO_Modes(max_LO)
    luvoir.make_segment_zernike_primary(max_MID)
    luvoir.make_HI_Modes(max_HI)
    luvoir.make_DM(num_DM_act)

    n_LO = luvoir.zm.num_actuators
    n_MID = luvoir.sm.num_actuators
    n_HI = luvoir.fm.num_actuators
    n_DM = luvoir.dm.num_actuators


    ### Set up the sampling for zernike sensors
    z_pup_downsample = CONFIG_INI.getfloat('numerical', 'z_pup_downsample')
    N_pup_z = np.int(luvoir.pupil_grid.shape[0] / z_pup_downsample)
    grid_zernike = hc.field.make_pupil_grid(N_pup_z, diameter=luvoir.diam)

    ### Dark hole mask
    dh_outer = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_inner = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(luvoir.focal_det)
    dh_mask = (dh_outer - dh_inner).astype('bool')

    ### Reference images for contrast normalization and coronagraph floor

    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes

    unaberrated_coro_psf, ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    norm = np.max(ref)

    dh_intensity = (unaberrated_coro_psf / norm) * dh_mask
    contrast_floor = np.mean(dh_intensity[np.where(dh_mask != 0)])
    print('contrast floor: {}'.format(contrast_floor))
    nonaberrated_coro_psf, ref,inter_ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate='efield')
    Efield_ref = inter_ref['at_science_focus'].electric_field

    ### Generating the Efield for LO modes to science plane

    print('Generating the Efield for LO modes to science plane')
    print('number of LO modes'.format(n_LO))
    # matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    # all_psfs = []
    # all_contrasts = []

    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)

    print('nm_aber: {} m'.format(nm_aber))
    start_time = time.time()
    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0,n_LO):
        print(pp)
        LO_modes = np.zeros(n_LO)
        LO_modes[pp] = nm_aber / 2
        luvoir.zm.actuators  = LO_modes
        aberrated_coro_psf, ref, inter = luvoir.calc_psf(ref=True, display_intermediate=False,
                                                            return_intermediate='efield')
        focus_field1 = inter['at_science_focus']
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    mat_fast = np.zeros([n_LO, n_LO])
    for i in range(0, n_LO):
        for j in range(0, n_LO):
            test = np.real(
                (focus_fieldS[i].electric_field - Efield_ref) * np.conj(focus_fieldS[j].electric_field - Efield_ref))
            dh_test = (test / norm) * dh_mask
            contrast = np.mean(dh_test[np.where(dh_mask != 0)])
            mat_fast[i, j] = contrast
    matrix_pastis = np.copy(mat_fast)     # This will be the final PASTIS matrix.
    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    matrix_pastis /= np.square(nm_aber * 1e9)    #  1e9 converts the calibration aberration back to nanometers

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_LO_' + str(max_LO)
    hc.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Re_matrix_num_LO_' + str(max_LO)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Im_matrix_num_LO_' + str(max_LO)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))


    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for LO modes:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))

    ### Generating the Efield for MID modes to science plane

    print('Generating the Efield for MID modes to science plane')
    print('number of MID modes'.format(n_MID))
    # matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    # all_psfs = []
    # all_contrasts = []

    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)

    print('nm_aber: {} m'.format(nm_aber))
    start_time = time.time()
    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0,n_MID):
        print(pp)
        MID_modes = np.zeros(n_MID)
        MID_modes[pp] = nm_aber / 2
        luvoir.sm.actuators  = MID_modes
        aberrated_coro_psf, ref, inter = luvoir.calc_psf(ref=True, display_intermediate=False,
                                                            return_intermediate='efield')
        focus_field1 = inter['at_science_focus']
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    mat_fast = np.zeros([n_MID, n_MID])
    for i in range(0, n_MID):
        for j in range(0, n_MID):
            test = np.real(
                (focus_fieldS[i].electric_field - Efield_ref) * np.conj(focus_fieldS[j].electric_field - Efield_ref))
            dh_test = (test / norm) * dh_mask
            contrast = np.mean(dh_test[np.where(dh_mask != 0)])
            mat_fast[i, j] = contrast
    matrix_pastis = np.copy(mat_fast)     # This will be the final PASTIS matrix.
    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    matrix_pastis /= np.square(nm_aber * 1e9)    #  1e9 converts the calibration aberration back to nanometers

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_MID_' + str(max_MID)
    hc.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Re_matrix_num_MID_' + str(max_MID)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Im_matrix_num_MID_' + str(max_MID)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for MID modes:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))

    ### Generating the Efield for HI modes to science plane

    print('Generating the Efield for HI modes to science plane')
    print('number of HI modes'.format(n_HI))
    # matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    # all_psfs = []
    # all_contrasts = []

    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)

    print('nm_aber: {} m'.format(nm_aber))
    start_time = time.time()
    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0, n_HI):
        print(pp)
        HI_modes = np.zeros(n_HI)
        HI_modes[pp] = nm_aber / 2
        luvoir.fm.actuators = HI_modes
        aberrated_coro_psf, ref, inter = luvoir.calc_psf(ref=True, display_intermediate=False,
                                                         return_intermediate='efield')
        focus_field1 = inter['at_science_focus']
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    mat_fast = np.zeros([n_HI, n_HI])
    for i in range(0, n_HI):
        for j in range(0, n_HI):
            test = np.real(
                (focus_fieldS[i].electric_field - Efield_ref) * np.conj(focus_fieldS[j].electric_field - Efield_ref))
            dh_test = (test / norm) * dh_mask
            contrast = np.mean(dh_test[np.where(dh_mask != 0)])
            mat_fast[i, j] = contrast
    matrix_pastis = np.copy(mat_fast)  # This will be the final PASTIS matrix.
    # Normalize matrix for the input aberration - this defines what units the PASTIS matrix will be in. The PASTIS
    # matrix propagation function (util.pastis_contrast()) then needs to take in the aberration vector in these same
    # units. I have chosen to keep this to 1nm, so, we normalize the PASTIS matrix to units of nanometers.
    matrix_pastis /= np.square(nm_aber * 1e9)  # 1e9 converts the calibration aberration back to nanometers

    # Save matrix to file
    filename_matrix = 'PASTISmatrix_num_HI_' + str(max_HI)
    hc.write_fits(matrix_pastis, os.path.join(resDir, filename_matrix + '.fits'))
    print('Matrix saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Re_matrix_num_HI_' + str(max_HI)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_Im_matrix_num_HI_' + str(max_HI)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for HI modes:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))

    ### Generating the Efield for LO modes to LOWFS

    print('Generating the Efield for LO modes to LOWFS')
    print('number of LO modes'.format(n_LO))
    # matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    # all_psfs = []
    # all_contrasts = []


    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes

    zernike_ref = luvoir.prop_LOWFS()
    zernike_ref_sub_real = hc.field.subsample_field(zernike_ref.real, z_pup_downsample, grid_zernike, statistic='mean')
    zernike_ref_sub_imag = hc.field.subsample_field(zernike_ref.imag, z_pup_downsample, grid_zernike, statistic='mean')
    Efield_ref = zernike_ref_sub_real + 1j*zernike_ref_sub_imag


    print('nm_aber: {} m'.format(nm_aber))
    start_time = time.time()
    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0, n_LO):
        print(pp)
        LO_modes = np.zeros(n_LO)
        LO_modes[pp] = nm_aber / 2
        luvoir.zm.actuators = LO_modes
        zernike_meas = luvoir.prop_LOWFS()
        zernike_meas_sub_real = hc.field.subsample_field(zernike_meas.real, z_pup_downsample, grid_zernike,
                                                        statistic='mean')
        zernike_meas_sub_imag = hc.field.subsample_field(zernike_meas.imag, z_pup_downsample, grid_zernike,
                                                        statistic='mean')
        focus_field1 = zernike_meas_sub_real + 1j * zernike_meas_sub_imag
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    # Save matrix to file

    filename_matrix = 'EFIELD_LOWFS_Re_matrix_num_LO_' + str(max_LO)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_LOWFS_Im_matrix_num_LO_' + str(max_LO)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for LO modes and LOWFS:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))


    ### Generating the Efield for MID modes to LOWBFS

    print('Generating the Efield for MID modes to OBWFS')
    print('number of MID modes'.format(n_MID))
    # matrix_direct = np.zeros([number_of_modes, number_of_modes])   # Generate empty matrix
    # all_psfs = []
    # all_contrasts = []

    ### Reference images for contrast normalization and coronagraph floor

    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes

    zernike_ref = luvoir.prop_OBWFS()
    zernike_ref_sub_real = hc.field.subsample_field(zernike_ref.real, z_pup_downsample, grid_zernike, statistic='mean')
    zernike_ref_sub_imag = hc.field.subsample_field(zernike_ref.imag, z_pup_downsample, grid_zernike, statistic='mean')
    Efield_ref = zernike_ref_sub_real + 1j*zernike_ref_sub_imag


    print('nm_aber: {} m'.format(nm_aber))
    start_time = time.time()
    focus_fieldS = []
    focus_fieldS_Re = []
    focus_fieldS_Im = []

    for pp in range(0, n_MID):
        print(pp)
        MID_modes = np.zeros(n_MID)
        MID_modes[pp] = nm_aber / 2
        luvoir.sm.actuators = MID_modes
        zernike_meas = luvoir.prop_OBWFS()
        zernike_meas_sub_real = hc.field.subsample_field(zernike_meas.real, z_pup_downsample, grid_zernike,
                                                        statistic='mean')
        zernike_meas_sub_imag = hc.field.subsample_field(zernike_meas.imag, z_pup_downsample, grid_zernike,
                                                        statistic='mean')
        focus_field1 = zernike_meas_sub_real + 1j * zernike_meas_sub_imag
        focus_fieldS.append(focus_field1)
        focus_fieldS_Re.append(focus_field1.real)
        focus_fieldS_Im.append(focus_field1.imag)

    # Save matrix to file

    filename_matrix = 'EFIELD_OBWFS_Re_matrix_num_MID_' + str(max_MID)
    hc.write_fits(focus_fieldS_Re, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Real saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    filename_matrix = 'EFIELD_OBWFS_Im_matrix_num_MID_' + str(max_MID)
    hc.write_fits(focus_fieldS_Im, os.path.join(resDir, filename_matrix + '.fits'))
    print('Efield Imag saved to:', os.path.join(resDir, filename_matrix + '.fits'))

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for MID modes and OBWFS:', end_time - start_time, 'sec =', (end_time - start_time) / 60, 'min')
    print('Data saved to {}'.format(resDir))

    return overall_dir


if __name__ == '__main__':

        # Pick the function of the telescope you want to run
        #num_matrix_jwst()

        coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
        num_fields_luvoir(design=coro_design)
