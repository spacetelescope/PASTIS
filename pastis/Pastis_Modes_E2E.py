"""
This module contains functions that calculates the field sensitivities for a given coronagraph design

"""

import os
import time
import functools
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import astropy.units as u
import hcipy as hc

from config import CONFIG_INI
import util_pastis as util
from e2e_simulators.luvoir_imaging_onephot import LuvoirAPLC
import pretty_figures as pf
import astropy.table


# def test_e2e_modes(design, savepsfs=True, saveopds=True):
# """
# Test if the matrix calcuation got it right and compare with e2e calcuations
# """

print(' Loading basic parameters for this specific telescope \n')

### Parameters
design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
# System parameters
root_dir = CONFIG_INI.get('local', 'local_data_path')
output_dir = CONFIG_INI.get('local', 'output_data_folder')
overall_dir = root_dir + output_dir + 'luvoirA-' + design
# Moving parts parameters
analysis_name = 'LUVOIRA_APLC_' + design
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

# #  Copy configfile to resulting matrix directory
# util.copy_config(resDir)

### Instantiate Luvoir telescope with chosen apodizer design
### Extract interresting parameters from object to have them ready in code
optics_input = CONFIG_INI.get('LUVOIR', 'optics_path')
datadir = optics_input
luvoir = LuvoirAPLC(optics_input, design, sampling)
nb_seg = CONFIG_INI.getint('LUVOIR', 'nb_subapertures')
wvln = CONFIG_INI.getfloat('LUVOIR', 'lambda') * 1e-9  # m
diam = CONFIG_INI.getfloat('LUVOIR', 'diameter')  # m
nm_aber = CONFIG_INI.getfloat('calibration', 'single_aberration') * 1e-9   # m
eunit = nm_aber
nm = eunit
im_lamD = CONFIG_INI.getfloat('numerical', 'im_size_lamD_hcipy')  # image size in lambda/D
sampling = CONFIG_INI.getfloat('numerical', 'sampling')
luvoir = LuvoirAPLC(optics_input, design, sampling)
npup = np.int(np.sqrt(luvoir.pupil_grid.x.shape[0]))
nimg = np.int(np.sqrt(luvoir.focal_det.x.shape[0]))




# Load the matrices
savedpath = overall_dir
filename_matrix = 'EFIELD_Re_matrix_num_LO_' + str(max_LO) +'.fits'
G_LO_real = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix = 'EFIELD_Im_matrix_num_LO_' + str(max_LO) +'.fits'
G_LO_imag = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix =  'EFIELD_Re_matrix_num_MID_' + str(max_MID) +'.fits'
G_MID_real = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix =  'EFIELD_Im_matrix_num_MID_' + str(max_MID) +'.fits'
G_MID_imag = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix =  'EFIELD_Re_matrix_num_HI_' + str(max_HI) +'.fits'
G_HI_real = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix =  'EFIELD_Im_matrix_num_HI_' + str(max_HI) +'.fits'
G_HI_imag = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix = 'EFIELD_LOWFS_Re_matrix_num_LO_' + str(max_LO) +'.fits'
G_LOWFS_real = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix = 'EFIELD_LOWFS_Im_matrix_num_LO_' + str(max_LO) +'.fits'
G_LOWFS_imag = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix = 'EFIELD_OBWFS_Re_matrix_num_MID_' + str(max_MID) +'.fits'
G_OBWFS_real = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
filename_matrix = 'EFIELD_OBWFS_Im_matrix_num_MID_' + str(max_MID) +'.fits'
G_OBWFS_imag = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))

# filename_matrix = 'EFIELD_Re_matrix_num_LO_15_OLD.fits'
# G_LO_real_OLD = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))
# filename_matrix = 'EFIELD_Im_matrix_num_LO_15_OLD.fits'
# G_LO_imag_OLD = fits.getdata(os.path.join(savedpath, 'matrix_numerical', filename_matrix))


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

# Make matrices
mat_LO = np.zeros([n_LO-1, n_LO-1])
for i in range(1, n_LO):
    for j in range(1, n_LO):
        tmpI = (G_LO_real[i]+1j*G_LO_imag[i] - Efield_ref)
        tmpJ = (G_LO_real[j]+1j*G_LO_imag[j] - Efield_ref)
        test = np.real(tmpI*np.conj(tmpJ))
        dh_test = (test / norm) * dh_mask
        contrast = np.mean(dh_test[np.where(dh_mask != 0)])
        mat_LO[i-1, j-1] = contrast
mat_MID = np.zeros([n_MID, n_MID])
for i in range(0, n_MID):
    for j in range(0, n_MID):
        tmpI = G_MID_real[i]+1j*G_MID_imag[i] - Efield_ref
        tmpJ = G_MID_real[j]+1j*G_MID_imag[j] - Efield_ref
        test = np.real(tmpI*np.conj(tmpJ))
        dh_test = (test / norm) * dh_mask
        contrast = np.mean(dh_test[np.where(dh_mask != 0)])
        mat_MID[i, j] = contrast
mat_HI = np.zeros([n_HI, n_HI])
for i in range(0, n_HI):
    for j in range(0, n_HI):
        tmpI = G_HI_real[i]+1j*G_HI_imag[i] - Efield_ref
        tmpJ = G_HI_real[j]+1j*G_HI_imag[j] - Efield_ref
        test = np.real(tmpI*np.conj(tmpJ))
        dh_test = (test / norm) * dh_mask
        contrast = np.mean(dh_test[np.where(dh_mask != 0)])
        mat_HI[i, j] = contrast

# Figures with eigenvalues
figures_path = overall_dir + '/figures'
os.makedirs(figures_path, exist_ok=True)
evalsLO, evecsLO = np.linalg.eig(mat_LO)
sorted_evalsLO = np.sort(evalsLO)
sorted_indicesLO = np.argsort(evalsLO)
sorted_evecsLO = evecsLO[:, sorted_indicesLO]
fig = plt.figure(figsize=(14, 8))
plt.plot(sorted_evalsLO, label='Sorted lowest to highest evals',linewidth=1.5,color = 'blue')
plt.semilogy()
plt.xlabel('LO Eigenmodes')
plt.ylabel('Log LO Eigenvalues')
plt.legend()
file_name = figures_path + '/' + analysis_name + '_LO_eigenvalues' + '.pdf'
fig.savefig(file_name)

evalsMID, evecsMID = np.linalg.eig(mat_MID)
sorted_evalsMID = np.sort(evalsMID)
sorted_indicesMID = np.argsort(evalsMID)
sorted_evecsMID = evecsMID[:, sorted_indicesMID]
fig = plt.figure(figsize=(14, 8))
#plt.pMIDt(evals, label='Unsorted from eigendecomposition')
plt.plot(sorted_evalsMID, label='Sorted lowest to highest evals',linewidth=1.5,color = 'blue')
plt.semilogy()
plt.xlabel('MID Eigenmodes')
plt.ylabel('log MID Eigenvalues')
plt.legend()
file_name =  figures_path + '/' + analysis_name + 'MID_eigenvalues' + '.pdf'
fig.savefig(file_name)

evalsHI, evecsHI = np.linalg.eig(mat_HI)
sorted_evalsHI = np.sort(evalsHI)
sorted_indicesHI = np.argsort(evalsHI)
sorted_evecsHI = evecsHI[:, sorted_indicesHI]
fig = plt.figure(figsize=(14, 8))
#plt.pHIt(evals, label='Unsorted from eigendecomposition')
plt.plot(sorted_evalsHI, label='Sorted lowest to highest evals',linewidth=1.5,color = 'blue')
plt.semilogy()
plt.xlabel('HI Eigenmodes')
plt.ylabel('log HI Eigenvalues')
plt.legend()
file_name = figures_path + '/' + analysis_name + 'HI_eigenvalues' + '.pdf'
fig.savefig(file_name)

# Calculate the segment based constraints
c_target_log = CONFIG_INI.getint('e2eparameters', 'c_target_log')
c_target = 10**(c_target_log)
n_repeat = CONFIG_INI.getint('e2eparameters', 'n_repeat')
mu_mapLO = np.sqrt(
    ((c_target   ) / (n_LO-1)) / (np.diag(mat_LO)))
mu_mapMID = np.sqrt(
    ((c_target ) / (n_MID)) / (np.diag(mat_MID)))
mu_mapHI = np.sqrt(
    ((c_target ) / (n_HI)) / (np.diag(mat_HI)))


# e2e calcuations for LO modes
# Initialization
LO_modes = np.zeros(n_LO)
MID_modes = np.zeros(n_MID)
HI_modes = np.zeros(n_HI)
DM_modes = np.zeros(n_DM)
luvoir.zm.actuators = LO_modes
luvoir.sm.actuators = MID_modes
luvoir.fm.actuators = HI_modes
luvoir.dm.actuators = DM_modes
unaber_e_field, ref, inter = luvoir.calc_psf(ref=True, return_intermediate='efield')

all_contr_rand_seg1 = []
for rep in range(n_repeat):
    print('Segment realization {}/{}'.format(rep + 1, n_repeat))
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    input_wf = luvoir.wf_aper
    LO_modes[1:n_LO] = np.random.normal(0,mu_mapLO*eunit,n_LO-1)
    luvoir.zm.actuators  = LO_modes/2
    tmp_pupil = luvoir.zm(input_wf)
    psf, ref, inter = luvoir.calc_psf(ref=True, return_intermediate='efield')
    dh_intensity = psf.intensity/norm * dh_mask
    # calibrated_intensity = np.abs(psf.electric_field - unaber_e_field.electric_field)**2
    # dh_calibrated_intensity = calibrated_intensity/norm * dh_mask
    test_contrast1 = np.mean(dh_intensity[np.where(dh_intensity != 0)]) - contrast_floor
    all_contr_rand_seg1.append(test_contrast1)
    print('contrast:', test_contrast1)

tmp_std = np.dot(mat_LO,np.diag(mu_mapLO)**2)
var = 2*np.dot(tmp_std,tmp_std)
std = np.sqrt(np.trace(var))
c_mean_exp = np.mean(all_contr_rand_seg1)
c_std_exp = np.std(all_contr_rand_seg1)
print(std)
fig = plt.figure(figsize=(16, 10))
plt.hist(all_contr_rand_seg1,30)
plt.title('E2E raw contrast, {} realizations, target contrast 1e-11'.format(n_repeat), size=20)
plt.xlabel('Mean contrast in DH', size=20)
plt.ylabel('PDF', size=20)
plt.axvline(c_target, c='r', ls='-.', lw='3')
plt.axvline(c_target + std, c='b', ls=':', lw=4)
plt.axvline(c_target - std, c='b', ls=':', lw=4)
plt.axvline(c_mean_exp, c='orange', ls='-.', lw='3')
plt.axvline(c_mean_exp + c_std_exp, c='c', ls=':', lw=4)
plt.axvline(c_mean_exp - c_std_exp, c='c', ls=':', lw=4)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
file_name = figures_path + '/' + analysis_name + '_LO_e2e_histogram' + '.pdf'
fig.savefig(file_name)


# e2e calcuations for MID modes
# Initialization
all_contr_rand_seg1 = []
for rep in range(n_repeat):
    print('Segment realization {}/{}'.format(rep + 1, n_repeat))
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    input_wf = luvoir.wf_aper
    MID_modes = np.random.normal(0,mu_mapMID*eunit,n_MID)
    luvoir.sm.actuators  = MID_modes/2
    tmp_pupil = luvoir.zm(input_wf)
    psf, ref, inter = luvoir.calc_psf(ref=True, return_intermediate='efield')
    dh_intensity = psf.intensity/norm * dh_mask
    calibrated_intensity = np.abs(psf.electric_field - unaber_e_field.electric_field)**2
    dh_calibrated_intensity = calibrated_intensity/norm * dh_mask
    test_contrast1 = np.mean(dh_intensity[np.where(dh_intensity != 0)]) - contrast_floor
    all_contr_rand_seg1.append(test_contrast1)
    print('contrast:', test_contrast1)

tmp_std = np.dot(mat_MID,np.diag(mu_mapMID)**2)
var = 2*np.dot(tmp_std,tmp_std)
std = np.sqrt(np.trace(var))
c_mean_exp = np.mean(all_contr_rand_seg1)
c_std_exp = np.std(all_contr_rand_seg1)
print(std)
fig = plt.figure(figsize=(16, 10))
plt.hist(all_contr_rand_seg1,30)
plt.title('E2E raw contrast, {} realizations, target contrast 1e-11'.format(n_repeat), size=20)
plt.xlabel('Mean contrast in DH', size=20)
plt.ylabel('PDF', size=20)
plt.axvline(c_target, c='r', ls='-.', lw='3')
plt.axvline(c_target + std, c='b', ls=':', lw=4)
plt.axvline(c_target - std, c='b', ls=':', lw=4)
plt.axvline(c_mean_exp, c='orange', ls='-.', lw='3')
plt.axvline(c_mean_exp + c_std_exp, c='c', ls=':', lw=4)
plt.axvline(c_mean_exp - c_std_exp, c='c', ls=':', lw=4)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
file_name = figures_path + '/' + analysis_name + '_MID_e2e_histogram' + '.pdf'
fig.savefig(file_name)

# e2e calcuations for MID modes
# Initialization
all_contr_rand_seg1 = []
for rep in range(n_repeat):
    print('Segment realization {}/{}'.format(rep + 1, n_repeat))
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    input_wf = luvoir.wf_aper
    HI_modes = np.random.normal(0,mu_mapHI*eunit,n_HI)
    luvoir.fm.actuators  = HI_modes/2
    tmp_pupil = luvoir.fm(input_wf)
    psf, ref, inter = luvoir.calc_psf(ref=True, return_intermediate='efield')
    dh_intensity = psf.intensity/norm * dh_mask
    calibrated_intensity = np.abs(psf.electric_field - unaber_e_field.electric_field)**2
    dh_calibrated_intensity = calibrated_intensity/norm * dh_mask
    test_contrast1 = np.mean(dh_intensity[np.where(dh_intensity != 0)]) - contrast_floor
    all_contr_rand_seg1.append(test_contrast1)
    print('contrast:', test_contrast1)

tmp_std = np.dot(mat_HI,np.diag(mu_mapHI)**2)
var = 2*np.dot(tmp_std,tmp_std)
std = np.sqrt(np.trace(var))
c_mean_exp = np.mean(all_contr_rand_seg1)
c_std_exp = np.std(all_contr_rand_seg1)
print(std)
fig = plt.figure(figsize=(16, 10))
plt.hist(all_contr_rand_seg1,30)
plt.title('E2E raw contrast, {} realizations, target contrast 1e-11'.format(n_repeat), size=20)
plt.xlabel('Mean contrast in DH', size=20)
plt.ylabel('PDF', size=20)
plt.axvline(c_target, c='r', ls='-.', lw='3')
plt.axvline(c_target + std, c='b', ls=':', lw=4)
plt.axvline(c_target - std, c='b', ls=':', lw=4)
plt.axvline(c_mean_exp, c='orange', ls='-.', lw='3')
plt.axvline(c_mean_exp + c_std_exp, c='c', ls=':', lw=4)
plt.axvline(c_mean_exp - c_std_exp, c='c', ls=':', lw=4)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=25)
file_name = figures_path + '/' + analysis_name + '_HI_e2e_histogram' + '.pdf'
fig.savefig(file_name)

# Make figures with nu maps for MID modes
N_zernike = max_MID
zernike_coeffs_numaps = np.zeros([N_zernike,n_MID])
for qq in range(N_zernike):
    zernike_coeffs_tmp = np.zeros([n_MID])
    for pp in range(120):
        zernike_coeffs_tmp[qq+(pp)*N_zernike] = mu_mapMID[qq+(pp)*N_zernike]
    zernike_coeffs_numaps[qq] = zernike_coeffs_tmp
zernike_coeffs_table = np.zeros([N_zernike,120])
for qq in range(N_zernike):
    zernike_coeffs_tmp = np.zeros([120])
    for pp in range(120):
        zernike_coeffs_table[qq,pp] = mu_mapMID[qq+(pp)*N_zernike]
nu_maps = []
for qq in range(N_zernike):
    zernike_coeffs = eunit*zernike_coeffs_numaps[qq]/2
    luvoir.sm.actuators = zernike_coeffs*10**(12)
    nu_maps.append(luvoir.sm.surface)


if len(nu_maps) ==1:
    fig, axs = plt.subplots(1, len(nu_maps), figsize=(60, 20))
    phaseplot = axs.imshow(np.reshape(nu_maps[0], [npup, npup]), cmap='hot')
    axs.axes.get_xaxis().set_visible(False)
    axs.axes.get_yaxis().set_visible(False)
    clb = fig.colorbar(phaseplot, ax=axs, shrink=0.4)
    clb.set_label('pm', rotation=90)

else:
    fig, axs = plt.subplots(1,len(nu_maps),figsize=(60,20))
    for qq in range(0,len(nu_maps)):
        print(qq)
        phaseplot = axs[qq].imshow(np.reshape(nu_maps[qq],[npup,npup]),cmap = 'hot')
        axs[qq].axes.get_xaxis().set_visible(False)
        axs[qq].axes.get_yaxis().set_visible(False)
        clb=  fig.colorbar(phaseplot,ax = axs[qq],shrink=0.4)
        clb.set_label('pm', rotation=90)

plt.show()
file_name = figures_path + '/' + analysis_name + '_MID_NU_Maps' + '.pdf'
fig.savefig(file_name)

# Make figures with nu maps for Hi modes
# mu_mapHI_Fudged = np.zeros(n_HI)
# mu_mapHI_Fudged[20:300] = mu_mapHI[20:300]

fourier_grid = hc.make_pupil_grid(dims=max_HI, diameter=max_HI)
rho0 = 2*np.sqrt(fourier_grid.x**2+fourier_grid.y**2)
rho = np.sort(rho0)

fig = plt.figure(figsize=(14, 8))
plt.plot(rho,np.log10(mu_mapHI*1000),'bo')
plt.xlabel('Spatial frequency, cycles/aperture')
plt.ylabel('log HI requirement, pm')
file_name = figures_path + '/' + analysis_name + '_HI_NU_Maps' + '.pdf'
fig.savefig(file_name)

# Making tables
# For LO modes

ZernikeList = []
for pp in range(0,n_LO-1):
    ZernikeList.append('Global Z' + np.str(pp +1));
table_LO = astropy.table.Table()
LO_Values_col = astropy.table.Table.Column(
        name = '$\Delta \epsilon_{11}^{LO}$ (nm)',
        data = mu_mapLO, dtype = float, format="7.2f")
LO_Labels_col = astropy.table.Table.Column(
        name = 'LO Modes', data = ZernikeList, dtype = str)
table_LO.add_column(LO_Labels_col)
table_LO.add_column(LO_Values_col)

filename = figures_path + '/' + 'Table_LO_w11.csv'
table_LO.write(filename, format='csv',overwrite=True)
filename = figures_path + '/' +  'Table_LO_w11.tex'
table_LO.write(filename, format='latex',overwrite=True)


# For MID mode

zernike_coeffs_numaps_max = np.zeros(max_MID)
for pp in range(0,max_MID):
    zernike_coeffs_numaps_max[pp] = np.max(zernike_coeffs_table[pp,:])*1000/2
zernike_coeffs_numaps_min = np.zeros(max_MID)
for pp in range(0,max_MID):
    zernike_coeffs_numaps_min[pp] = np.min(zernike_coeffs_table[pp,:])*1000/2
zernike_coeffs_numaps_median = np.zeros(max_MID)
for pp in range(0,max_MID):
    zernike_coeffs_numaps_median[pp] = np.median(zernike_coeffs_table[pp,:])*1000/2

ZernikeList = []
for pp in range(0,max_MID):
    ZernikeList.append('Segment Z' + np.str(pp));

table_MID = astropy.table.Table()
MID_Max_Values_col = astropy.table.Table.Column(
        name = 'max $ \Delta \epsilon_{11}^{MID}$ (pm)',
        data = zernike_coeffs_numaps_max, dtype = float, format="7.1f")
MID_Min_Values_col = astropy.table.Table.Column(
        name = 'min$ \Delta \epsilon_{11}^{MID}$ (pm)',
        data = zernike_coeffs_numaps_min, dtype = float, format="7.1f")
MID_Median_Values_col = astropy.table.Table.Column(
        name = 'median$ \Delta \epsilon_{11}^{MID}$ (pm)',
        data = zernike_coeffs_numaps_median, dtype = float, format="7.2f")
MID_Labels_col = astropy.table.Table.Column(
        name = 'MID Modes', data = ZernikeList, dtype = str)
table_MID.add_column(MID_Labels_col)
table_MID.add_column(MID_Max_Values_col)
table_MID.add_column(MID_Min_Values_col)
table_MID.add_column(MID_Median_Values_col)


filename = figures_path + '/' + 'Table_MID_w11.csv'
table_MID.write(filename, format='csv',overwrite=True)
filename = figures_path + '/' + 'Table_MID_w11.tex'
table_MID.write(filename, format='latex',overwrite=True)

# For HI modes


ZernikeList = []
for pp in range(0,n_LO-1):
    ZernikeList.append('Global Z' + np.str(pp +1));
table_LO = astropy.table.Table()
LO_Values_col = astropy.table.Table.Column(
        name = '$\Delta \epsilon_{11}^{LO}$ (nm)',
        data = mu_mapLO, dtype = float, format="7.2f")
LO_Labels_col = astropy.table.Table.Column(
        name = 'LO Modes', data = ZernikeList, dtype = str)
table_LO.add_column(LO_Labels_col)
table_LO.add_column(LO_Values_col)

filename = figures_path + '/' + 'Table_LO_w11.csv'
table_LO.write(filename, format='csv',overwrite=True)
filename = figures_path + '/' + 'Table_LO_w11.tex'
table_LO.write(filename, format='latex',overwrite=True)

SFList = ['Inside DH','Oustide DH']

where_core = rho < luvoir.apod_dict[design]['iwa'] /2
where_dh = (rho > luvoir.apod_dict[design]['iwa']  )& (rho <  luvoir.apod_dict[design]['owa'] )
where_outside_dh = (rho >  luvoir.apod_dict[design]['owa'])



SFList = ['Inside DH','Oustide DH']
SFreq = [np.mean(mu_mapHI[where_dh])*1000,np.mean(mu_mapHI[where_outside_dh])*1000]
table_HI = astropy.table.Table()
HI_Values_col = astropy.table.Table.Column(
        name = ' $ \Delta \epsilon_{11}^{HI}$ (pm)',
        data = SFreq, dtype = float, format="7.2f")
HI_Labels_col = astropy.table.Table.Column(
        name = 'MID Modes', data = SFList, dtype = str)
table_HI.add_column(HI_Labels_col)
table_HI.add_column(HI_Values_col)

filename = figures_path + '/' + 'Table_HI_w11.csv'
table_HI.write(filename, format='csv',overwrite=True)
filename = figures_path + '/' + 'Table_HI_w11.tex'
table_HI.write(filename, format='latex',overwrite=True)

# Make pretty pictures of modes and images at science camera and LOWFS

nimg = np.int(np.sqrt(luvoir.focal_det.x.shape))

# LO modes to science camera
# Poked modes for figures are  [1,3,4,7,9] need to automate


poked_modes = [1,3,4,7,9]
n_pokes = len(poked_modes)
phases = np.zeros([len(poked_modes),npup,npup])
detectors = np.zeros([len(poked_modes),nimg,nimg])
for pp in range(0,n_pokes):
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    LO_modes[poked_modes[pp]] = eunit/2*100
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    tmp_surface_poke = luvoir.zm.surface
    tmp_psf_poke, ref, = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    phases[pp] = np.reshape(tmp_surface_poke*luvoir.aper,[npup,npup])
    detectors[pp] = np.reshape(tmp_psf_poke,[nimg,nimg])
fig, axs = plt.subplots(2, n_pokes,figsize=(30,10))
for qq in range(0,n_pokes):
    phaseplot = axs[0,qq].imshow(phases[qq],cmap = 'hot')
    axs[0,qq].axes.get_xaxis().set_visible(False)
    axs[0,qq].axes.get_yaxis().set_visible(False)
    psfplot =  axs[1,qq].imshow(detectors[qq],norm=LogNorm(),vmin=10**(-11), vmax=10**(-5),cmap = 'hot')
    fig.colorbar(psfplot,ax = axs[1,qq],shrink=0.9)
    axs[1,qq].axes.get_xaxis().set_visible(False)
    axs[1,qq].axes.get_yaxis().set_visible(False)
plt.show()

file_name = figures_path + '/' + analysis_name + '_Poked_LO_camera' + '.pdf'
fig.savefig(file_name)

# LO modes to LOWFS
# Poked modes for figures are  [1,3,4,7,9] need to automate


LO_modes = np.zeros(n_LO)
MID_modes = np.zeros(n_MID)
HI_modes = np.zeros(n_HI)
DM_modes = np.zeros(n_DM)
luvoir.zm.actuators = LO_modes
luvoir.sm.actuators = MID_modes
luvoir.fm.actuators = HI_modes
luvoir.dm.actuators = DM_modes
zernike_ref = luvoir.prop_LOWFS()
poked_modes = [1,3,4,7,9]
n_pokes = len(poked_modes)
phases = np.zeros([len(poked_modes),npup,npup])
detectors = np.zeros([len(poked_modes),npup,npup])
for pp in range(0,n_pokes):
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    LO_modes[poked_modes[pp]] = eunit/2*100
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    tmp_surface_poke = luvoir.zm.surface
    tmp_psf_poke = luvoir.prop_LOWFS()
    phases[pp] = np.reshape(tmp_surface_poke*luvoir.aper,[npup,npup])
    detectors[pp] = np.reshape(tmp_psf_poke.power - zernike_ref.power,[npup,npup])
fig, axs = plt.subplots(2, n_pokes,figsize=(30,10))
for qq in range(0,n_pokes):
    phaseplot = axs[0,qq].imshow(phases[qq],cmap = 'hot')
    axs[0,qq].axes.get_xaxis().set_visible(False)
    axs[0,qq].axes.get_yaxis().set_visible(False)
    psfplot =  axs[1,qq].imshow(detectors[qq],cmap = 'hot')
#     fig.colorbar(psfplot,ax = axs[1,qq],shrink=0.9)
    axs[1,qq].axes.get_xaxis().set_visible(False)
    axs[1,qq].axes.get_yaxis().set_visible(False)
plt.show()

file_name = figures_path + '/' + analysis_name + '_Poked_LO_LOWFS' + '.pdf'
fig.savefig(file_name)

# MID modes to science camera

poked_modes = [0,1,3,4,5]
n_pokes = len(poked_modes)
phases = np.zeros([len(poked_modes),npup,npup])
detectors = np.zeros([len(poked_modes),nimg,nimg])
for pp in range(0,n_pokes):
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    MID_modes[poked_modes[pp]] = eunit/2*10
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    tmp_surface_poke = luvoir.sm.surface
    tmp_psf_poke, ref, = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    phases[pp] = np.reshape(tmp_surface_poke,[npup,npup])
    detectors[pp] = np.reshape(tmp_psf_poke,[nimg,nimg])
fig, axs = plt.subplots(2, n_pokes,figsize=(30,10))
for qq in range(0,n_pokes):
    phaseplot = axs[0,qq].imshow(phases[qq],cmap = 'hot')
    axs[0,qq].axes.get_xaxis().set_visible(False)
    axs[0,qq].axes.get_yaxis().set_visible(False)
    psfplot =  axs[1,qq].imshow(detectors[qq],norm=LogNorm(),vmin=10**(-11), vmax=10**(-5))
    fig.colorbar(psfplot,ax = axs[1,qq],shrink=0.9,cmap = 'hot')
    axs[1,qq].axes.get_xaxis().set_visible(False)
    axs[1,qq].axes.get_yaxis().set_visible(False)
plt.show()

file_name = figures_path + '/' + analysis_name + '_Poked_MID_camera' + '.pdf'
fig.savefig(file_name)



# MID modes to MIDWFS

LO_modes = np.zeros(n_LO)
MID_modes = np.zeros(n_MID)
HI_modes = np.zeros(n_HI)
DM_modes = np.zeros(n_DM)
luvoir.zm.actuators = LO_modes
luvoir.sm.actuators = MID_modes
luvoir.fm.actuators = HI_modes
luvoir.dm.actuators = DM_modes
zernike_ref = luvoir.prop_OBWFS()
poked_modes = [1,3,4,7,9]
n_pokes = len(poked_modes)
phases = np.zeros([len(poked_modes),npup,npup])
detectors = np.zeros([len(poked_modes),npup,npup])
for pp in range(0,n_pokes):
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    MID_modes[poked_modes[pp]] = eunit/2*100
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    tmp_surface_poke = luvoir.sm.surface
    tmp_psf_poke = luvoir.prop_OBWFS()
    phases[pp] = np.reshape(tmp_surface_poke*luvoir.aper,[npup,npup])
    detectors[pp] = np.reshape(tmp_psf_poke.power - zernike_ref.power,[npup,npup])
fig, axs = plt.subplots(2, n_pokes,figsize=(30,10))
for qq in range(0,n_pokes):
    phaseplot = axs[0,qq].imshow(phases[qq],cmap = 'hot')
    axs[0,qq].axes.get_xaxis().set_visible(False)
    axs[0,qq].axes.get_yaxis().set_visible(False)
    psfplot =  axs[1,qq].imshow(detectors[qq],cmap = 'hot')
#     fig.colorbar(psfplot,ax = axs[1,qq],shrink=0.9)
    axs[1,qq].axes.get_xaxis().set_visible(False)
    axs[1,qq].axes.get_yaxis().set_visible(False)
plt.show()

file_name = figures_path + '/' + analysis_name + '_Poked_MID_MIDWFS' + '.pdf'
fig.savefig(file_name)


# HI modes to science camera

poked_modes = [0,1,5,10,14]
n_pokes = len(poked_modes)
phases = np.zeros([len(poked_modes),npup,npup])
detectors = np.zeros([len(poked_modes),nimg,nimg])
for pp in range(0,n_pokes):
    LO_modes = np.zeros(n_LO)
    MID_modes = np.zeros(n_MID)
    HI_modes = np.zeros(n_HI)
    DM_modes = np.zeros(n_DM)
    HI_modes[poked_modes[pp]] = eunit/2*10
    luvoir.zm.actuators = LO_modes
    luvoir.sm.actuators = MID_modes
    luvoir.fm.actuators = HI_modes
    luvoir.dm.actuators = DM_modes
    tmp_surface_poke = luvoir.fm.surface
    tmp_psf_poke, ref, = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate=False)
    phases[pp] = np.reshape(tmp_surface_poke*luvoir.aper,[npup,npup])
    detectors[pp] = np.reshape(tmp_psf_poke,[nimg,nimg])
fig, axs = plt.subplots(2, n_pokes,figsize=(30,10))
for qq in range(0,n_pokes):
    phaseplot = axs[0,qq].imshow(phases[qq],cmap = 'hot')
    axs[0,qq].axes.get_xaxis().set_visible(False)
    axs[0,qq].axes.get_yaxis().set_visible(False)
    psfplot =  axs[1,qq].imshow(detectors[qq],norm=LogNorm(),vmin=10**(-11), vmax=10**(-5),cmap = 'hot')
    fig.colorbar(psfplot,ax = axs[1,qq],shrink=0.9)
    axs[1,qq].axes.get_xaxis().set_visible(False)
    axs[1,qq].axes.get_yaxis().set_visible(False)
plt.show()


file_name = figures_path + '/' + analysis_name + '_Poked_HI_Camera' + '.pdf'
fig.savefig(file_name)


#
# # if __name__ == '__main__':
#
#         # Pick the function of the telescope you want to run
#         #num_matrix_jwst()
#
#         coro_design = CONFIG_INI.get('LUVOIR', 'coronagraph_size')
#         test_e2e_modes(design=coro_design)
