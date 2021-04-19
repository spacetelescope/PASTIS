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
import astropy.constants as c
import astropy.table
import hcipy as hc

from config import CONFIG_INI
import util_pastis as util
from e2e_simulators.luvoir_imaging_onephot import LuvoirAPLC
import pretty_figures as pf
import exoscene.image
import exoscene.star
import exoscene.planet
from exoscene.planet import Planet


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

# PASTIS eigenvalues for all scales

evalsLO, evecsLO = np.linalg.eig(mat_LO)
sorted_evalsLO = np.sort(evalsLO)
sorted_indicesLO = np.argsort(evalsLO)
sorted_evecsLO = evecsLO[:, sorted_indicesLO]

evalsMID, evecsMID = np.linalg.eig(mat_MID)
sorted_evalsMID = np.sort(evalsMID)
sorted_indicesMID = np.argsort(evalsMID)
sorted_evecsMID = evecsMID[:, sorted_indicesMID]

evalsHI, evecsHI = np.linalg.eig(mat_HI)
sorted_evalsHI = np.sort(evalsHI)
sorted_indicesHI = np.argsort(evalsHI)
sorted_evecsHI = evecsHI[:, sorted_indicesHI]

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

# Getting the flux together
sptype = 'A0V' # Put this on config
Vmag = 0.0 # Put this in loop
minlam = 500*u.nanometer # Put this on config
maxlam = 600*u.nanometer # Put this on config
star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype = sptype,Vmag = Vmag,minlam = minlam.value,maxlam = maxlam.value)
Nph = star_flux.value*15**2*np.sum(luvoir.apodizer**2)/npup**2

# Reference fluxes for the WF sensors
# In particular we downsample everything

LO_modes = np.zeros(n_LO)
MID_modes = np.zeros(n_MID)
HI_modes = np.zeros(n_HI)
DM_modes = np.zeros(n_DM)
luvoir.zm.actuators = LO_modes
luvoir.sm.actuators = MID_modes
luvoir.fm.actuators = HI_modes
luvoir.dm.actuators = DM_modes

nonaberrated_coro_psf, refshit,inter_ref = luvoir.calc_psf(ref=True, display_intermediate=False, return_intermediate='efield')
Efield_ref = inter_ref['at_science_focus'].electric_field
zernike_ref = luvoir.prop_LOWFS()
zernike_ref_sub_real = hc.field.subsample_field(zernike_ref.real, z_pup_downsample, grid_zernike, statistic='mean')
zernike_ref_sub_imag = hc.field.subsample_field(zernike_ref.imag, z_pup_downsample, grid_zernike, statistic='mean')
Efield_ref_LOWFS = (zernike_ref_sub_real + 1j*zernike_ref_sub_imag)*z_pup_downsample

zernike_ref = luvoir.prop_OBWFS()
zernike_ref_sub_real = hc.field.subsample_field(zernike_ref.real, z_pup_downsample, grid_zernike, statistic='mean')
zernike_ref_sub_imag = hc.field.subsample_field(zernike_ref.imag, z_pup_downsample, grid_zernike, statistic='mean')
Efield_ref_OBWFS = (zernike_ref_sub_real + 1j*zernike_ref_sub_imag)*z_pup_downsample


nyquist_sampling = 2.

# Actual grid for LUVOIR images
grid_test = hc.make_focal_grid(
            luvoir.sampling,
            luvoir.imlamD,
            pupil_diameter=luvoir.diam,
            focal_length=1,
            reference_wavelength=luvoir.wvln,
        )

# Actual grid for LUVOIR images that are nyquist sampled
grid_det_subsample = hc.make_focal_grid(
            nyquist_sampling,
            np.floor(luvoir.imlamD),
            pupil_diameter=luvoir.diam,
            focal_length=1,
            reference_wavelength=luvoir.wvln,
        )
n_nyquist = np.int(np.sqrt(grid_det_subsample.x.shape[0]))

### Dark hole mask
dh_outer_nyquist = hc.circular_aperture(2 * luvoir.apod_dict[design]['owa'] * luvoir.lam_over_d)(grid_det_subsample)
dh_inner_nyquist = hc.circular_aperture(2 * luvoir.apod_dict[design]['iwa'] * luvoir.lam_over_d)(grid_det_subsample)
dh_mask_nyquist = (dh_outer_nyquist - dh_inner_nyquist).astype('bool')

dh_size = len(np.where(dh_mask != 0)[0])
dh_size_nyquist = len(np.where(dh_mask_nyquist != 0)[0])
dh_index = np.where(dh_mask != 0)[0]
dh_index_nyquist = np.where(dh_mask_nyquist != 0)[0]

### Rebinning everything to the right sampling
#
E0_LOWFS = np.zeros([N_pup_z*N_pup_z,1,2])
E0_LOWFS[:,0,0] = Efield_ref_LOWFS.real
E0_LOWFS[:,0,1] = Efield_ref_LOWFS.imag
E0_OBWFS = np.zeros([N_pup_z*N_pup_z,1,2])
E0_OBWFS[:,0,0] = Efield_ref_OBWFS.real
E0_OBWFS[:,0,1] = Efield_ref_OBWFS.imag
E0_coron = np.zeros([nimg*nimg,1,2])
E0_coron[:,0,0] = Efield_ref.real
E0_coron[:,0,1] = Efield_ref.imag
E0_coron_nyquist = np.zeros([n_nyquist*n_nyquist,1,2])
tmp0 = hc.interpolation.make_linear_interpolator_separated(Efield_ref, grid=grid_test)
Efield_ref_nyquist = (luvoir.sampling/nyquist_sampling)**2*tmp0(grid_det_subsample)
E0_coron_nyquist[:,0,0] = Efield_ref_nyquist.real
E0_coron_nyquist[:,0,1] = Efield_ref_nyquist.imag
E0_coron_DH = np.zeros([dh_size,1,2])
E0_coron_DH[:,0,0] = Efield_ref.real[dh_index]
E0_coron_DH[:,0,1] = Efield_ref.imag[dh_index]
E0_coron_DH_nyquist = np.zeros([dh_size_nyquist,1,2])
E0_coron_DH_nyquist[:,0,0] = Efield_ref_nyquist.real[dh_index_nyquist]
E0_coron_DH_nyquist[:,0,1] = Efield_ref_nyquist.real[dh_index_nyquist]

G_coron_LO_nyquist = np.zeros([n_nyquist*n_nyquist,2,n_LO-1])
for pp in range(1,n_LO):
    tmp0 = G_LO_real[pp] + 1j*G_LO_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_LO_nyquist[:,0,pp-1] = tmp2.real - Efield_ref_nyquist.real
    G_coron_LO_nyquist[:,1,pp-1] = tmp2.real - Efield_ref_nyquist.imag
G_coron_MID_nyquist= np.zeros([n_nyquist*n_nyquist,2,n_MID])
for pp in range(0,n_MID):
    tmp0 = G_MID_real[pp] + 1j*G_MID_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_MID_nyquist[:,0,pp] = tmp2.real - Efield_ref_nyquist.real
    G_coron_MID_nyquist[:,1,pp] = tmp2.real - Efield_ref_nyquist.imag
G_coron_HI_nyquist= np.zeros([n_nyquist*n_nyquist,2,n_HI])
for pp in range(0,n_HI):
    tmp0 = G_HI_real[pp] + 1j*G_HI_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_HI_nyquist[:,0,pp] = tmp2.real - Efield_ref_nyquist.real
    G_coron_HI_nyquist[:,1,pp] = tmp2.real - Efield_ref_nyquist.imag

G_coron_LO_DH = np.zeros([dh_size,2,n_LO-1])
for pp in range(1,n_LO):
    G_coron_LO_DH[:,0,pp-1] = G_LO_real[pp,dh_index] - Efield_ref.real[dh_index]
    G_coron_LO_DH[:,1,pp-1] = G_LO_imag[pp,dh_index] - Efield_ref.imag[dh_index]
G_coron_MID_DH= np.zeros([dh_size,2,n_MID])
for pp in range(0,n_MID):
    G_coron_MID_DH[:,0,pp] = G_MID_real[pp,dh_index] - Efield_ref.real[dh_index]
    G_coron_MID_DH[:,1,pp] = G_MID_imag[pp,dh_index] - Efield_ref.imag[dh_index]
G_coron_HI_DH= np.zeros([dh_size,2,n_HI])
for pp in range(0,n_HI):
    G_coron_HI_DH[:,0,pp] = G_HI_real[pp,dh_index] - Efield_ref.real[dh_index]
    G_coron_HI_DH[:,1,pp] = G_HI_imag[pp,dh_index] - Efield_ref.imag[dh_index]

G_coron_LO_DH_nyquist = np.zeros([dh_size_nyquist,2,n_LO-1])
for pp in range(1,n_LO):
    tmp0 = G_LO_real[pp] + 1j*G_LO_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_LO_DH_nyquist[:,0,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.real[dh_index_nyquist]
    G_coron_LO_DH_nyquist[:,1,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.imag[dh_index_nyquist]
G_coron_MID_DH_nyquist= np.zeros([dh_size_nyquist,2,n_MID])
for pp in range(0,n_MID):
    tmp0 = G_MID_real[pp] + 1j*G_MID_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_MID_DH_nyquist[:,0,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.real[dh_index_nyquist]
    G_coron_MID_DH_nyquist[:,1,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.imag[dh_index_nyquist]
G_coron_HI_DH_nyquist= np.zeros([dh_size_nyquist,2,n_HI])
for pp in range(0,n_HI):
    tmp0 = G_HI_real[pp] + 1j*G_HI_imag[pp]
    tmp1 = hc.interpolation.make_linear_interpolator_separated(tmp0, grid=grid_test)
    tmp2 = (luvoir.sampling/nyquist_sampling)**2*tmp1(grid_det_subsample)
    G_coron_HI_DH_nyquist[:,0,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.real[dh_index_nyquist]
    G_coron_HI_DH_nyquist[:,1,pp-1] = tmp2.real[dh_index_nyquist] - Efield_ref_nyquist.imag[dh_index_nyquist]

G_coron_LO = np.zeros([nimg*nimg,2,n_LO-1])
for pp in range(1,n_LO):
    G_coron_LO[:,0,pp-1] = G_LO_real[pp] - Efield_ref.real
    G_coron_LO[:,1,pp-1] = G_LO_imag[pp] - Efield_ref.imag
G_coron_MID= np.zeros([nimg*nimg,2,n_MID])
for pp in range(0,n_MID):
    G_coron_MID[:,0,pp] = G_MID_real[pp] - Efield_ref.real
    G_coron_MID[:,1,pp] = G_MID_imag[pp] - Efield_ref.imag
G_coron_HI= np.zeros([nimg*nimg,2,n_HI])
for pp in range(0,n_HI):
    G_coron_HI[:,0,pp] = G_HI_real[pp] - Efield_ref.real
    G_coron_HI[:,1,pp] = G_HI_imag[pp] - Efield_ref.imag


G_LOWFS = np.zeros([N_pup_z*N_pup_z,2,n_LO-1])
for pp in range(1,n_LO):
    G_LOWFS[:,0,pp-1] = G_LOWFS_real[pp]*z_pup_downsample - Efield_ref_LOWFS.real
    G_LOWFS[:,1,pp-1] = G_LOWFS_imag[pp]*z_pup_downsample - Efield_ref_LOWFS.imag
G_OBWFS= np.zeros([N_pup_z*N_pup_z,2,n_MID])
for pp in range(0,n_MID):
    G_OBWFS[:,0,pp] = G_OBWFS_real[pp]*z_pup_downsample - Efield_ref_OBWFS.real
    G_OBWFS[:,1,pp] = G_OBWFS_imag[pp]*z_pup_downsample - Efield_ref_OBWFS.imag


def req_closedloop_calc_recursive(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor, t_exp, flux, Q, Niter, dh_mask,
                                  norm):
    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    contrast_hist = np.zeros(Niter)
    intensity_WFS_hist = np.zeros(Niter)
    cal_I_hist = np.zeros(Niter)
    eps_hist = np.zeros([Niter, r])
    averaged_hist = np.zeros(Niter)
    contrasts = []
    for pp in range(Niter):
        eps = np.random.multivariate_normal(np.zeros(r), P + Q * t_exp).reshape((1, 1, r))  # random modes
        G_eps = np.sum(Gsensor * eps, axis=2).reshape((N, 1, 2 * c)) + E0sensor  # electric field
        G_eps_squared = np.sum(G_eps * G_eps, axis=2, keepdims=True)
        G_eps_G = np.matmul(G_eps, Gsensor)
        G_eps_G_scaled = G_eps_G / np.sqrt(G_eps_squared + Dsensor / flux / t_exp)  # trick to save RAM
        cal_I = 4 * flux * t_exp * np.einsum("ijk,ijl->kl", G_eps_G_scaled, G_eps_G_scaled)  # information matrix
        P = np.linalg.inv(np.linalg.inv(P + Q * t_exp / 2) + cal_I)
        #         P = np.linalg.inv(cal_I)

        # Coronagraph
        G_eps_coron = np.sum(Gcoro * eps, axis=2).reshape((N_img, 1, 2 * c)) + E0coro
        G_eps_coron_squared = np.sum(G_eps_coron * G_eps_coron, axis=2, keepdims=True)
        intensity = G_eps_coron_squared * flux * t_exp + Dcoro

        # Wavefront sensor
        intensity_WFS = G_eps_squared * flux * t_exp + Dsensor

        # Archive
        test_DH0 = intensity[:, 0, 0] * dh_mask
        test_DH = np.mean(test_DH0[np.where(test_DH0 != 0)])
        contrasts.append(test_DH / flux / t_exp / norm)
        intensity_WFS_hist[pp] = np.sum(intensity_WFS) / flux
        cal_I_hist[pp] = np.mean(cal_I) / flux
        eps_hist[pp] = eps
        averaged_hist[pp] = np.mean(contrasts)
        #         print("est. contrast", np.mean(contrasts))

        outputs = {'intensity_WFS_hist': intensity_WFS_hist,
                   'cal_I_hist': cal_I_hist,
                   'eps_hist': eps_hist,
                   'averaged_hist': averaged_hist,
                   'contrasts': contrasts}
    return outputs


def req_closedloop_calc_batch(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor, t_exp, flux, Q, Niter, dh_mask, norm):
    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    contrast_hist = np.zeros(Niter)
    intensity_WFS_hist = np.zeros(Niter)
    cal_I_hist = np.zeros(Niter)
    eps_hist = np.zeros([Niter, r])
    averaged_hist = np.zeros(Niter)
    contrasts = []
    for pp in range(Niter):
        eps = np.random.multivariate_normal(np.zeros(r), P + Q * t_exp).reshape((1, 1, r))  # random modes
        G_eps = np.sum(Gsensor * eps, axis=2).reshape((N, 1, 2 * c)) + E0sensor  # electric field
        G_eps_squared = np.sum(G_eps * G_eps, axis=2, keepdims=True)
        G_eps_G = np.matmul(G_eps, Gsensor)
        G_eps_G_scaled = G_eps_G / np.sqrt(G_eps_squared + Dsensor / flux / t_exp)  # trick to save RAM
        cal_I = 4 * flux * t_exp * np.einsum("ijk,ijl->kl", G_eps_G_scaled, G_eps_G_scaled)  # information matrix
        #         P = np.linalg.inv(np.linalg.inv(P+Q*t_exp/2) + cal_I)
        P = np.linalg.pinv(cal_I)

        # Coronagraph
        G_eps_coron = np.sum(Gcoro * eps, axis=2).reshape((N_img, 1, 2 * c)) + E0coro
        G_eps_coron_squared = np.sum(G_eps_coron * G_eps_coron, axis=2, keepdims=True)
        intensity = G_eps_coron_squared * flux * t_exp + Dcoro

        # Wavefront sensor
        intensity_WFS = G_eps_squared * flux * t_exp + Dsensor

        # Archive
        test_DH0 = intensity[:, 0, 0] * dh_mask
        test_DH = np.mean(test_DH0[np.where(test_DH0 != 0)])
        contrasts.append(test_DH / flux / t_exp / norm)
        intensity_WFS_hist[pp] = np.sum(intensity_WFS) / flux
        cal_I_hist[pp] = np.mean(cal_I) / flux
        eps_hist[pp] = eps
        averaged_hist[pp] = np.mean(contrasts)
    #         print("est. contrast", np.mean(contrasts))
    #         print("est. contrast", np.mean(contrasts))

    outputs = {'intensity_WFS_hist': intensity_WFS_hist,
               'cal_I_hist': cal_I_hist,
               'eps_hist': eps_hist,
               'averaged_hist': averaged_hist,
               'contrasts': contrasts}

    return outputs


flux = Nph
QLO = np.diag(np.asarray(mu_mapLO**2))
QMID = np.diag(np.asarray(mu_mapMID**2))
QHI = np.diag(np.asarray(mu_mapHI**2))

# Running a bunch of tests for time series

Ntimes = 20
TimeMinus = -2
TimePlus = 3.5
Nwavescale = 20
WaveScaleMinus = -2
WaveScalePlus = 2
Nflux = 10
fluxPlus = 9
fluxMinus = 0

timeVec = np.logspace(TimeMinus,TimePlus,Ntimes)
WaveVec = np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale)
fluxVec = np.linspace(fluxMinus,fluxPlus,Nflux)
wavescale = np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale)[6]

niter = 10

print('Mid modes with recursive OBWFS')

import time
timer1 = time.time()


res = np.zeros([Ntimes,Nwavescale,Nflux,1])
pp = 0
for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
    qq = 0
    print(tscale)
    for wavescale in np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale):
        rr = 0
        for StarMag in np.linspace(fluxMinus,fluxPlus,Nflux):
            Starfactor = 10**(-StarMag/2.5)
            tmp0 = req_closedloop_calc_recursive(G_coron_MID,G_OBWFS,E0_coron,E0_OBWFS,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            res[pp,qq,rr] =  np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
            rr = rr + 1
        qq = qq + 1
    pp = pp + 1


res_line = np.reshape(res,[Ntimes*Nwavescale*Nflux])
text_files_name = overall_dir + '/MID_OBWFS_Recursive.csv'
np.savetxt(text_files_name, res_line, delimiter=",")

timer2 = time.time()
print(timer2 - timer1)


print('Mid modes with batch OBWFS')

import time
timer1 = time.time()


res = np.zeros([Ntimes,Nwavescale,Nflux,1])
pp = 0
for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
    qq = 0
    print(tscale)
    for wavescale in np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale):
        rr = 0
        for StarMag in np.linspace(fluxMinus,fluxPlus,Nflux):
            Starfactor = 10**(-StarMag/2.5)
            tmp0 = req_closedloop_calc_batch(G_coron_MID,G_OBWFS,E0_coron,E0_OBWFS,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            res[pp,qq,rr] =  np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
            rr = rr + 1
        qq = qq + 1
    pp = pp + 1


res_line = np.reshape(res,[Ntimes*Nwavescale*Nflux])
text_files_name = overall_dir + '/MID_OBWFS_Batch.csv'
np.savetxt(text_files_name, res_line, delimiter=",")

timer2 = time.time()
print(timer2 - timer1)


print('Mid modes with recursive binned science images')

import time
timer1 = time.time()


res = np.zeros([Ntimes,Nwavescale,Nflux,1])
pp = 0
for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
    qq = 0
    print(tscale)
    for wavescale in np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale):
        rr = 0
        for StarMag in np.linspace(fluxMinus,fluxPlus,Nflux):
            Starfactor = 10**(-StarMag/2.5)
            tmp0 = req_closedloop_calc_recursive(G_coron_MID,G_coron_MID_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            res[pp,qq,rr] =  np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
            rr = rr + 1
        qq = qq + 1
    pp = pp + 1


res_line = np.reshape(res,[Ntimes*Nwavescale*Nflux])
text_files_name = overall_dir + '/MID_ScienceBinned_Recursive.csv'
np.savetxt(text_files_name, res_line, delimiter=",")

timer2 = time.time()
print(timer2 - timer1)

print('Mid modes with batch binned science images')

import time
timer1 = time.time()


res = np.zeros([Ntimes,Nwavescale,Nflux,1])
pp = 0
for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
    qq = 0
    print(tscale)
    for wavescale in np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale):
        rr = 0
        for StarMag in np.linspace(fluxMinus,fluxPlus,Nflux):
            Starfactor = 10**(-StarMag/2.5)
            tmp0 = req_closedloop_calc_recursive(G_coron_MID,G_coron_MID_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            res[pp,qq,rr] =  np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
            rr = rr + 1
        qq = qq + 1
    pp = pp + 1


res_line = np.reshape(res,[Ntimes*Nwavescale*Nflux])
text_files_name = overall_dir + '/MID_ScienceBinned_Batch.csv'
np.savetxt(text_files_name, res_line, delimiter=",")

timer2 = time.time()
print(timer2 - timer1)

# res = np.zeros([Ntimes,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     niter = np.int(np.max([100/tscale,1000]))
#     tmp0 = req_closedloop_calc_recursive(G_coron_LO,G_LOWFS,E0_coron,E0_LOWFS,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1





# res2 = np.zeros([Ntimes,1])
# pp = 0
# niter = 1000
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     tmp0 = req_closedloop_calc_batch(G_coron_LO,G_LOWFS,E0_coron,E0_LOWFS,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res2[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
# text_files_name = overall_dir + '/LO_LOWFS_Recursive.csv'
# np.savetxt(text_files_name, res2, delimiter=",")

# res = np.zeros([Ntimes,Nwavescale,Nflux,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     qq = 0
#     print(tscale)
#     for wavescale in np.logspace(WaveScaleMinus,WaveScalePlus,Nwavescale):
#         rr = 0
#         for StarMag in np.linspace(fluxMinus,fluxPlus,Nflux):
#             Starfactor = 10**(-StarMag/2.5)
#             tmp0 = req_closedloop_calc_batch(G_coron_MID,G_OBWFS,E0_coron,E0_OBWFS,0,0,tscale,flux*Starfactor,wavescale**2*QMID,NiterBatch,dh_mask,norm)
#             tmp1 = tmp0['averaged_hist']
#             n_tmp1 = len(tmp1)
#             res[pp,qq,rr] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1])
#             rr = rr + 1
#         qq = qq + 1
#     pp = pp + 1
# res_line = np.reshape(res,[Ntimes*Nwavescale*Nflux])
# np.savetxt("MIDbatchOBWFS.csv", res_line, delimiter=",")

#
# res3 = np.zeros([Ntimes,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     niter = np.int(np.max([100/tscale,1000]))
#     tmp0 = req_closedloop_calc_recursive(G_coron_LO,G_coron_LO_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res3[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
# res4 = np.zeros([Ntimes,1])
# pp = 0
# niter = 1000
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     tmp0 = req_closedloop_calc_batch(G_coron_LO,G_coron_LO_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res4[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
# res5 = np.zeros([Ntimes,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     niter = np.int(np.max([100/tscale,1000]))
#     tmp0 = req_closedloop_calc_recursive(G_coron_LO,G_coron_LO_DH,E0_coron,E0_coron_DH,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res5[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
# res6 = np.zeros([Ntimes,1])
# pp = 0
# niter = 1000
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     tmp0 = req_closedloop_calc_batch(G_coron_LO,G_coron_LO_DH,E0_coron,E0_coron_DH,0,0,tscale,flux*Starfactor,wavescale**2*QLO,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res6[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
# timeVec = np.logspace(TimeMinus,TimePlus,Ntimes)
# fig = plt.figure(figsize=(16, 10))
# plt.plot(timeVec,res)
# plt.plot(timeVec,res2)
# plt.plot(timeVec,res3)
# plt.plot(timeVec,res4)
# plt.plot(timeVec,res5)
# plt.plot(timeVec,res6)
# plt.semilogx()
# plt.semilogy()
# file_name = figures_path + '/test_fion1' + '.pdf'
# fig.savefig(file_name)
#
#
# res = np.zeros([Ntimes,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     niter = np.int(np.max([100/tscale,1000]))
#     tmp0 = req_closedloop_calc_recursive(G_coron_MID,G_OBWFS,E0_coron,E0_OBWFS,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
#
#
# res2 = np.zeros([Ntimes,1])
# pp = 0
# niter = 1000
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     tmp0 = req_closedloop_calc_batch(G_coron_MID,G_OBWFS,E0_coron,E0_OBWFS,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res2[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
# res3 = np.zeros([Ntimes,1])
# pp = 0
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     niter = np.int(np.max([100/tscale,1000]))
#     tmp0 = req_closedloop_calc_recursive(G_coron_MID,G_coron_MID_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res3[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
# res4 = np.zeros([Ntimes,1])
# pp = 0
# niter = 1000
# for tscale in np.logspace(TimeMinus,TimePlus,Ntimes):
#     Starfactor = 10**(-10/2.5)
#     tmp0 = req_closedloop_calc_batch(G_coron_MID,G_coron_MID_DH_nyquist,E0_coron,E0_coron_DH_nyquist,0,0,tscale,flux*Starfactor,wavescale**2*QMID,niter,dh_mask,norm)
#     tmp1 = tmp0['averaged_hist']
#     n_tmp1 = len(tmp1)
#     res4[pp] = np.mean(tmp1[np.int(n_tmp1/2):n_tmp1]) - contrast_floor
#     pp = pp + 1
#
#
#
# timeVec = np.logspace(TimeMinus,TimePlus,Ntimes)
# fig = plt.figure(figsize=(16, 10))
# # plt.plot(timeVec,res)
# # plt.plot(timeVec,res2)
# plt.plot(timeVec,res3)
# plt.plot(timeVec,res4)
# plt.plot(timeVec,res5)
# plt.plot(timeVec,res6)
# plt.semilogx()
# plt.semilogy()
# file_name = figures_path + '/test_fion3' + '.pdf'
# fig.savefig(file_name)
#
#text_files_path = overall_dir
#
#
# # tmp = hc.interpolation.make_linear_interpolator_separated(ref,grid=grid_test)
# # ref_nyquist = (luvoir.sampling/nyquist_sampling)**2*tmp(grid_det_subsample)
# # norm_nyquist = np.max(ref_nyquist)
# # tmp = hc.interpolation.make_linear_interpolator_separated(unaberrated_coro_psf, grid=grid_test)
# # test = (luvoir.sampling/nyquist_sampling)**2*tmp(grid_det_subsample)
#
#


