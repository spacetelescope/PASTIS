import os
import hcipy
from astropy.io import fits
import numpy as np
import time
import exoscene.star
import astropy.units as u
import matplotlib.pyplot as plt
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
import pastis.util as util
from pastis.temporal_analysis.close_loop_analysis import req_closedloop_calc_batch


def matrix_subsample(matrix, n, m):
    # return a matrix of shape (n,m)
    arr_sum = []
    l = matrix.shape[0]//n  # block length
    b = matrix.shape[1]//m  # block breadth
    for i in range(n):
        for j in range(m):
            sum_pixels = np.sum(matrix[i*l: (i+1)*l, j*b: (j+1)*b])
            arr_sum.append(sum_pixels)
    data_reduced = np.reshape(np.array(arr_sum), (n, m))
    return data_reduced


def matrix_subsample_fast(matrix, n, m):
    l = matrix.shape[0] // n  # block length
    b = matrix.shape[1] // m  # block breadth
    new_shape = (n, l, m, b)
    reshaped_array = matrix.reshape(new_shape)
    data_reduced = np.sum(reshaped_array, axis=(1, 3))
    return data_reduced


def calculate_sensitivity_matrices(e0_coron, e0_obwfs, efield_coron_real,
                                   efield_coron_imag, efield_obwfs_real,
                                   efield_obwfs_imag, subsample_factor):

    total_sci_pix = np.square(e0_coron.shape[1])
    total_pupil_pix = np.square(e0_obwfs.shape[1])

    ref_coron_real = np.reshape(e0_coron[0], total_sci_pix)
    ref_coron_imag = np.reshape(e0_coron[1], total_sci_pix)

    ref_obwfs_real = np.reshape(e0_obwfs[0], total_pupil_pix)
    ref_obwfs_imag = np.reshape(e0_obwfs[1], total_pupil_pix)

    ref_coron = np.zeros([total_sci_pix, 1, 2])
    ref_coron[:, 0, 0] = ref_coron_real
    ref_coron[:, 0, 1] = ref_coron_imag

    n_sub_pix = int(np.sqrt(total_pupil_pix) // subsample_factor)
    ref_wfs_real_sub = np.reshape(matrix_subsample(e0_obwfs[0], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_imag_sub = np.reshape(matrix_subsample(e0_obwfs[1], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_sub = (ref_wfs_real_sub + 1j * ref_wfs_imag_sub) / subsample_factor
    # subsample_factor**2 is multiplied here to preserve total intensity, same goes for g_obwfs_downsampled

    ref_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 1, 2])
    ref_obwfs_downsampled[:, 0, 0] = ref_wfs_sub.real
    ref_obwfs_downsampled[:, 0, 1] = ref_wfs_sub.imag

    num_all_modes = efield_coron_real.shape[0]
    g_coron = np.zeros([total_sci_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_coron[:, 0, i] = np.reshape(efield_coron_real[i], total_sci_pix) - ref_coron_real
        g_coron[:, 1, i] = np.reshape(efield_coron_imag[i], total_sci_pix) - ref_coron_imag

    g_obwfs = np.zeros([total_pupil_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_obwfs[:, 0, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_real
        g_obwfs[:, 1, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_imag

    g_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 2, num_all_modes])
    for i in range(num_all_modes):
        efields_per_mode_wfs_real_sub = np.reshape(matrix_subsample(efield_obwfs_real[i],
                                                                    n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        efields_per_mode_wfs_imag_sub = np.reshape(matrix_subsample(efield_obwfs_imag[i],
                                                                    n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        g_obwfs_downsampled[:, 0, i] = efields_per_mode_wfs_real_sub - ref_wfs_sub.real
        g_obwfs_downsampled[:, 1, i] = efields_per_mode_wfs_imag_sub - ref_wfs_sub.imag

    matrix = {"ref_image_plane": ref_coron,
              "ref_wfs_plane": ref_obwfs_downsampled,
              "senitivity_image_plane": g_coron,
              "sensitvity_wfs_plane": g_obwfs_downsampled}

    return matrix

#def sort_tolerance_maps():



# Create necessary directories if they don't exist yet
data_dir = CONFIG_PASTIS.get('local', 'local_data_path')

# Instantiate the LUVOIR telescope
optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
coronagraph_design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
tel = LuvoirA_APLC(optics_input, coronagraph_design, sampling)
nm_aber = CONFIG_PASTIS.getfloat('LUVOIR', 'calibration_aberration') * 1e-9

# Create harris deformable mirror
pad_orientation = np.pi/2*np.ones(tel.nseg)
filepath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')
tel.create_segmented_harris_mirror(filepath, pad_orientation, thermal=True, mechanical=False, other=False)
tel.harris_sm

# get number of poking modes
num_actuators = tel.harris_sm.num_actuators
num_modes = 5

# calculate dark hole contrast
tel.harris_sm.flatten()
unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
norm = np.max(ref)
dh_intensity = (unaberrated_coro_psf / norm) * tel.dh_mask
contrast_floor = np.mean(dh_intensity[np.where(tel.dh_mask != 0)])
print(f'contrast floor: {contrast_floor}')


# Calculate pastis matrix
DM = 'harris_seg_mirror'
DM_SPEC = (filepath, pad_orientation, True, False, False)


run_matrix = MatrixEfieldLuvoirA(which_dm=DM, dm_spec=DM_SPEC, design='small',
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
                                 norm_one_photon=True)
run_matrix.calc()
data_dir = run_matrix.overall_dir
print(f'Matrix and Efields saved to {data_dir}.')

pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))

# Define target contrast
c_target_log = -11
c_target = 1*10**(c_target_log)
mu_map_harris = np.sqrt(((c_target) / (num_actuators)) / (np.diag(pastis_matrix)))
np.savetxt(os.path.join(data_dir, 'mu_map_harris_%s.csv' % c_target), mu_map_harris, delimiter=',')


# Temporal Analysis
#data_dir = '/Users/asahoo/Desktop/data_repos/harris_data/2022-09-19T16-28-34_luvoir'
efield_science_real = fits.getdata(os.path.join(data_dir,'matrix_numerical', 'efield_coron_real.fits'))
efield_science_imag = fits.getdata(os.path.join(data_dir,'matrix_numerical','efield_coron_imag.fits'))
efield_wfs_real = fits.getdata(os.path.join(data_dir,'matrix_numerical', 'efield_obwfs_real.fits'))
efield_wfs_imag = fits.getdata(os.path.join(data_dir,'matrix_numerical','efield_obwfs_imag.fits'))
ref_coron = fits.getdata(os.path.join(data_dir, 'ref_e0_coron.fits'))
ref_obwfs = fits.getdata(os.path.join(data_dir, 'ref_e0_wfs.fits'))

# Get the total pupil and imaging camera pixels
sensitivity_matrices = calculate_sensitivity_matrices(ref_coron, ref_obwfs, efield_science_real, efield_science_imag,
                                                      efield_wfs_real, efield_wfs_imag, subsample_factor=8)
g_coron = sensitivity_matrices['senitivity_image_plane']
g_wfs = sensitivity_matrices['sensitvity_wfs_plane']
e0_coron = sensitivity_matrices['ref_image_plane']
e0_wfs = sensitivity_matrices['ref_wfs_plane']

npup = int(np.sqrt(tel.pupil_grid.x.shape[0]))
sptype = 'A0V'
Vmag = 5.0
minlam = 500 * u.nanometer
maxlam = 600 * u.nanometer
dark_current = 0
CIC = 0
star_flux = exoscene.star.bpgs_spectype_to_photonrate(spectype=sptype, Vmag=Vmag, minlam=minlam.value, maxlam=maxlam.value)
Nph = star_flux.value*15**2*np.sum(tel.apodizer**2) / npup**2
flux = Nph
Qharris = np.diag(np.asarray(mu_map_harris**2))
detector_noise = 0.0
niter = 10
TimeMinus = -2
TimePlus = 5.5
Ntimes = 20
Nwavescale = 8
Nflux = 3
res = np.zeros([Ntimes, Nwavescale, Nflux, 1])
result_wf_test =[]
norm = 0.010242195657579547
for wavescale in range (1, 15, 2):
    print('recurssive close loop batch estimation and wavescale %f'% wavescale)
    niter = 10
    timer1 = time.time()
    StarMag = 0.0
    for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
        Starfactor = 10**(-StarMag/2.5)
        print(tscale)
        tmp0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, tscale, flux*Starfactor, 0.0001*wavescale**2*Qharris,
                                         niter, tel.dh_mask, norm)
        tmp1 = tmp0['averaged_hist']
        n_tmp1 = len(tmp1)
        result_wf_test.append(tmp1[n_tmp1-1])

delta_wf = []
for wavescale in range(1, 15, 2):
    wf = np.sqrt(np.mean(np.diag(0.0001*wavescale**2*Qharris)))*1e3
    delta_wf.append(wf)

texp = np.logspace(TimeMinus, TimePlus, Ntimes)
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}
contrast_floor = 4.237636070056418e-11
result_wf_test = np.asarray(result_wf_test)
plt.figure(figsize=(15, 10))
plt.title('Target contrast = %s, Vmag= %s' % (c_target, Vmag), fontdict=font)
plt.plot(texp, result_wf_test[0:20] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[0]))
plt.plot(texp, result_wf_test[20:40] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[1]))
plt.plot(texp, result_wf_test[40:60] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[2]))
plt.plot(texp, result_wf_test[60:80] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[3]))
plt.plot(texp, result_wf_test[80:100] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[4]))
plt.plot(texp, result_wf_test[100:120] - contrast_floor, label=r'$\Delta_{wf}= %.2f\ pm/s$' % (delta_wf[5]))
plt.plot(texp, result_wf_test[120:140] - contrast_floor, label=r'$\Delta_{wf}= % .2f\ pm/s$' % (delta_wf[6]))
plt.xlabel("$t_{WFS}$ in secs", fontsize=20)
plt.ylabel("$\Delta$ contrast", fontsize=20)
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='upper center', fontsize=20)
plt.tick_params(top=True, bottom=True, left=True,
                    right=True, labelleft=True, labelbottom=True,
                    labelsize=20)
plt.tick_params(axis='both', which='major', length=10, width=2)
plt.tick_params(axis='both', which='minor', length=6, width=2)
plt.grid()
plt.savefig(os.path.join(data_dir, 'cont_wf_%s.png' % c_target))

# Sort to individual modes
harris_coeffs_numaps = np.zeros([num_modes, num_actuators])
for qq in range(num_modes):
    harris_coeffs_tmp = np.zeros([num_actuators])
    for kk in range(tel.nseg):
        harris_coeffs_tmp[qq+(kk)*num_modes] = mu_map_harris[qq+(kk)*num_modes] # arranged per modal basis
    harris_coeffs_numaps[qq] = harris_coeffs_tmp # arranged into 5 groups of 600 elements and in units of nm

nu_maps = []
for qq in range(num_modes):
    harris_coeffs = harris_coeffs_numaps[qq] # in units of nm
    tel.harris_sm.actuators = harris_coeffs*nm_aber/2 # in units of m
    nu_maps.append(tel.harris_sm.surface) # in units of m, each nu_map is now of the order of 1e-9 m

coeffs_table = np.zeros([num_modes, tel.nseg])
for qq in range(num_modes):
    for kk in range(tel.nseg):
        coeffs_table[qq, kk] = mu_map_harris[qq + (kk) * num_modes]

# Final Individual Tolerance allocation across 5 modes in units of pm
opt_wavescale = 13 # This is wavescale value corresponding to lowest contrast
Q_total = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(mu_map_harris**2))))
Q_1 = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(coeffs_table[0]**2))))
Q_2 = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(coeffs_table[1]**2))))
Q_3 = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(coeffs_table[2]**2))))
Q_4 = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(coeffs_table[3]**2))))
Q_5 = 1e3*np.sqrt(0.0001*opt_wavescale**2*np.sqrt(np.mean(np.square(coeffs_table[4]**2))))
print(Q_1, Q_2, Q_3, Q_4, Q_5)
print(Q_total, np.sqrt((Q_1**2 + Q_2**2 + Q_3**2 + Q_4**2 + Q_5**2)/5))


# check temporal maps for individual modes
q0 = np.diag(0.0001*opt_wavescale**2*Qharris)
q1 = 0.0001*opt_wavescale**2*(coeffs_table[0]**2)
q2 = 0.0001*opt_wavescale**2*(coeffs_table[1]**2)
q3 = 0.0001*opt_wavescale**2*(coeffs_table[2]**2)
q4 = 0.0001*opt_wavescale**2*(coeffs_table[3]**2)
q5 = 0.0001*opt_wavescale**2*(coeffs_table[4]**2)

tscale = 1000
print(np.sqrt(np.mean(q0))*1e3*tscale, '\n', np.sqrt(np.mean(q1))*1e3*tscale, '\n', np.sqrt(np.mean(q2))*1e3*tscale, '\n',
      np.sqrt(np.mean(q3))*1e3*tscale, '\n', np.sqrt(np.mean(q4))*1e3*tscale, '\n', np.sqrt(np.mean(q5))*1e3*tscale, '\n')

Qharris = np.diag(np.asarray(mu_map_harris**2)) # 19.4/1e3
Qharris0 = np.diag(np.asarray(harris_coeffs_numaps[0]**2))
Qharris1 = np.diag(np.asarray(harris_coeffs_numaps[1]**2))
Qharris2 = np.diag(np.asarray(harris_coeffs_numaps[2]**2))
Qharris3 = np.diag(np.asarray(harris_coeffs_numaps[3]**2))
Qharris4 = np.diag(np.asarray(harris_coeffs_numaps[4]**2))

opt_tscale = 30
c_contrast0 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris,
                                         niter, tel.dh_mask, norm)

c_contrast1 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris0,
                                         niter, tel.dh_mask, norm)

c_contrast2 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris1,
                                         niter, tel.dh_mask, norm)

c_contrast3 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris2,
                                         niter, tel.dh_mask, norm)

c_contrast4 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris3,
                                         niter, tel.dh_mask, norm)

c_contrast5 = req_closedloop_calc_batch(g_coron, g_wfs, e0_coron, e0_wfs, detector_noise,
                                         detector_noise, opt_tscale, flux*Starfactor, 0.0001*opt_wavescale**2*Qharris4,
                                         niter, tel.dh_mask, norm)

result_c0 = []
c0 = c_contrast0['averaged_hist']
n_tmp1 = len(c0)
result_c0.append(c0[n_tmp1-1])
print(result_c0[0]-contrast_floor)

result_c1 = []
c1 = c_contrast1['averaged_hist']
n_tmp1 = len(c1)
result_c1.append(c1[n_tmp1-1])
print(result_c1[0]-contrast_floor)

result_c2 = []
c2 = c_contrast2['averaged_hist']
n_tmp1 = len(c2)
result_c2.append(c2[n_tmp1-1])
print(result_c2[0]-contrast_floor)

result_c3 = []
c3 = c_contrast3['averaged_hist']
n_tmp1 = len(c3)
result_c3.append(c3[n_tmp1-1])
print(result_c3[0]-contrast_floor)

result_c4 = []
c4 = c_contrast4['averaged_hist']
n_tmp1 = len(c4)
result_c4.append(c4[n_tmp1-1])
print(result_c4[0]-contrast_floor)

result_c5 = []
c5 = c_contrast5['averaged_hist']
n_tmp1 = len(c5)
result_c5.append(c5[n_tmp1-1])
print(result_c5[0]-contrast_floor)
