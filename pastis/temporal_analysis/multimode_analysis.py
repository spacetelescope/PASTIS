import os
import hcipy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pastis.util as util
from pastis.simulators.scda_telescopes import HexRingAPLC
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.util import find_repo_location
from pastis.config import CONFIG_PASTIS

# Define input parameters
optics_dir = os.path.join(find_repo_location(), 'data', 'SCDA')
sampling = 4
num_rings = 2
robust = 4
nm_aber = CONFIG_PASTIS.getfloat('HexRingTelescope', 'calibration_aberration') * 1e-9
filepath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')

# # Create necessary directories if they don't exist yet
# data_dir = CONFIG_PASTIS.get('local', 'local_data_path')
# repo_dir = os.path.join(util.find_repo_location(package='pastis'))
# overall_dir = util.create_data_path(data_dir, telescope='hexaplc_design_%s' % num_rings)
# resDir = os.path.join(overall_dir, 'matrix_numerical')
# os.makedirs(resDir, exist_ok=True)
# print('Data saved to:', resDir)

# Instantiate the HeXRingAPLC telescope
tel = HexRingAPLC(optics_dir, num_rings, sampling, robustness_px=robust)
coro, ref, inter = tel.calc_psf(ref=True, display_intermediate=True,
                                return_intermediate='intensity', norm_one_photon=True)
#
# # Create harris deformable mirror
# pad_orientation = np.pi/2*np.ones(tel.nseg)
# tel.create_segmented_harris_mirror(filepath, pad_orientation, thermal=True, mechanical=False, other=False)
# tel.harris_sm
#
# # get number of poking modes
# num_actuators = tel.harris_sm.num_actuators
# num_modes = 5
#
# # calculate dark hole contrast
# tel.harris_sm.flatten()
# unaberrated_coro_psf, ref = tel.calc_psf(ref=True, display_intermediate=False, norm_one_photon=True)
# norm = np.max(ref)
#
# dh_intensity = (unaberrated_coro_psf / norm) * tel.dh_mask
# contrast_floor = np.mean(dh_intensity[np.where(tel.dh_mask != 0)])
# print(f'contrast floor: {contrast_floor}')
#
#
# # Calculate pastis matrix
# DM = 'harris_seg_mirror'
# DM_SPEC = (filepath, pad_orientation, True, False, False)
# run_matrix = MatrixEfieldHex(which_dm=DM, dm_spec=DM_SPEC, num_rings=num_rings,
#                                      calc_science=True, calc_wfs=False,
#                                      initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
#                                      norm_one_photon=True)
# run_matrix.calc()
# data_dir = run_matrix.overall_dir
# print(f'Matrix and Efields saved to {data_dir}.')
#
# pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
#
#
# c_target_log = -8
# c_target = 10**(c_target_log)
# mu_map_harris = np.sqrt(((c_target) / (num_actuators)) / (np.diag(pastis_matrix)))
#
# #plot and save mu_map_harris
# plt.figure(figsize=(20,10))
# plt.title("Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$"%c_target_log, fontsize=30)
# plt.ylabel("Weight per mode (in units of nm)",fontsize =20)
# plt.xlabel("Mode index", fontsize=20)
# plt.tick_params(top=True, bottom=True, left=True, right=True,labelleft=True, labelbottom=True, labelsize=20)
# plt.plot(mu_map_harris)
# plt.tight_layout()
# plt.savefig(os.path.join(data_dir, 'mu_map_harris_%s.png'%c_target))
#
# ### plot mu_maps
# harris_coeffs_numaps = np.zeros([num_modes, num_actuators])
#
# for qq in range(num_modes):
#     harris_coeffs_tmp = np.zeros([num_actuators])
#     for kk in range(tel.nseg):
#         harris_coeffs_tmp[qq+(kk)*num_modes] = mu_map_harris[qq+(kk)*num_modes] #arranged per modal basis
#     harris_coeffs_numaps[qq] = harris_coeffs_tmp #arragned into 5 groups of 35 elements and in units of nm
#
# nu_maps = []
# for qq in range(num_modes):
#     harris_coeffs = harris_coeffs_numaps[qq] #in units of nm
#     tel.harris_sm.actuators = harris_coeffs*nm_aber/2 # in units of m
#     nu_maps.append(tel.harris_sm.surface) #in units of m, each nu_map is now of the order of 1e-9 m
#
# plt.figure(figsize=(20, 10))
# ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
# plt.title("Segment Level 1mK Faceplates Silvered", fontsize=15)
# hcipy.imshow_field((nu_maps[0])*1e9, cmap='RdBu')  # nu_map is already in 1e-9 m
# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("$nm$", fontsize=15)
#
# ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
# plt.title("Segment Level 1mK bulk", fontsize=15)
# hcipy.imshow_field((nu_maps[1])*1e9, cmap='RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("$nm$", fontsize=15)
#
# ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
# plt.title("Segment Level 1mK gradiant radial", fontsize=15)
# hcipy.imshow_field((nu_maps[2])*1e9, cmap='RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("$nm$", fontsize=15)
#
# ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
# plt.title("Segment Level 1mK gradient X lateral", fontsize=15)
# hcipy.imshow_field((nu_maps[3])*1e9, cmap='RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("$nm$", fontsize=15)
#
# ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
# plt.title("Segment Level 1mK gradient Z axial", fontsize=15)
# hcipy.imshow_field((nu_maps[4])*1e9, cmap='RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("$nm$", fontsize=15)
# plt.tight_layout()
# plt.savefig(os.path.join(data_dir, 'stat_mu_maps_nm_%s.png' % c_target))
#
# harris_coeffs_table = np.zeros([num_modes, tel.nseg])
# for qq in range(num_modes):
#     for kk in range(tel.nseg):
#         harris_coeffs_table[qq,kk] = mu_map_harris[qq+(kk)*num_modes] #numpy ndarray 120 # in units of nm
#
# # Faceplate silvered
# tel2 = HexRingAPLC(optics_dir, num_rings, sampling)
# tel2.create_segmented_mirror(1)
# tel2.sm.actuators =harris_coeffs_table[0]
#
# # Bulk
# tel3 = HexRingAPLC(optics_dir, num_rings, sampling)
# tel3.create_segmented_mirror(1)
# tel3.sm.actuators = harris_coeffs_table[1]
#
# # Gradient Radial
# tel4 = HexRingAPLC(optics_dir, num_rings, sampling)
# tel4.create_segmented_mirror(1)
# tel4.sm.actuators = harris_coeffs_table[2]
#
# # Gradient X Lateral
# tel5 = HexRingAPLC(optics_dir, num_rings, sampling)
# tel5.create_segmented_mirror(1)
# tel5.sm.actuators = harris_coeffs_table[3]
#
# # Gradient z axial
# tel6 = HexRingAPLC(optics_dir, num_rings, sampling)
# tel6.create_segmented_mirror(1)
# tel6.sm.actuators = harris_coeffs_table[4]
#
# print("Allowable temperature change (in mK) for DH target contrast %s" % c_target)
# print("Faceplate silvered: ", np.sqrt(np.mean(np.square(tel2.sm.actuators)))*1e3)
# print("Bulk: ", np.sqrt(np.mean(np.square(tel3.sm.actuators)))*1e3)
# print("Gradient Radial:", np.sqrt(np.mean(np.square(tel4.sm.actuators)))*1e3)
# print("Gradient X Lateral: ", np.sqrt(np.mean(np.square(tel5.sm.actuators)))*1e3)
# print("Gradient Z axial: ", np.sqrt(np.mean(np.square(tel6.sm.actuators)))*1e3)
#
# plt.figure(figsize=(20, 10))
# ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# plt.title("Faceplates Silvered", fontsize=15)
# hcipy.imshow_field((tel2.sm.surface)*1e3, cmap = 'RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("mK",fontsize =15)
#
# ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
# plt.title("Bulk",fontsize=15)
# hcipy.imshow_field((tel3.sm.surface)*1e3, cmap = 'RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("mK",fontsize=15)
#
# ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
# plt.title("Gradiant Radial",fontsize=15)
# hcipy.imshow_field((tel4.sm.surface)*1e3, cmap = 'RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("mK",fontsize =15)
#
# ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
# plt.title("Gradient X lateral",fontsize=15)
# hcipy.imshow_field((tel5.sm.surface)*1e3,cmap = 'RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("mK",fontsize=15)
#
# ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
# plt.title("Gradient Z axial",fontsize=15)
# hcipy.imshow_field((tel6.sm.surface)*1e3,cmap = 'RdBu')
# plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
# cbar.set_label("mK",fontsize =15)
# plt.tight_layout()
# plt.savefig(os.path.join(data_dir, 'mu_map_K_%s.png'%c_target))