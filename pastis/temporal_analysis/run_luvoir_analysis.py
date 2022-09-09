import os
import hcipy
from astropy.io import fits
import numpy as np
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
import pastis.util as util


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

# Create necessary directories if they don't exist yet
data_dir = CONFIG_PASTIS.get('local', 'local_data_path')

# Instantiate the LUVOIR telescope
optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
coronagraph_design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
tel = LuvoirA_APLC(optics_input, coronagraph_design, sampling)

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
c_target = 10**(c_target_log)
mu_map_harris = np.sqrt(((c_target) / (num_actuators)) / (np.diag(pastis_matrix)))
np.savetxt(os.path.join(data_dir, 'mu_map_harris_%s.csv' % c_target), mu_map_harris, delimiter=',')



# Temporal Analysis
efield_coron_real = fits.getdata(os.path.join(data_dir, 'efield_coron_real.fits'))
efield_coron_imag = fits.getdata(os.path.join(data_dir, 'efield_coron_imag.fits'))
efield_obwfs_real = fits.getdata(os.path.join(data_dir, 'efield_obwfs_real.fits'))
efield_obwfs_imag = fits.getdata(os.path.join(data_dir, 'efield_obwfs_imag.fits'))
e0_coron = fits.getdata(os.path.join(data_dir, 'e0_coron.fits'))
e0_obwfs = fits.getdata(os.path.join(data_dir, 'e0_wfs.fits'))

# Get the total pupil and imaging camera pixels
total_sci_pix = np.square(e0_coron.shape[1])
total_pupil_pix = np.square(e0_obwfs.shape[1])

e0_coron_real = np.reshape(e0_coron[0], total_sci_pix)
e0_coron_imag = np.reshape(e0_coron[1], total_sci_pix)

e0_obwfs_real = np.reshape(e0_obwfs[0], total_pupil_pix)
e0_obwfs_imag = np.reshape(e0_obwfs[1], total_pupil_pix)

E0_coron = np.zeros([total_sci_pix, 1, 2])
E0_coron[:, 0, 0] = e0_coron_real
E0_coron[:, 0, 1] = e0_coron_imag

z_pup_downsample = CONFIG_PASTIS.getint('numerical', 'z_pup_downsample')
e0_wfs_sub_real = np.reshape(matrix_subsample(e0_obwfs[0], 125, 125), 125*125)
e0_wfs_sub_imag = np.reshape(matrix_subsample(e0_obwfs[1], 125, 125), 125*125)
efield_ref_wfs_sub = (e0_wfs_sub_real + 1j * e0_wfs_sub_imag)

N_pup_z = efield_ref_wfs_sub.real.shape[0]
E0_OBWFS_downsampled = np.zeros([int(N_pup_z), 1, 2])
E0_OBWFS_downsampled[:, 0, 0] = efield_ref_wfs_sub.real
E0_OBWFS_downsampled[:, 0, 1] = efield_ref_wfs_sub.imag

num_all_modes = efield_coron_real.shape[0]
G_coron = np.zeros([total_sci_pix, 2, num_all_modes])
for i in range(num_all_modes):
    G_coron[:, 0, i] = np.reshape(efield_coron_real[i], total_sci_pix) - e0_coron_real
    G_coron[:, 1, i] = np.reshape(efield_coron_imag[i], total_sci_pix) - e0_coron_imag

G_OBWFS = np.zeros([total_pupil_pix, 2, num_all_modes])
for i in range(num_all_modes):
    G_OBWFS[:, 0, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - e0_obwfs_real
    G_OBWFS[:, 1, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - e0_obwfs_imag

G_OBWFS = np.empty([2, num_all_modes, total_pupil_pix])
for i in range(num_all_modes):
    G_OBWFS[0, i, :] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - e0_obwfs_real
    G_OBWFS[1, i, :] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - e0_obwfs_imag
G_OBWFS = np.transpose(G_OBWFS, axes=(2, 0, 1))

G_OBWFS_downsampled = np.empty([int(N_pup_z), 2, num_all_modes])
for i in range(num_all_modes):
    efields_per_mode_wfs_real_sub = np.reshape(matrix_subsample(efield_obwfs_real[i], 125, 125), int(N_pup_z))
    efields_per_mode_wfs_imag_sub = np.reshape(matrix_subsample(efield_obwfs_imag[i], 125, 125), int(N_pup_z))
    G_OBWFS_downsampled[:, 0, i] = efields_per_mode_wfs_real_sub - e0_wfs_sub_real
    G_OBWFS_downsampled[:, 1, i] = efields_per_mode_wfs_imag_sub - e0_wfs_sub_imag
