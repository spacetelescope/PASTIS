import os
from astropy.io import fits
import numpy as np
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
import pastis.util as util


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

