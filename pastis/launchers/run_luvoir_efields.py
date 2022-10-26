import os
from astropy.io import fits
import numpy as np
from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.pastis_analysis import  calculate_segment_constraints
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
import pastis.plotting as ppl

if __name__ == '__main__':

    NUM_RINGS = 1
    WHICH_DM = 'harris_seg_mirror'   # 'harris_seg_mirror' or 'seg_mirror', or (global) 'zernike_mirror'

    # DM_SPEC = tuple or int, specification for the used DM -
    # for seg_mirror: int, number of local Zernike modes on each segment
    # for harris_seg_mirror: tuple (string, array, bool, bool, bool),
    # absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets (thermal, mechanical, other)
    # for zernike_mirror: int, number of global Zernikes

    # If using Harris deformable mirror
    if WHICH_DM == 'harris_seg_mirror':
        fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
        pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
        DM_SPEC = (fpath, pad_orientations, True, False, False)

    # If using Segmented Zernike Mirror
    if WHICH_DM == 'seg_mirror':
        DM_SPEC = 3

    APLC_DESIGN = 'small'
    # First generate a couple of matrices
    run_matrix = MatrixEfieldLuvoirA(which_dm=WHICH_DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                     calc_science=True, calc_wfs=True,
                                     initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
                                     norm_one_photon=True)
    run_matrix.calc()
    dir_run = run_matrix.overall_dir
    print(f'All saved to {dir_run}.')

    # get the automatically saved pastis_matrix
    pastis_matrix = fits.getdata(os.path.join(dir_run, 'matrix_numerical', 'pastis_matrix.fits'))

    # get the unaberrated coro_psf after the matrix run
    e0_psf = fits.getdata(os.path.join(dir_run, 'unaberrated_coro_psf.fits'))  # already normalized to max of direct pdf
    dh_mask = np.array(run_matrix.simulator.dh_mask.shaped)
    contrast_floor = util.dh_mean(e0_psf, dh_mask)

    # Calculate the static tolerances
    c_target = 5.3*1e-11
    mus = calculate_segment_constraints(pastis_matrix, c_target=c_target, coronagraph_floor=contrast_floor)
    np.savetxt(os.path.join(dir_run, f'mu_map_{c_target:.2e}.csv'), mus, delimiter=',')

    num_modes = 5  # for harris thermal map or number of localized zernike modes
    nseg = run_matrix.simulator.nseg

    coeffs_table = util.sort_1d_mus_per_segment(mus, num_modes, nseg)
    mu_list = []
    label_list = []
    for i in range(coeffs_table.shape[0]):
        mu_list.append(coeffs_table[i])
        if WHICH_DM == 'seg_mirror':
            label_list.append(f'Zernike mode {i}')
    if WHICH_DM == 'harris_seg_mirror':
        label_list = ['Faceplates Silvered', 'Bulk', 'Gradient Radial', 'Gradient X lateral', 'Gradient Z axial']

    ppl.plot_segment_weights(mu_list, dir_run, c_target, labels=label_list, fname=f'stat_1d_mus_{c_target:.2e}', save=True)
    tel = run_matrix.simulator
    ppl.plot_multimode_mus_surface_map(tel, mus, num_modes, c_target, dir_run, mirror='sm', cmin=-5, cmax=5, save=True)
