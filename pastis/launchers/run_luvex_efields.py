import os
import numpy as np
from astropy.io import fits
from pastis.config import CONFIG_PASTIS
import pastis.util as util
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.simulators.scda_telescopes import HexRingAPLC
import hcipy
import matplotlib.pyplot as plt
from pastis.pastis_analysis import  calculate_segment_constraints
import pastis.plotting as ppl

if __name__ == '__main__':
    # DM_SPEC = tuple or int, specification for the used DM -
    # for seg_mirror: int, number of local Zernike modes on each segment
    # for harris_seg_mirror: tuple (string, array, bool, bool, bool),
    # absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets (thermal, mechanical, other)
    # for zernike_mirror: int, number of global Zernikes

    # If Harris deformable mirror, uncomment the following lines
    # DM = 'harris_seg_mirror'  # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    # fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
    # pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
    # DM_SPEC = (fpath, pad_orientations, True, False, False)

    # If Segmented Zernike Mirror, uncomment the following two lines
    DM = 'seg_mirror' # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    DM_SPEC = 3

    NUM_RINGS = 1

    # First generate a couple of matrices
    run_matrix = MatrixEfieldHex(which_dm=DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)
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
    c_target = 6.3*1e-11
    mus = calculate_segment_constraints(pastis_matrix, c_target=c_target, coronagraph_floor=contrast_floor)
    np.savetxt(os.path.join(dir_run, f'mus_{c_target:.2e}_{NUM_RINGS:d}.csv'), mus, delimiter=',')

    num_modes = DM_SPEC   # for harris thermal map or number of localized zernike modes = "DM_SPEC"
    nseg = run_matrix.simulator.nseg

    coeffs_table = util.sort_1d_mus_per_segment(mus, num_modes, nseg)
    mu_list = []
    label_list = []
    for i in range(coeffs_table.shape[0]):
        mu_list.append(coeffs_table[i])
        label_list.append(f'Zernike mode {i}')

    ppl.plot_segment_weights(mu_list, dir_run, c_target, labels=label_list, fname=f'stat_1d_mus_{c_target:.2e}', save=True)
    # ppl.plot_thermal_mus(mus, num_modes, nseg, c_target, dir_run, save=True)
    tel = run_matrix.simulator
    ppl.plot_multimode_mus_surface_map(tel, mus, num_modes, c_target, dir_run, mirror='sm', cmin=-5, cmax=5, save=True)
