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

    APLC_DESIGN = 'small'
    DM = 'harris_seg_mirror'   # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"

    # Needed for Harris mirror
    fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
    pad_orientations = np.pi / 2 * np.ones(120)

    DM_SPEC = (fpath, pad_orientations, True, False, False)
    # DM_SPEC = tuple or int, specification for the used DM -
    #    for seg_mirror: int, number of local Zernike modes on each segment
    #    for harris_seg_mirror: tuple (string, array, bool, bool, bool),
    #    absolute path to Harris spreadsheet, pad orientations, choice of Harris mode sets (thermal, mechanical, other)
    #    for zernike_mirror: int, number of global Zernikes

    # First generate a couple of matrices
    run_matrix = MatrixEfieldLuvoirA(which_dm=DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                     calc_science=True, calc_wfs=True,
                                     initial_path=CONFIG_PASTIS.get('local', 'local_data_path'),
                                     norm_one_photon=True)
    run_matrix.calc()
    dir_run = run_matrix.overall_dir
    print(f'All saved to {dir_run}.')


    # Instantiate the LUVOIR telescope
    optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    coronagraph_design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    tel = LuvoirA_APLC(optics_input, coronagraph_design, sampling)
    nm_aber = CONFIG_PASTIS.getfloat('LUVOIR', 'calibration_aberration') * 1e-9

    # Create harris deformable mirror
    tel.create_segmented_harris_mirror(fpath, pad_orientations, thermal=True, mechanical=False, other=False)
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

    # Calculate and save per segment static tolerance
    pastis_matrix = fits.getdata(os.path.join(dir_run, 'matrix_numerical', 'pastis_matrix.fits'))
    c_target = 5.3*1e-11
    mus = calculate_segment_constraints(pastis_matrix, c_target=c_target, coronagraph_floor=contrast_floor)
    np.savetxt(os.path.join(dir_run, 'mu_map_harris_%s.csv' % c_target), mus, delimiter=',')

    ppl.plot_thermal_mus(mus, num_modes, tel.nseg, c0=c_target, out_dir=dir_run, save=True)
    ppl.plot_multimode_mus_surface_map(tel, mus, num_modes, num_actuators, c_target, dir_run, save=False)
