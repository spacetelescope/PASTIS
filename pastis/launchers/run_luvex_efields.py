import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex
from pastis.ultra_analysis import MultiModeAnalysis


if __name__ == '__main__':
    
    NUM_RINGS = 1
    WHICH_DM = 'harris_seg_mirror'   # 'harris_seg_mirror' or 'seg_mirror', or (global) 'zernike_mirror'
    C_TARGET = 6.3 * 1e-11
    
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
        NUM_MODES = 5  # TODO: works only for thermal modes currently

    # If using Segmented Zernike Mirror
    if WHICH_DM in ['seg_mirror', 'zernike_mirror']:
        DM_SPEC = 3
        NUM_MODES = DM_SPEC

    # Calculate sensitivity matrix
    run_matrix = MatrixEfieldHex(which_dm=WHICH_DM, dm_spec=DM_SPEC, num_rings=NUM_RINGS,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=CONFIG_PASTIS.get('local', 'local_data_path'), norm_one_photon=True)
    run_matrix.calc()
    dir_run = run_matrix.overall_dir
    print(f'All saved to {dir_run}.')

    # Run the analysis
    analysis = MultiModeAnalysis(C_TARGET, run_matrix.simulator, WHICH_DM, NUM_MODES, dir_run)
    analysis.run()
