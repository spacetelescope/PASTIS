from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA


if __name__ == '__main__':

    APLC_DESIGN = 'small'
    DM = 'seg_mirror'   # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    DM_SPEC = 3
    # DM_SPEC = tuple or int, specification for the used DM -
    #    for seg_mirror: int, number of local Zernike modes on each segment
    #    for harris_seg_mirror: tuple (string, array), absolute path to Harris spreadsheet, pad orientations
    #    for zernike_mirror: int, number of global Zernikes

    # First generate a couple of matrices
    run_matrix = MatrixEfieldLuvoirA(which_dm=DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                     initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    run_matrix.calc()
    dir_run = run_matrix.overall_dir
    print(f'All saved to {dir_run}.')
