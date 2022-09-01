import numpy as np
from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex

if __name__ == '__main__':

    # Set the number of rings
    num_rings = 1
    ini_path = CONFIG_PASTIS.get('local', 'local_data_path')

    # Needed for Harris mirror
    DM = 'harris_seg_mirror'  # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
    pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
    DM_SPEC = (fpath, pad_orientations, True, False, False)

    # First generate a PASTIS matrix
    run_matrix = MatrixEfieldHex(which_dm=DM, dm_spec=DM_SPEC, num_rings=num_rings,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=ini_path,
                                 saveefields=True, saveopds=True,
                                 norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'Matrix and Efields saved to {data_dir}.')
