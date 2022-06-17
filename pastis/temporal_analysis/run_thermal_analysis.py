import os
import numpy as np
from astropy.io import fits
import astropy.units as u
import pandas as pd
import exoscene.image
import exoscene.star
import exoscene.planet
from exoscene.planet import Planet
import pastis.util as util
from pastis.config import CONFIG_PASTIS
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldLuvoirA
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.plotting import plot_segment_weights
from pastis.temporal_analysis.close_loop_analysis import req_closedloop_calc_batch
from pastis.e2e_simulators.luvoir_imaging import LuvoirA_APLC
from pastis.e2e_simulators.generic_segmented_telescopes import SegmentedAPLC

if __name__ == '__main__':
    # create pastis_matrix using thermal modes
    APLC_DESIGN = 'small'
    DM = 'harris_seg_mirror'  # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"

    # Needed for Harris mirror
    fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
    pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
    DM_SPEC = (fpath, pad_orientations, True, False, False)

    # First generate a PASTIS matrix
    run_matrix = MatrixEfieldLuvoirA(which_dm=DM, dm_spec=DM_SPEC, design=APLC_DESIGN,
                                     initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'Matrix and Efields saved to {data_dir}.')

    # Calculate tolerance per segments per mode
    pastis_matrix = fits.getdata(os.path.join(data_dir,'matrix_numerical', 'pastis_matrix.fits'))
    mus_harris = calculate_segment_constraints(pastis_matrix, c_target=1e-11, coronagraph_floor=0)

    # Plot the
    plot_segment_weights(mus=mus_harris, out_dir=data_dir, c_target=1e-11, labels=None, fname_suffix='', save=True)

