"""
Launcher script to start a full HiCAT run on the BostonDM: generate matrix and run full PASTIS analysis.
"""
import os

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve
from pastis.matrix_building_numerical import num_matrix_multiprocess
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util


if __name__ == '__main__':

    # Generate the matrix
    dir_run = num_matrix_multiprocess(instrument='HiCAT_continuous', savepsfs=False, saveopds=False)

    # Alternatively, pick data location to run PASTIS analysis on
    #dir_run = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-10-14T10-51-49_hicat_continuous')

    # Set up loggers for data analysis
    util.setup_pastis_logging(dir_run, 'pastis_analysis')

    # Then generate hockeystick curve
    result_dir = os.path.join(dir_run, 'results')
    matrix_dir = os.path.join(dir_run, 'matrix_numerical')
    #hockeystick_curve(instrument='HiCAT_continuous', matrixdir=matrix_dir, resultdir=result_dir, range_points=10, no_realizations=3)

    # Finally run the analysis
    run_full_pastis_analysis(instrument='HiCAT_continuous', run_choice=dir_run, c_target=1e-7)
