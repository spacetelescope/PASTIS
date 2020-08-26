"""
Launcher script to start a full HiCAT run: generate matrix and run full PASTIS analysis.
"""
import os

from config import CONFIG_INI
from hockeystick_contrast_curve import hockeystick_curve
from matrix_building_numerical import num_matrix_multiprocess
from pastis_analysis import run_full_pastis_analysis_luvoir
import util_pastis as util


if __name__ == '__main__':

    # Generate the matrix
    dir_matrix = num_matrix_multiprocess(instrument='HiCAT')

    # Alternatively, pick data location to run PASTIS analysis on
    #dir_matrix = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'your-data-directory')

    # Set up loggers for data analysis
    util.setup_pastis_logging(dir_matrix, 'pastis_analysis')

    # Then generate hockeystick curve
    result_dir_small = os.path.join(dir_matrix, 'results')
    matrix_dir_small = os.path.join(dir_matrix, 'matrix_numerical')
    hockeystick_curve(instrument='HiCAT', matrixdir=matrix_dir_small, resultdir=result_dir_small, range_points=10, no_realizations=2)

    # Finally run the analysis
    run_full_pastis_analysis_luvoir(instrument='HiCAT', design='small', run_choice=matrix_dir_small)
