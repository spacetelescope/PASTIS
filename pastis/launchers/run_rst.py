"""
Launcher script to start a full RST run: generate matrix and run full PASTIS analysis.
"""
import os
import logging

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve
from pastis.matrix_generation.matrix_building_numerical import MatrixIntensityRST
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldRST
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util


log = logging.getLogger()

if __name__ == '__main__':

    # Generate the matrix
    generation_mode = CONFIG_PASTIS.get('RST', 'generation_mode')
    if generation_mode == 'intensity':
        run_matrix = MatrixIntensityRST(initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    elif generation_mode == 'E_field':
        run_matrix = MatrixEfieldRST(initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    else:
        error_msg = f"generation_mode from config_pastis.ini is {generation_mode}, not a valid name!"
        log.error(error_msg)
        raise ValueError(error_msg)
    run_matrix.calc()
    dir_run = run_matrix.overall_dir

    # Alternatively, pick data location to run PASTIS analysis on
    # dir_run = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-08-26T00-00-00_rst')

    # Set up loggers for data analysis
    util.setup_pastis_logging(dir_run, 'pastis_analysis')

    # Then generate hockey stick curve
    result_dir = os.path.join(dir_run, 'results')
    matrix_dir = os.path.join(dir_run, 'matrix_numerical')
    hockeystick_curve(instrument='RST', matrixdir=matrix_dir, resultdir=result_dir, range_points=10, no_realizations=3)
"""
    In development...
    # Finally run the analysis
    run_full_pastis_analysis(instrument='RST', run_choice=dir_run, c_target=1e-8)
"""
