"""
Launcher script to start a full RST run: generate matrix and run full PASTIS analysis.
"""
import os

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve
from pastis.matrix_generation.matrix_building_numerical import MatrixIntensityRST
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldRST
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util
import logging


log = logging.getLogger()
mplfm_logger = logging.getLogger('matplotlib.font_manager')
mplcb_logger = logging.getLogger('matplotlib.colorbar')
mplt_logger = logging.getLogger('matplotlib.ticker')
mplbe_logger = logging.getLogger('matplotlib.backends')

mplfm_logger.setLevel(logging.WARNING)
mplcb_logger.setLevel(logging.WARNING)
mplt_logger.setLevel(logging.WARNING)
mplbe_logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    '''
    # Generate intensity matrix
    #run_matrix = MatrixIntensityRST(initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))

    # Generate E_field matrix
    run_matrix = MatrixEfieldRST(initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))

    run_matrix.calc()
    dir_run = run_matrix.overall_dir
    '''
    # Alternatively, pick data location to run PASTIS analysis on
    dir_run = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '24x24_E_field_nlambda_2021-06-29T15-13-56_rst')

    # Set up loggers for data analysis
    util.setup_pastis_logging(dir_run, 'pastis_analysis')

    # Then generate hockey stick curve
    result_dir = os.path.join(dir_run, 'results')
    matrix_dir = os.path.join(dir_run, 'matrix_numerical')
    hockeystick_curve(instrument='RST', matrixdir=matrix_dir, resultdir=result_dir, range_points=30, no_realizations=1)

    # Finally run the analysis
    run_full_pastis_analysis(instrument='RST', run_choice=dir_run, c_target=1e-8)

