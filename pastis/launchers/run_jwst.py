"""
Launcher script to start a full JWST run: generate matrix and run full PASTIS analysis.
"""
import os

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve
from pastis.matrix_generation.matrix_building_numerical import MatrixIntensityJWST
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util


if __name__ == '__main__':

    # Generate the matrix
    run_matrix = MatrixIntensityJWST(initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    run_matrix.calc()
    dir_run = run_matrix.overall_dir

    # Alternatively, pick data location to run PASTIS analysis on
    # dir_run = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '2020-08-26T00-00-00_jwst')

    # Set up loggers for data analysis
    util.setup_pastis_logging(dir_run, 'pastis_analysis')

    # Then generate hockey stick curve
    result_dir = os.path.join(dir_run, 'results')
    matrix_dir = os.path.join(dir_run, 'matrix_numerical')
    hockeystick_curve(instrument='JWST', matrixdir=matrix_dir, resultdir=result_dir, range_points=10, no_realizations=3)

    # Finally run the analysis
    run_full_pastis_analysis(instrument='JWST', run_choice=dir_run, c_target=1e-6)
