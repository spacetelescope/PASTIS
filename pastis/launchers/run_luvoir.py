"""
Launcher script to start a full LUVOIR-A run: combinations of matrix generation, hockey stick curve and PASTIS analysis,
freely choosable between the small, medium and large coronagraph designs.
"""
import os

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve
from pastis.matrix_building_numerical import MatrixIntensityLuvoirA
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util


if __name__ == '__main__':

    APLC_DESIGN = 'small'

    # First generate a couple of matrices
    run_matrix = MatrixIntensityLuvoirA(design=APLC_DESIGN, initial_path=CONFIG_PASTIS.get('local', 'local_data_path'))
    run_matrix.calc()
    dir_run = run_matrix.overall_dir

    # Alternatively, pick data locations to run PASTIS analysis on
    #dir_run = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), 'your-data-directory')

    # Set up loggers for data analysis in all cases
    util.setup_pastis_logging(dir_run, 'pastis_analysis')
    
    # Then generate all hockey  stick curves
    result_dir_small = os.path.join(dir_run, 'results')
    matrix_dir_small = os.path.join(dir_run, 'matrix_numerical')
    hockeystick_curve(instrument='LUVOIR', apodizer_choice=APLC_DESIGN, matrixdir=matrix_dir_small, resultdir=result_dir_small, range_points=10, no_realizations=3)

    # Finally run full analysis on all three cases
    run_full_pastis_analysis(instrument='LUVOIR', design=APLC_DESIGN, run_choice=dir_run)
