"""
Launcher script to start a full LUVOIR run: combinations of matrix generation, hockey stick curve and PASTIS analysis,
freely choosable between the small, medium and large coronagraph designs.
"""
import os

from config import CONFIG_INI
from hockeystick_contrast_curve import hockeystick_curve
from matrix_building_numerical import num_matrix_multiprocess
from pastis_analysis import run_full_pastis_analysis
import util_pastis as util


if __name__ == '__main__':

    # First generate a couple of matrices
    dir_small = num_matrix_multiprocess(instrument='LUVOIR', design='small')
    #dir_medium = num_matrix_multiprocess(instrument='LUVOIR', design='medium')
    #dir_large = num_matrix_multiprocess(instrument='LUVOIR', design='large')

    # Alternatively, pick data locations to run PASTIS analysis on
    #dir_small = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'your-data-directory_small')
    #dir_medium = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'your-data-directory_medium')
    #dir_large = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'your-data-directory_medium')

    # Set up loggers for data analysis in all cases
    util.setup_pastis_logging(dir_small, 'pastis_analysis')
    #util.setup_pastis_logging(dir_medium, 'pastis_analysis')
    #util.setup_pastis_logging(dir_large, 'pastis_analysis')
    
    # Then generate all hockeystick curves
    result_dir_small = os.path.join(dir_small, 'results')
    matrix_dir_small = os.path.join(dir_small, 'matrix_numerical')
    hockeystick_curve(instrument='LUVOIR', apodizer_choice='small', matrixdir=matrix_dir_small, resultdir=result_dir_small, range_points=50, no_realizations=20)

    #result_dir_medium = os.path.join(dir_medium, 'results')
    #matrix_dir_medium = os.path.join(dir_medium, 'matrix_numerical')
    #hockeystick_curve(instrument='LUVOIR', apodizer_choice='medium', matrixdir=matrix_dir_medium, resultdir=result_dir_medium, range_points=50, no_realizations=20)

    #result_dir_large = os.path.join(dir_large, 'results')
    #matrix_dir_large = os.path.join(dir_large, 'matrix_numerical')
    #hockeystick_curve(instrument='LUVOIR', apodizer_choice='large', matrixdir=matrix_dir_large, resultdir=result_dir_large, range_points=50, no_realizations=20)
    
    # Finally run full analysis on all three cases
    run_full_pastis_analysis(instrument='LUVOIR', design='small', run_choice=dir_small)
    #run_full_pastis_analysis(instrument='LUVOIR', design='medium', run_choice=dir_medium)
    #run_full_pastis_analysis(instrument='LUVOIR', design='large', run_choice=dir_large)
