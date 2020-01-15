"""
Run different cases, free combination of matrices and analysis scripts.
"""
import os
from hockeystick_contrast_curve import hockeystick_luvoir
from matrix_building_numerical import num_matrix_luvoir
from modal_analysis import run_full_pastis_analysis_luvoir


if __name__ == '__main__':
    
    # First generate a couple of matrices, all with 1 nm of calibration aberration
    # matrix_pastis /= np.square(nm_aber) *after* fixing of off-axis elements is included
    dir_small = num_matrix_luvoir(design='small')
    dir_medium = num_matrix_luvoir(design='medium')
    dir_large = num_matrix_luvoir(design='large')
    
    # Then generate all hockeystick curves
    #dir_small = '/Users/ilaginja/Documents/data_from_repos/pastis_data/2020-01-13T22-05-24_luvoir-small'
    result_dir_small = os.path.join(dir_small, 'results')
    matrix_dir_small = os.path.join(dir_small, 'matrix_numerical')
    hockeystick_luvoir(apodizer_choice='small', matrixdir=matrix_dir_small, resultdir=result_dir_small, range_points=50, no_realizations=20)

    #dir_medium = '/Users/ilaginja/Documents/data_from_repos/pastis_data/2020-01-14T00-19-14_luvoir-medium'
    result_dir_medium = os.path.join(dir_medium, 'results')
    matrix_dir_medium = os.path.join(dir_medium, 'matrix_numerical')
    hockeystick_luvoir(apodizer_choice='medium', matrixdir=matrix_dir_medium, resultdir=result_dir_medium, range_points=50, no_realizations=20)

    #dir_large = '/Users/ilaginja/Documents/data_from_repos/pastis_data/2020-01-14T02-40-04_luvoir-large'
    result_dir_large = os.path.join(dir_large, 'results')
    matrix_dir_large = os.path.join(dir_large, 'matrix_numerical')
    hockeystick_luvoir(apodizer_choice='large', matrixdir=matrix_dir_large, resultdir=result_dir_large, range_points=50, no_realizations=20)
    
    # Finally run full analysis on all three cases
    run_full_pastis_analysis_luvoir(design='small', run_choice=dir_small)
    run_full_pastis_analysis_luvoir(design='medium', run_choice=dir_medium)
    run_full_pastis_analysis_luvoir(design='large', run_choice=dir_large)
