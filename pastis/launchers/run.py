"""
Launcher script to start a full RST run: generate matrix and run full PASTIS analysis.
"""
import os

from pastis.config import CONFIG_PASTIS
from pastis.hockeystick_contrast_curve import hockeystick_curve_class
from pastis.pastis_analysis import run_full_pastis_analysis
import pastis.util as util
import pastis.launchers.parameters as parameters


if __name__ == '__main__':
    run_gen = CONFIG_PASTIS.getboolean('generation','run')
    run_hockey = CONFIG_PASTIS.getboolean('hockeystick','run')
    run_analysis = CONFIG_PASTIS.getboolean('analysis','run')

    #Generate PASTIS matrix
    if run_gen :
        run_matrix = parameters.gen_method()
        run_matrix.calc()

    # Set up loggers for data analysis
    if run_hockey or run_analysis :
        dir_run = parameters.dir_analysis()
        util.setup_pastis_logging(dir_run, 'pastis_analysis')
        result_dir = os.path.join(dir_run, 'results')
        matrix_dir = os.path.join(dir_run, 'matrix_numerical')

    # Then generate hockey stick curve
    if run_hockey :
        hockeystick_curve_class(dir_run)

    #In development...
    # Finally run the analysis
    #if run_analysis :
        #run_full_pastis_analysis(instrument='RST', run_choice=dir_run, c_target=1e-8)
