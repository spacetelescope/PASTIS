"""
Creating a hockey stick curve of contrast vs. RMS WFE from PASTIS and E2E calculation.

There are three methods to calculate the final mean contrast:
1. E2E coronagraph
2. (Image-based PASTIS)
3. Matrix-based PASTIS

All three methods are currently only supported for JWST, and you can pick between the analytical or numerical matrix.
HiCAT and LUVOIR only have an E2E vs numerical PASTIS comparison (1 + 3).
"""

import os
import time
import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pastis.config import CONFIG_PASTIS
import pastis.contrast_calculation_simple as consim
from pastis.matrix_generation.matrix_building_numerical import calculate_unaberrated_contrast_and_normalization
import pastis.plotting as ppl

log = logging.getLogger()


def hockeystick_jwst(range_points=3, no_realizations=3, matrix_mode='analytical'):
    """
    Construct a PASTIS hockeystick contrast curve for validation of the PASTIS matrix for JWST.

    The aberration range is a fixed parameter in the function body since it depends on the coronagraph (and telescope)
    used. We define how many realizations of a specific rms error we want to run through, and also how many points we
    want to fill the aberration range with. At each point we calculate the contrast for all realizations and plot the
    mean of this set of results in a figure that shows contrast vs. rms phase error.

    :param range_points: int, How many points of rms error (OPD) to use in the predefined aberration range.
    :param no_realizations: int, How many realizations per rms error (OPD) should be calculated; the mean of the realizations
                                is used.
    :param matrix_mode: string, Choice of PASTIS matrix to validate: 'analytical' or 'numerical'
    :return:
    """

    # Keep track of time
    start_time = time.time()

    ##########################
    WORKDIRECTORY = "active"                # you can chose here what data directory to work in
                                            # anything else than "active" works only with im_pastis=False
    rms_range = np.logspace(-1, 3, range_points)      # Create range of RMS values to test
    realiz = no_realizations                             # how many random realizations per RMS values to do
    ##########################

    # Set up path for results
    outDir = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), WORKDIRECTORY, 'results')
    os.makedirs(outDir, exist_ok=True)
    os.makedirs(os.path.join(outDir, 'dh_images_'+matrix_mode), exist_ok=True)

    # Loop over different RMS values and calculate contrast with PASTIS and E2E simulation
    e2e_contrasts = []        # contrasts from E2E sim
    am_contrasts = []         # contrasts from image PASTIS
    matrix_contrasts = []     # contrasts from matrix PASTIS

    log.info("RMS range: {}".format(rms_range, fmt="%e"))
    log.info(f"Random realizations: {realiz}")

    for i, rms in enumerate(rms_range):

        rms *= u.nm  # Making sure this has the correct units

        e2e_rand = []
        am_rand = []
        matrix_rand = []

        for j in range(realiz):
            log.info("\n#####################################")
            log.info("CALCULATING CONTRAST FOR {:.4f}".format(rms))
            log.info(f"RMS {i + 1}/{len(rms_range)}")
            log.info(f"Random realization: {j+1}/{realiz}")
            log.info(f"Total: {(i*realiz)+(j+1)}/{len(rms_range)*realiz}\n")

            c_e2e, c_am, c_matrix = consim.contrast_jwst_ana_num(matdir=WORKDIRECTORY, matrix_mode=matrix_mode, rms=rms,
                                                                 im_pastis=True, plotting=True)

            e2e_rand.append(c_e2e)
            am_rand.append(c_am)
            matrix_rand.append(c_matrix)

        e2e_contrasts.append(np.mean(e2e_rand))
        am_contrasts.append(np.mean(am_rand))
        matrix_contrasts.append(np.mean(matrix_rand))

    e2e_contrasts = np.array(e2e_contrasts)
    am_contrasts = np.array(am_contrasts)
    matrix_contrasts = np.array(matrix_contrasts)

    # Save results to txt file
    df = pd.DataFrame({'rms': rms_range, 'c_e2e': e2e_contrasts, 'c_am': am_contrasts, 'c_matrix': matrix_contrasts})
    df.to_csv(os.path.join(outDir, "hockey_contrasts_"+matrix_mode+".txt"), sep=' ', na_rep='NaN')

    # Plot
    plt.clf()
    plt.title("Contrast calculation")
    plt.plot(rms_range, e2e_contrasts, label="E2E")
    plt.plot(rms_range, am_contrasts, label="Image PASTIS")
    plt.plot(rms_range, matrix_contrasts, label="Matrix PASTIS")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("WFE RMS (OPD) in " + str(u.nm))
    plt.ylabel("Contrast")
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(outDir, "PASTIS_HOCKEY_STICK_"+matrix_mode+".pdf"))

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'Runtime for pastis_vs_e2e_contrast_calc.py: {runtime} sec = {runtime/60} min')


def hockeystick_curve(instrument, apodizer_choice=None, matrixdir='', resultdir='', range_points=3, no_realizations=3):
    """
    Construct a PASTIS hockeystick contrast curve for validation of the PASTIS matrix, for one particular instrument.

    The aberration range is a fixed parameter in the function body since it depends on the coronagraph (and telescope)
    used. We define how many realizations of a specific WFE rms error we want to run through, and also how many points we
    want to fill the aberration range with. At each point we calculate the contrast for all realizations and plot the
    mean of this set of results in a figure that shows contrast vs. WFE rms error.

    :param instrument: string, 'LUVOIR', 'HiCAT' or 'JWST'
    :param apodizer_choice: string, needed if instrument='LUVOIR'; use "small", "medium" or "large" FPM coronagraph
    :param matrixdir: string, Path to matrix that should be used.
    :param resultdir: string, Path to directory where results will be saved.
    :param range_points: int, How many points of WFE rms error to use in the predefined aberration range.
    :param no_realizations: int, How many realizations per WFE rms error should be calculated; the mean of the realizations
                                is used in the plot
    :return:
    """

    if instrument == 'LUVOIR' and apodizer_choice is None:
        raise ValueError('Need to specify apodizer_choice when working with LUVOIR instrument.')

    # Keep track of time
    start_time = time.time()

    # Create range of WFE RMS values to test
    rms_range = np.logspace(CONFIG_PASTIS.getfloat(instrument, 'valid_range_lower'),
                            CONFIG_PASTIS.getfloat(instrument, 'valid_range_upper'),
                            range_points)

    # Create results directory if it doesn't exist yet
    os.makedirs(resultdir, exist_ok=True)

    # Calculate coronagraph floor, and normalization factor from direct image
    contrast_floor, norm = calculate_unaberrated_contrast_and_normalization(instrument, apodizer_choice, return_coro_simulator=False)

    # Loop over different RMS values and calculate contrast with MATRIX PASTIS and E2E simulation
    e2e_contrasts = []        # contrasts from E2E sim
    matrix_contrasts = []     # contrasts from matrix PASTIS

    log.info("WFE RMS range: {} nm".format(rms_range, fmt="%e"))
    log.info(f"Random realizations: {no_realizations}")

    for i, rms in enumerate(rms_range):

        rms *= u.nm  # Making sure this has the correct units

        e2e_rand = []
        matrix_rand = []

        for j in range(no_realizations):
            log.info("CALCULATING CONTRAST FOR {:.4f}".format(rms))
            log.info(f"WFE RMS number {i + 1}/{len(rms_range)}")
            log.info(f"Random realization: {j+1}/{no_realizations}")
            log.info(f"Total: {(i*no_realizations)+(j+1)}/{len(rms_range)*no_realizations}")

            c_e2e, c_matrix = consim.contrast_general_num(matrix_dir=matrixdir, rms=rms)
            e2e_rand.append(c_e2e)
            matrix_rand.append(c_matrix)

        e2e_contrasts.append(np.mean(e2e_rand))
        matrix_contrasts.append(np.mean(matrix_rand))

    # Save contrasts and rms range
    np.savetxt(os.path.join(resultdir, 'hockey_rms_range.txt'), rms_range)
    np.savetxt(os.path.join(resultdir, 'hockey_e2e_contrasts.txt'), e2e_contrasts)
    np.savetxt(os.path.join(resultdir, 'hockey_matrix_contrasts.txt'), matrix_contrasts)

    # Plot
    plt.clf()
    ppl.plot_hockey_stick_curve(rms_range, matrix_contrasts, e2e_contrasts,
                                wvln=CONFIG_PASTIS.getfloat(instrument, 'lambda'),
                                out_dir=resultdir,
                                fname_suffix=f'{no_realizations}_realizations_each',
                                save=True)

    end_time = time.time()
    runtime = end_time - start_time
    log.info(f'\nTotal runtime for pastis_vs_e2e_contrast_calc.py: {runtime} sec = {runtime/60} min')


if __name__ == '__main__':

    # Pick one to run
    #hockeystick_jwst()

    instrument = CONFIG_PASTIS.get('telescope', 'name')
    run_choice = CONFIG_PASTIS.get('numerical', 'current_analysis')
    coro_design = CONFIG_PASTIS.get('LUVOIR', 'coronagraph_design')
    result_dir = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), run_choice, 'results')
    matrix_dir = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), run_choice, 'matrix_numerical')
    hockeystick_curve(instrument, apodizer_choice=coro_design, matrixdir=matrix_dir, resultdir=result_dir, range_points=30, no_realizations=10)
