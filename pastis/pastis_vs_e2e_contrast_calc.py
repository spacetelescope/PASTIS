"""
Creating a hockey stick curve of contrast vs. RMS WFE from PASTIS and E2E calculation.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt

from config import CONFIG_INI
from contrast_calculation_simple import pastis_vs_e2e


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()

    ##########################
    WORKDIRECTORY = "active"                # you can chose here what data directory to work in
                                            # anything else than "active" works only with im_pastis=False
    matrix = "analytical"                   # "analytical" or "numerical"
    rms_range = np.logspace(-1, 3, 50)      # Create range of RMS values to test
    realiz = 10                             # how many random realizations per RMS values to do
    ##########################

    # Set up path for results
    outDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), WORKDIRECTORY, 'results')
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    if not os.path.isdir(os.path.join(outDir, 'dh_images_'+matrix)):
        os.mkdir(os.path.join(outDir, 'dh_images_'+matrix))

    # Loop over different RMS values and calculate contrast with PASTIS and E2E simulation
    e2e_contrasts = []        # contrasts from E2E sim
    am_contrasts = []         # contrasts from image PASTIS
    matrix_contrasts = []     # contrasts from matrix PASTIS

    print("RMS range: {}".format(rms_range, fmt="%e"))
    print("Random realizations: {}".format(realiz))

    for i, rms in enumerate(rms_range):

        rms *= u.nm  # Making sure this has the correct units

        e2e_rand = []
        am_rand = []
        matrix_rand = []

        for j in range(realiz):
            print("\n#####################################")
            print("CALCULATING CONTRAST FOR {:.4f}".format(rms))
            print("RMS {}/{}".format(i + 1, len(rms_range)))
            print("Random realization: {}/{}".format(j+1, realiz))
            print("Total: {}/{}\n".format((i*realiz)+(j+1), len(rms_range)*realiz))

            c_e2e, c_am, c_matrix = pastis_vs_e2e(dir=WORKDIRECTORY, matrix_mode=matrix, rms=rms,
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
    df.to_csv(os.path.join(outDir, "contrasts_"+matrix+".txt"), sep=' ', na_rep='NaN')

    # Plot results
    dataDir = CONFIG_INI.get('local', 'local_data_path')

    plt.clf()
    plt.title("Contrast calculation")
    plt.plot(rms_range, e2e_contrasts, label="E2E")
    plt.plot(rms_range, am_contrasts, label="Image PASTIS")
    plt.plot(rms_range, matrix_contrasts, label="Matrix PASTIS")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("Surface RMS in " + str(u.nm))
    plt.ylabel("Contrast")
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(outDir, "PASTIS_HOCKEY_STICK_"+matrix+".pdf"))

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for pastis_vs_e2e_contrast_calc.py: {} sec = {} min'.format(runtime, runtime/60))
