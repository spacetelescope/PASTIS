"""
Creating a hockey stick curve of contrast vs. RMS WFE from PASTIS and E2E calculation.
"""

import os
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from config import CONFIG_INI
import util_pastis as util
from contrast_calculation_simple import pastis_vs_e2e


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()

    # Set up path for results
    outDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'results')
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # Create range of RMS values to test
    rms_range = np.logspace(-3, 3, 10)
    print("RMS range:", rms_range)

    # Loop over different RMS values and calculate contrast with PASTIS and E2E simulation
    e2e_contrasts = []        # contrasts from E2E sim
    am_contrasts = []         # contrasts from image PASTIS
    matrix_contrasts = []     # contrasts from matrix PASTIS

    for i, rms in enumerate(rms_range):

        print("\n#####################################")
        print("CALCULATING CONTRAST FOR {:.2f}".format(rms))
        print("Run {}/{}".format(i, len(rms_range)))

        c_e2e, c_am, c_matrix = pastis_vs_e2e(rms=rms)

        e2e_contrasts.append(c_e2e)
        am_contrasts.append(c_am)
        matrix_contrasts.append(c_matrix)

    e2e_contrasts = np.array(e2e_contrasts)
    am_contrasts = np.array(am_contrasts)
    matrix_contrasts = np.array(matrix_contrasts)

    # Plot results
    dataDir = CONFIG_INI.get('local', 'local_data_path')

    plt.title("Contrast calculation")
    plt.plot(rms_range, e2e_contrasts, label="E2E")
    plt.plot(rms_range, matrix_contrasts, label="PASTIS")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("RMS WFE")
    plt.ylabel("Contrast")
    plt.legend()
    #plt.show()
    plt.savefig("PASTIS_HOCKEY_STICK.pdf")

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for pastis_vs_e2e_contrast_calc.py: {} sec = {} min'.format(runtime, runtime/60))
