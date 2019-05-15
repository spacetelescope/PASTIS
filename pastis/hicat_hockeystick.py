"""
Creating a hockey stick curve of contrast vs. RMS WFE from HiCAT sim numerical PASTIS.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt

from config import CONFIG_INI
from contrast_calc_hicat import contrast_hicat_num


if __name__ == '__main__':

    # Keep track of time
    start_time = time.time()

    ##########################
    rms_range = np.logspace(-1, 3, 5)      # Create range of RMS values to test
    realiz = 2                             # how many random realizations per RMS values to do
    ##########################

    # Loop over different RMS values and calculate contrast with MATRIX PASTIS and E2E simulation
    e2e_contrasts = []        # contrasts from E2E sim
    matrix_contrasts = []     # contrasts from matrix PASTIS

    print("RMS range: {}".format(rms_range, fmt="%e"))
    print("Random realizations: {}".format(realiz))

    for i, rms in enumerate(rms_range):

        rms *= u.nm  # Making sure this has the correct units

        e2e_rand = []
        matrix_rand = []

        for j in range(realiz):
            print("\n#####################################")
            print("CALCULATING CONTRAST FOR {:.4f}".format(rms))
            print("RMS {}/{}".format(i + 1, len(rms_range)))
            print("Random realization: {}/{}".format(j+1, realiz))
            print("Total: {}/{}\n".format((i*realiz)+(j+1), len(rms_range)*realiz))

            c_e2e, c_matrix = contrast_hicat_num(dir='/Users/ilaginja/Documents/Git/PASTIS/Jupyter Notebooks/HiCAT',
                                                 rms=rms,)

            e2e_rand.append(c_e2e)
            matrix_rand.append(c_matrix)

        e2e_contrasts.append(np.mean(e2e_rand))
        matrix_contrasts.append(np.mean(matrix_rand))

    plt.clf()
    plt.title("Contrast calculation")
    plt.plot(rms_range, e2e_contrasts, label="E2E")
    plt.plot(rms_range, matrix_contrasts, label="Matrix PASTIS")
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("Surface RMS in " + str(u.nm))
    plt.ylabel("Contrast")
    plt.legend()
    plt.show()

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime for pastis_vs_e2e_contrast_calc.py: {} sec = {} min'.format(runtime, runtime/60))
