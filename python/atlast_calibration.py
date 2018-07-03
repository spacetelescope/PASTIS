"""
translation of
"""

import os
import numpy as np
from astropy.io import fits


if __name__ == '__main__':

    # Import APLC and Lyot stop
    pup_up = fits.getdata('/Users/ilaginja/Documents/Git/PASTIS/data/ApodSol_APLC_quart_atlastX025cobs1gap1_N0354r_FPM450M060_LSann20D70clear_Img097C_40DA100_BW10Nlam04fpres2_linbarhompre1.fits')   # APLC
    pup_do = fits.getdata('/Users/ilaginja/Documents/Git/PASTIS/data/LS_full_atlast.fits')   # Lyot stop