"""
Testing the integrated energy of images produced by HCIPy vs Poppy segmented DMs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from hcipy import *
import poppy

from config import CONFIG_INI
import util_pastis as util
import atlast_imaging as atim


def aber_to_opd(aber_rad, wvln):
    aber_m = aber_rad * wvln / (2 * np.pi)
    return aber_m


if __name__ == '__main__':

    # Parameters
    which_tel = CONFIG_INI.get('telescope', 'name')
    NPIX = CONFIG_INI.getint('numerical', 'tel_size_px')
    PUP_DIAMETER = CONFIG_INI.getfloat(which_tel, 'diameter')
    GAPSIZE = CONFIG_INI.getfloat(which_tel, 'gaps')
    FLATTOFLAT = CONFIG_INI.getfloat(which_tel, 'flat_to_flat')

    wvln = 638e-9
    lamD = 20
    samp = 4
    norm = False

    fac = 6.55

    # --------------------------------- #
    #aber_rad = 6.2
    aber_array = np.linspace(0, 2*np.pi, 50, True)
    print('Aber in rad: \n{}'.format(aber_array))
    print('Aber in m: \n{}'.format(aber_to_opd(aber_array, wvln)))
    # --------------------------------- #

    ### HCIPy SM

    # HCIPy grids and propagator
    pupil_grid = make_pupil_grid(dims=NPIX, diameter=PUP_DIAMETER)
    focal_grid = make_focal_grid(pupil_grid, samp, lamD, wavelength=wvln)
    prop = FraunhoferPropagator(pupil_grid, focal_grid)

    # Generate an aperture
    aper, seg_pos = atim.get_atlast_aperture(normalized=norm)
    aper = evaluate_supersampled(aper, pupil_grid, 1)

    # Instantiate the segmented mirror
    hsm = atim.SegmentedMirror(aper, seg_pos)

    # Make a pupil plane wavefront from aperture
    wf = Wavefront(aper, wavelength=wvln)

    ### Poppy SM

    psm = poppy.dms.HexSegmentedDeformableMirror(name='Poppy SM',
                                                 rings=3,
                                                 flattoflat=FLATTOFLAT * u.m,
                                                 gap=GAPSIZE * u.m,
                                                 center=False)

    ### Apply pistons
    hc_ims = []
    pop_ims = []
    for aber_rad in aber_array:

        # Flatten both SMs
        hsm.flatten()
        psm.flatten()

        # HCIPy
        for i in [19, 28]:
            hsm.set_segment(i, aber_to_opd(aber_rad, wvln)/2, 0, 0)

        # Poppy
        for i in [34, 25]:
            psm.set_actuator(i, aber_to_opd(aber_rad, wvln) * u.m, 0, 0)  # 34 in poppy is 19 in HCIPy

        ### Propagate to image plane
        ### HCIPy
        # Apply SM to pupil plane wf
        wf_fp_pistoned = hsm(wf)

        # Propagate from SM to image plane
        im_pistoned_hc = prop(wf_fp_pistoned)

        ### Poppy
        # Make an optical system with the Poppy SM and a detector
        osys = poppy.OpticalSystem()
        osys.add_pupil(psm)
        pxscle = 0.0031 * fac  # I'm tweaking pixelscale and fov_arcsec to match the HCIPy image
        fovarc = 0.05 * fac
        osys.add_detector(pixelscale=pxscle, fov_arcsec=fovarc, oversample=10)

        # Calculate the PSF
        psf = osys.calc_psf(wvln)

        # Get the PSF as an array
        im_pistoned_pop = psf[0].data

        hc_ims.append(im_pistoned_hc.intensity.shaped/np.max(im_pistoned_hc.intensity))
        pop_ims.append(im_pistoned_pop/np.max(im_pistoned_pop))

    ### Trying to do it with numbers
    hc_ims = np.array(hc_ims)
    pop_ims = np.array(pop_ims)

    sum_hc = np.sum(hc_ims, axis=(1,2))
    sum_pop = np.sum(pop_ims, axis=(1,2)) - 1.75   # the -1.75 is just there because I didn't bother about image normalization too much

    plt.suptitle('Image degradation of SMs')
    plt.plot(aber_array, sum_hc, label='HCIPy SM')
    plt.plot(aber_array, sum_pop, label='Poppy SM')
    plt.xlabel('rad')
    plt.ylabel('image sum')
    plt.legend()
    plt.show()
