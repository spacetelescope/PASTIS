"""
This is a module containing convenience functions to create the JWST aperture and coronagraphic images with WebbPSF.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import logging
import poppy

from pastis.config import CONFIG_PASTIS
import pastis.util as util

log = logging.getLogger()

try:
    import webbpsf
except ImportError:
    log.info('WebbPSF was not imported.')

# Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
# webbpsf.utils.get_webbpsf_data_path()
# --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
os.environ['WEBBPSF_PATH'] = CONFIG_PASTIS.get('local', 'webbpsf_data_path')


NB_SEG = CONFIG_PASTIS.getint('JWST', 'nb_subapertures')
FLAT_TO_FLAT = CONFIG_PASTIS.getfloat('JWST', 'flat_to_flat')
WVLN = CONFIG_PASTIS.getfloat('JWST', 'lambda') * u.nm
IM_SIZE_PUPIL = CONFIG_PASTIS.getint('numerical', 'tel_size_px')
FLAT_DIAM = CONFIG_PASTIS.getfloat('JWST', 'flat_diameter') * u.m
WSS_SEGS = webbpsf.constants.SEGNAMES_WSS_ORDER
IM_SIZE_E2E = CONFIG_PASTIS.getint('numerical', 'im_size_px_webbpsf')


def get_jwst_coords(outDir):

    #-# Generate the pupil with segments and spiders

    # Use poppy to create JWST aperture without spiders
    log.info('Creating and saving aperture')
    jwst_pup = poppy.MultiHexagonAperture(rings=2, flattoflat=FLAT_TO_FLAT)   # Create JWST pupil without spiders
    jwst_pup.display(colorbar=False)   # Show pupil (will be saved to file)
    plt.title('JWST telescope pupil')
    # Number the segments
    for i in range(NB_SEG+1):
        ycen, xcen = jwst_pup._hex_center(i)
        plt.annotate(str(i), size='x-large', xy=(xcen-0.1, ycen-0.1))   # -0.1 is for shifting the numbers closer to the segment centers
    # Save a PDF version of the pupil
    plt.savefig(os.path.join(outDir, 'JWST_aperture.pdf'))

    # Since WebbPSF creates images by controlling the exit pupil,
    # let's also create the exit pupil instead of the entrance pupil.
    # I do this by flipping the y-coordinates of the segments.
    plt.clf()
    jwst_pup.display(colorbar=False)   # Show pupil
    plt.title('JWST telescope exit pupil')
    # Number the segments
    for i in range(NB_SEG+1):
        ycen, xcen = jwst_pup._hex_center(i)
        ycen *= -1
        plt.annotate(str(i), size='x-large', xy=(xcen-0.1, ycen-0.1))   # -0.1 is for shifting the number labels closer to the segment centers
    # Save a PDF version of the exit pupil
    plt.savefig(os.path.join(outDir, 'JWST_exit_pupil.pdf'))

    # Get pupil as fits image
    pupil_dir = jwst_pup.sample(wavelength=WVLN, npix=IM_SIZE_PUPIL, grid_size=FLAT_DIAM, return_scale=True)
    # If the image size is equivalent to the total diameter of the telescope, we don't have to worry about sampling later
    # But for the JWST case with poppy it makes such a small difference that I am skipping it for now
    util.write_fits(pupil_dir[0], os.path.join(outDir, 'pupil.fits'))

    #-# Get the coordinates of the central pixel of each segment
    seg_position = np.zeros((NB_SEG, 2))   # holds x and y position of each central pixel
    for i in range(NB_SEG+1):   # our pupil is still counting the central segment as seg 0, so we need to include it
                                # in the loop, however, we will just discard the values for the center
        if i == 0:     # Segment 0 is the central segment, which we want to skip and not put into seg_position
            continue   # Continues with the next iteration of the loop
        else:
            seg_position[i-1, 1], seg_position[i-1, 0] = jwst_pup._hex_center(i)   # y, x = center position
            seg_position[i - 1, 1] *= -1       # inverting the y-axis because we want to work with the EXIT PUPIL!!!
            # Units are meters!!!

    return seg_position


def nircam_coro(filter, fpm, ppm, Aber_WSS):
    """
    -- Deprecated function still used in analytical PASTIS and some notebooks. --

    Create NIRCam image with specified filter and coronagraph, and aberration input.
    :param filter: str, filter name
    :param fpm: focal plane mask
    :param ppm: pupil plane mask - Lyot stop
    :param Aber_WSS: list or array holding Zernike coefficients ordered in WSS convention and in METERS
    :return:
    """

    # Set up NIRCam and coronagraph
    nc = webbpsf.NIRCam()
    nc.filter = filter
    nc.image_mask = fpm
    nc.pupil_mask = ppm

    # Adjust OTE with aberrations
    nc, ote = webbpsf.enable_adjustable_ote(nc)
    nc.include_si_wfe = False  # set SI internal WFE to zero
    ote.reset()
    ote.zero()
    for i in range(NB_SEG):
        seg = WSS_SEGS[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(IM_SIZE_E2E), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf


def nircam_nocoro(filter, Aber_WSS):
    """
    -- Deprecated function still used in analytical PASTIS and some notebooks. --
    :param filter:
    :param Aber_WSS:
    :return:
    """
    # Create NIRCam object
    nc = webbpsf.NIRCam()
    # Set filter
    nc.filter = filter

    # Adjust OTE with aberrations
    nc, ote = webbpsf.enable_adjustable_ote(nc)
    nc.include_si_wfe = False  # set SI internal WFE to zero
    ote.reset()
    ote.zero()
    for i in range(NB_SEG):
        seg = WSS_SEGS[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(IM_SIZE_E2E), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf


def set_up_nircam():
    """
    Return a configured instance of the NIRCam simulator on JWST.

    Sets up the Lyots stop and filter from the configfile, turns of science insturment (SI) internal WFE and zeros
    the OTE.
    :return: Tuple of NIRCam instance, and its OTE
    """

    nircam = webbpsf.NIRCam()
    nircam.include_si_wfe = False
    nircam.filter = CONFIG_PASTIS.get('JWST', 'filter_name')
    nircam.pupil_mask = CONFIG_PASTIS.get('JWST', 'pupil_plane_stop')

    nircam, ote = webbpsf.enable_adjustable_ote(nircam)
    ote.zero()

    return nircam, ote


if __name__ == '__main__':

    pass