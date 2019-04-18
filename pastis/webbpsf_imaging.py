"""
This is a module containing convenience functions to create the JWST aperture and coronagraphic images with WebbPSF.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import poppy
import webbpsf

from config import CONFIG_INI
import util_pastis as util

# Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
# webbpsf.utils.get_webbpsf_data_path()
# --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')


which_tel = CONFIG_INI.get('telescope', 'name')
nb_seg = CONFIG_INI.getint(which_tel, 'nb_subapertures')
flat_to_flat = CONFIG_INI.getfloat(which_tel, 'flat_to_flat')
wvl = CONFIG_INI.getfloat(which_tel, 'lambda') * u.nm
im_size_pupil = CONFIG_INI.getint('numerical', 'tel_size_px')
flat_diam = CONFIG_INI.getfloat(which_tel, 'flat_diameter') * u.m
wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
im_size_e2e = CONFIG_INI.getint('numerical', 'im_size_px_webbpsf')
fpm = CONFIG_INI.get(which_tel, 'focal_plane_mask')  # focal plane mask
lyot_stop = CONFIG_INI.get(which_tel, 'pupil_plane_stop')  # Lyot stop
filter = CONFIG_INI.get(which_tel, 'filter_name')


def get_jwst_coords(outDir):

    #-# Generate the pupil with segments and spiders

    # Use poppy to create JWST aperture without spiders
    print('Creating and saving aperture')
    jwst_pup = poppy.MultiHexagonAperture(rings=2, flattoflat=flat_to_flat)   # Create JWST pupil without spiders
    jwst_pup.display(colorbar=False)   # Show pupil (will be saved to file)
    plt.title('JWST telescope pupil')
    # Number the segments
    for i in range(nb_seg+1):
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
    for i in range(nb_seg+1):
        ycen, xcen = jwst_pup._hex_center(i)
        ycen *= -1
        plt.annotate(str(i), size='x-large', xy=(xcen-0.1, ycen-0.1))   # -0.1 is for shifting the number labels closer to the segment centers
    # Save a PDF version of the exit pupil
    plt.savefig(os.path.join(outDir, 'JWST_exit_pupil.pdf'))

    # Get pupil as fits image
    pupil_dir = jwst_pup.sample(wavelength=wvl, npix=im_size_pupil, grid_size=flat_diam, return_scale=True)
    # If the image size is equivalent to the total diameter of the telescope, we don't have to worry about sampling later
    # But for the JWST case with poppy it makes such a small difference that I am skipping it for now
    util.write_fits(pupil_dir[0], os.path.join(outDir, 'pupil.fits'))

    #-# Get the coordinates of the central pixel of each segment
    seg_position = np.zeros((nb_seg, 2))   # holds x and y position of each central pixel
    for i in range(nb_seg+1):   # our pupil is still counting the central segment as seg 0, so we need to include it
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
    for i in range(nb_seg):
        seg = wss_segs[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(im_size_e2e), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf


def nircam_nocoro(filter, Aber_WSS):
    """
    Create PSF
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
    for i in range(nb_seg):
        seg = wss_segs[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(im_size_e2e), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf


def setup_coro(filter, fpm, ppm):
    """
    Set up a NIRCam coronagraph object.
    :param filter: str, filter name
    :param fpm: focal plane mask
    :param ppm: pupil plane mask - Lyot stop
    :return:
    """
    nc = webbpsf.NIRCam()
    nc.filter = filter
    nc.image_mask = fpm
    nc.pupil_mask = ppm

    return nc


if __name__ == '__main__':

    nc_coro = setup_coro('F335M', 'MASK335R', 'CIRCLYOT')
    nc_coro, ote_coro = webbpsf.enable_adjustable_ote(nc_coro)

    ote_coro.zero()
    #ote_coro._apply_hexikes_to_seg('A1', [1e-6])
    #ote_coro._apply_hexikes_to_seg('A3', [1e-6])
    #ote_coro.move_seg_local('A6', xtilt=0.5)
    psf = nc_coro.calc_psf(oversample=1)
    psf = psf[1].data

    plt.subplot(1,2,1)
    ote_coro.display_opd()
    plt.subplot(1,2,2)
    plt.imshow(psf, norm=LogNorm())
    plt.show()