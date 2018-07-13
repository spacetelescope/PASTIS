"""
This is a module containing functions to create JWST coronagraphic images.
"""
import webbpsf
from python.config import CONFIG_INI

im_size = CONFIG_INI.getint('numerical', 'im_size_px')
fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')  # focal plane mask
lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')  # Lyot stop
filter = CONFIG_INI.get('filter', 'name')


def nircam_coro(filter, fpm, ppm):

    # Set up NIRCam and coronagraph
    nc = webbpsf.NIRCam()
    nc.insert_si_wfe = False
    nc.filter = filter
    nc.image_mask = fpm
    nc.pupil_mask = ppm

    # Calculate PSF
    psf_nc = nc.calc_psf(fov_pixels=int(im_size), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf