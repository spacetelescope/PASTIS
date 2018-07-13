"""
This is a module containing functions to create JWST coronagraphic images.
"""
import os
import webbpsf
from python.config import CONFIG_INI

# Setting to ensure that PyCharm finds the webbpsf-data folder. If you don't know where it is, find it with:
# webbpsf.utils.get_webbpsf_data_path()
# --> e.g.: >>source activate astroconda   >>ipython   >>import webbpsf   >>webbpsf.utils.get_webbpsf_data_path()
os.environ['WEBBPSF_PATH'] = CONFIG_INI.get('local', 'webbpsf_data_path')


nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')
wss_segs = webbpsf.constants.SEGNAMES_WSS_ORDER
im_size = CONFIG_INI.getint('numerical', 'im_size_px')
fpm = CONFIG_INI.get('coronagraph', 'focal_plane_mask')  # focal plane mask
lyot_stop = CONFIG_INI.get('coronagraph', 'pupil_plane_stop')  # Lyot stop
filter = CONFIG_INI.get('filter', 'name')


def nircam_coro(filter, fpm, ppm, Aber_WSS):
    """
    Create NIRCam image with specified filter and coronagraph, and aberration input.
    :param filter: str, filter name
    :param fpm: focal plane mask
    :param ppm: pupil plane mask - Lyot stop
    :param Aber_WSS: list or array holding Zernike coefficients ordered in WSS convention
    :return:
    """

    # Set up NIRCam and coronagraph
    nc = webbpsf.NIRCam()
    nc.insert_si_wfe = False
    nc.filter = filter
    nc.image_mask = fpm
    nc.pupil_mask = ppm

    # Adjust OTE with aberrations
    nc, ote = webbpsf.enable_adjustable_ote(nc)
    for i in range(nb_seg):
        seg = wss_segs[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(fov_pixels=int(im_size), nlambda=1)
    psf_webbpsf = psf_nc[1].data

    return psf_webbpsf