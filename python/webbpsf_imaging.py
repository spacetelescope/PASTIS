"""
This is a module containing convenience functions to create JWST coronagraphic images.
"""
import os
import webbpsf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
    ote.zero()
    for i in range(nb_seg):
        seg = wss_segs[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(im_size), nlambda=1)
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
    ote.zero()
    for i in range(nb_seg):
        seg = wss_segs[i].split('-')[0]
        ote._apply_hexikes_to_seg(seg, Aber_WSS[i,:])

    # Calculate PSF
    psf_nc = nc.calc_psf(oversample=1, fov_pixels=int(im_size), nlambda=1)
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