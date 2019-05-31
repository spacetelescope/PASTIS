"""
Helper functions for PASTIS.
"""

import os
import numpy as np
from astropy.io import fits
import astropy.units as u


def write_fits(data, filepath, header=None, metadata=None):
    """
    Writes a fits file and adds header and metadata when necessary.
    :param data: numpy data (aka image)
    :param filepath: path to save the file, include filename.
    :param header: astropy hdu.header.
    :param metadata: list of MetaDataEntry objects that will get added to header.
    :return: filepath
    """
    # Make sure file ends with fit or fits.
    #if not (filepath.endswith(".fit") or filepath.endswith(".fits")):
    #    filepath += ".fits"

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    # Create a PrimaryHDU object to encapsulate the data.
    hdu = fits.PrimaryHDU(data)
    if header is not None:
        hdu.header = header

    # Add metadata to header.
    if metadata is not None:
        for entry in metadata:
            if len(entry.name_8chars) > 8:
                print('Fits Header Keyword: ' + entry.name_8chars +
                      ' is greater than 8 characters and will be truncated.')
            if len(entry.comment) > 47:
                print('Fits Header comment for ' + entry.name_8chars +
                      ' is greater than 47 characters and will be truncated.')
            hdu.header[entry.name_8chars[:8]] = (entry.value, entry.comment)

    # Create a HDUList to contain the newly created primary HDU, and write to a new file.
    fits.HDUList([hdu])
    hdu.writeto(filepath, overwrite=True)

    #print('Wrote ' + filepath)
    return filepath


def circle_mask(im, xc, yc, rcirc):
    """ Create a circle on array im centered on xc, yc with radius rcirc; inside circle equals 1."""
    x, y = np.shape(im)
    newy, newx = np.mgrid[0:y,0:x]
    circ = (newx-xc)**2 + (newy-yc)**2 < rcirc**2
    return circ


def zoom_point(im, x, y, bb):
    """
    Cut out a square box from image im centered on (x,y) with half-box size bb.
    :param im: image from which box will be taken
    :param x: x coordinate of center of box
    :param y: y coordinate of center of box
    :param bb: half-box size
    :return:
    """
    return (im[int(y - bb):int(y + bb), int(x - bb):int(x + bb)])


def zoom_cen(im, bb):
    """
    Cut out a square box from the image center with half-box size bb.
    :param im: image from which box will be taken
    :param bb: half-box size
    :return:
    """
    x = int(im.shape[1]/2)
    y = int(im.shape[0]/2)
    return im[int(y-bb):int(y+bb), int(x-bb):int(x+bb)]


def FFT(ef):
    """Do the numpy Fourier transform on complex array 'ef', together with all the shifting needed."""
    FFT_E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ef)))
    return FFT_E


def IFFT(ef):
    """Do the numpy inverse Fourier transform on complex array 'ef', together with all the shifting needed."""
    IFFT_E = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ef)))
    return IFFT_E


def create_dark_hole(pup_im, iwa, owa, samp):
    """
    Create a dark hole on pupil image pup_im.
    :param pup_im: np.array of pupil image
    :param iwa: inner working angle in lambda/D
    :param owa: outer working angle in lambda/D
    :param samp: sampling factor
    :return: dh_area, np.array
    """
    circ_inner = circle_mask(pup_im, pup_im.shape[0]/2., pup_im.shape[1]/2., iwa * samp) * 1   # *1 converts from booleans to integers
    circ_outer = circle_mask(pup_im, pup_im.shape[0]/2., pup_im.shape[1]/2., owa * samp) * 1
    dh_area = circ_outer - circ_inner

    return dh_area


def dh_mean(im, dh):
    """
    Return the dark hole contrast.

    Calculate the mean intensity in the dark hole area dh of the image im.
    im and dh have to have the same array size and shape.
    """
    darkh = im * dh
    con = np.mean(darkh[np.where(darkh != 0)])
    return con


@u.quantity_input(aber=u.nm)
def pastis_contrast(aber, matrix_pastis):
    """
    Calculate the contrast with PASTIS matrix model.
    :param aber: aberration vector, its length is number of segments, aberration coefficients in NANOMETERS
    :param matrix_pastis: PASTIS matrix
    :return:
    """
    result = np.matmul(np.matmul(aber, matrix_pastis), aber)
    return result#.value   # Had to comment this out for luvoir_hockeystick.py


def rms(ar):
    rms = np.sqrt(np.mean(np.square(ar)) - np.square(np.mean(ar)))
    return rms


def noll_to_wss(zern):
    """
    Transform a Noll Zernike index into a JWST WSS framework Zernike index.
    :param zern: int; Noll Zernike index
    :return: WSS Zernike index
    """
    noll = {1: 'piston', 2: 'tip', 3: 'tilt', 4: 'defocus', 5: 'astig45', 6: 'astig0', 7: 'ycoma', 8: 'xcoma',
            9: 'ytrefoil', 10: 'xtrefoil', 11: 'spherical'}
    wss = {'piston': 1, 'tip': 2, 'tilt': 3, 'defocus': 5, 'astig45': 4, 'astig0': 6, 'ycoma': 8, 'xcoma': 7,
            'ytrefoil': 10, 'xtrefoil': 11, 'spherical': 9}
    wss_ind = wss[noll[zern]]

    return wss_ind


def wss_to_noll(zern):
    """
    Transform a JWST WSS framework Zernike index into a Noll Zernike index.
    :param zern: int; WSS Zernike index
    :return: Noll Zernike index
    """
    noll = {'piston': 1, 'tip': 2, 'tilt': 3, 'defocus': 4, 'astig45': 5, 'astig0': 6, 'ycoma': 7, 'xcoma': 8,
            'ytrefoil': 9, 'xtrefoil': 10, 'spherical': 11}
    wss = {1: 'piston', 2: 'tip', 3: 'tilt', 5: 'defocus', 4: 'astig45', 6: 'astig0', 8: 'ycoma', 7: 'xcoma',
            10: 'ytrefoil', 11: 'xtrefoil', 9: 'spherical'}
    noll_ind = noll[wss[zern]]

    return noll_ind


def zernike_name(index, framework='Noll'):
    """Get the name of the Zernike with input index in inpit framework (Noll or WSS)."""
    noll_names = {1: 'piston', 2: 'tip', 3: 'tilt', 4: 'defocus', 5: 'astig45', 6: 'astig0', 7: 'ycoma', 8: 'xcoma',
                  9: 'ytrefoil', 10: 'xtrefoil', 11: 'spherical'}
    wss_names = {1: 'piston', 2: 'tip', 3: 'tilt', 5: 'defocus', 4: 'astig45', 6: 'astig0', 8: 'ycoma', 7: 'xcoma',
                 10: 'ytrefoil', 11: 'xtrefoil', 9: 'spherical'}

    if framework == 'Noll':
        zern_name = noll_names[index]
    elif framework == 'WSS':
        zern_name = wss_names[index]
    else:
        raise ValueError('No known Zernike convention passed.')

    return zern_name


class ZernikeMode:
    """
    A Zernike mode with Zernike mode index, name of mode and name of ordering convention.

    It can use framework = 'Noll' or 'WSS' and an index = between 1 and 11.
    It initializes with framework = 'Noll', but needs an index given.
    If you change ZernikeMode.convention directly, you screw things up... there are methods to change naming convention
    which are capable of changing everything that comes with that.
    """

    def __init__(self, index, framework='Noll'):
        self.convention = framework
        self.index = index

    def get_info(self):
        """Prints full Zernike mode info."""
        print('This is Zernike mode', self.index, 'in the', self.convention, 'convention, which is:', self.name)

    def change_to_wss(self):
        """Change form Noll to WSS Zernike index."""
        if self.convention == 'WSS':
            print('You are already in the WSS convention!')
        else:
            self.index = noll_to_wss(self.index)
            self.convention = 'WSS'

    def change_to_noll(self):
        """Change from WSS to Noll Zernike index."""
        if self.convention == 'Noll':
            print('You are already in the Noll convention!')
        else:
            self.index = wss_to_noll(self.index)
            self.convention = 'Noll'

    @property
    # this property needs some fixing
    def name(self):
        zern_name = zernike_name(self.index, self.convention)
        return zern_name
