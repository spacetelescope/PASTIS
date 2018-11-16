"""
Helper functions for PASTIS.
"""

import os
import numpy as np
from astropy.io import fits


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


def zoom(im, x, y, bb):
    """
    Cut out a square box from image im centered on (x,y) with half-box size bb.
    :param im: image from which box will be taken
    :param x: x coordinate of center of box
    :param y: y coordinate of center of box
    :param bb: half-box size
    :return:
    """
    return(im[y-bb:y+bb, x-bb:x+bb])


def FFT(ef):
    """Do the numpy Fourier transform on complex array 'ef', together with all the shifting needed."""
    FFT_E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ef)))
    return FFT_E


def IFFT(ef):
    """Do the numpy inverse Fourier transform on complex array 'ef', together with all the shifting needed."""
    IFFT_E = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ef)))
    return IFFT_E


def matrix_fourier(im, param, inverse=False, dim_tf=None):
    """
    Calculate the Matrix Fourier Transform MTF.

    Translated directly form the ONERA IDL scsript mtf.pro by bpaul.
    :param im: array with dimensions na x na of which we want to calculate the Fourier transform
    :param param: image size in px over sampling (na/samp)
    :param inverse: if True, inverse FT will beb calculated; default is False
    :param dim_tf: optional, size of the output Fourier transform array. Without it, the FT will have same dimensions like input im
    :return:
    """
    na = im.shape[0]
    nb = dim_tf
    if dim_tf == None:
        dim_tf = na

    # Coordinate grids in real space
    xx = ((np.arange(int(na)) + 0.5) - na/2.) / na
    xx = np.expand_dims(xx, axis=0)
    yy = np.copy(xx)

    # Coordinate grids in Fourier space
    uu = ((np.arange(int(nb)) + 0.5) - nb/2.) * param / nb
    uu = np.expand_dims(uu, axis=0)
    vv = np.copy(uu)

    # Adjust sign in FT to whether you want the inverse FT or not
    if inverse:
        sign = -1
    else:
        sign = 1

    # Dissect the matrix multiplications so that it's easier to deal with them
    expo1 = np.matmul(np.transpose(yy), vv)
    expo2 = np.matmul(np.transpose(uu), xx)

    squash = sign * 2.*1j*np.pi
    transform = (param / (na*nb)) + np.matmul(np.exp(squash * expo2), np.matmul(im, np.exp(squash * expo1)))

    return transform


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


def pastis_contrast(Aber, matrix_pastis):
    """
    Calculate the contrast with PASTIS matrix model.
    :param Aber: aberration vector, its length is number of segments, aberration coeffitiones in nm
    :param matrix_pastis: PASTIS matrix
    :return:
    """
    result = np.matmul(np.matmul(Aber, matrix_pastis), Aber)
    return result


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
