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
    Calculate the Matrix FOurier Transform MTF.

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


def create_dark_hole():
    pass