"""
Helper functions for PASTIS.
"""

import os
import datetime
import time
from shutil import copy
import sys
from astropy.io import fits
import astropy.units as u
import logging
import logging.handlers
import numpy as np

from config import CONFIG_INI

log = logging.getLogger()


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


def write_all_fits_to_cube(path):
    """
    Write all fits files in a directory to an image cube.

    Directory can *only* contain fits files, and only files that you want in the cube. Subdirectories will be ignored.
    :param path: string, path to directory that contains all fits files that should be put into cube; cube gets saved
                 into that same directory
    """
    # Collect all filenames
    all_file_names = [fname for fname in os.listdir(path) if os.path.isfile(os.path.join(path, fname))]
    # Read all files into list
    allfiles = []
    for fname in all_file_names:
        allfiles.append(fits.getdata(os.path.join(path, fname)))
    cube = np.array(allfiles)
    write_fits(cube, os.path.join(path, 'psf_cube.fits'))


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
    return im[int(y - bb):int(y + bb), int(x - bb):int(x + bb)]


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
    :return: dh_area: np.array
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
    :param im: array, normalized (by direct PSF peak pixel) image
    :param dh: array, dark hole mask
    """
    darkh = im * dh
    con = np.mean(darkh[np.where(dh != 0)])
    return con


@u.quantity_input(aber=u.nm)
def pastis_contrast(aber, matrix_pastis):
    """
    Calculate the contrast with PASTIS matrix model.
    :param aber: aberration vector, its length is number of segments, WFE aberration coefficients in NANOMETERS
    :param matrix_pastis: PASTIS matrix, in contrast/nm^2
    :return:
    """
    result = np.matmul(np.matmul(aber, matrix_pastis), aber)
    return result.value


def calc_statistical_mean_contrast(pastismatrix, cov_segments, coro_floor):
    """
    Analytically calculate the *statistical* mean contrast for a set of segment requirements.
    :param pastismatrix: array, PASTIS matrix [segs, modes]
    :param cov_segments: array, segment-space covariance matrix Ca
    :param coro_floor: float, coronagrpah contrast in absence of aberrations
    :return: mean_c_stat, float
    """
    mean_c_stat = np.trace(np.matmul(cov_segments, pastismatrix)) + coro_floor
    return mean_c_stat


def calc_variance_of_mean_contrast(pastismatrix, cov_segments):
    """
    Analytically calculate the variance of the *statistical* mean contrast for a set of segment requirements.
    :param pastismatrix: array, PASTIS matrix [segs, modes]
    :param cov_segments: array, segment-space covariance matrix Ca
    :return: var, float
    """
    var = 2 * np.trace(np.matmul(pastismatrix, np.matmul(cov_segments, (np.matmul(pastismatrix, cov_segments)))))
    return var


def get_segment_list(instrument):
    """
    Horribly hacky function to get correct segment numer list for an instrument.

    We can assume that both implemented instruments start their numbering at 0, at the center segment. LUVOIR doesn't
    use the center segment though, so we start at 1 and go until 120, for a total of 120 segments. HiCAT does use it,
    so we start at 0 and go to 36 for a total of 37 segments.
    :param instrument: string, "HiCAT" or "LUVOIR"
    :return: seglist, array of segment numbers (names!)
    """
    seglist = np.arange(CONFIG_INI.getint(instrument, 'nb_subapertures'))
    if instrument == 'LUVOIR':
        seglist += 1

    return seglist


def read_continuous_dm_maps_hicat(path_to_dm_maps):
    """
    Read Boston DM maps from disk and return as one list per DM.
    :param path_to_dm_maps: string, absolute path to folder containing DM maps to load
    :return: DM1 actuator map array, DM2 actuator map array; in m
    """

    surfaces = []
    for dmnum in [1, 2]:
        actuators_2d = fits.getdata(os.path.join(path_to_dm_maps, f'dm{dmnum}_command_2d_noflat.fits'))
        surfaces.append(actuators_2d)

    return surfaces[0], surfaces[1]


def rms(ar):
    """
    Manual root-mean-square calculation, assuming a zero-mean
    :param ar: quantity to calculate the rms for
    :return:
    """
    rms = np.sqrt(np.mean(np.square(ar)) - np.square(np.mean(ar)))
    return rms


def aber_to_opd(aber_rad, wvln):
    """
    Convert phase aberration in rad to OPD in meters.
    :param aber_rad: float, phase aberration in radians
    :param wvln: float, wavelength
    :return: aber_m: float, OPD in meters
    """
    aber_m = aber_rad * wvln / (2 * np.pi)
    return aber_m


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
    """
    Get the name of the Zernike with input index in input framework (Noll or WSS).
    :param index: int, Zernike index
    :param framework: str, 'Noll' or 'WSS' for Zernike ordering framework
    :return zern_name: str, name of the Zernike in the chosen framework
    """
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

    :param index: int, Zernike mode index
    :param framework: 'Noll' or 'WSS' for Zernike ordering framework, default is Noll
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


def create_data_path(initial_path, telescope="", suffix=""):
    """
    Will create a timestamp and join it to the output_path found in the INI.
    :param initial_path: str, output directory as defined in the configfile
    :param telescope: str, telescope name that gets added to data path name; somewhat redundant with suffix
    :param suffix: str, appends this to the end of the timestamp (ex: 2017-06-15T121212_suffix), also read from config
    :return: A path with the final folder containing a timestamp of the current datetime.
    """

    # Create a string representation of the current timestamp.
    time_stamp = time.time()
    date_time_string = datetime.datetime.fromtimestamp(time_stamp).strftime("%Y-%m-%dT%H-%M-%S")

    if suffix != "":
        suffix = "_" + suffix
    if telescope != "":
        telescope = "_" + telescope

    # Return the full path.
    print(initial_path)
    print(suffix)
    full_path = os.path.join(initial_path, date_time_string + telescope + suffix)
    return full_path


def copy_config(outdir):
    """
    Copy the config_local, or if non-existent, config.ini to outdir
    :param outdir: string, target location of copied configfile
    :return: 
    """
    print('Saving the configfile to outputs folder.')
    try:
        copy('config_local.ini', outdir)
    except IOError:
        copy('config.ini', outdir)


def setup_pastis_logging(experiment_path, name):
    ### General logger
    log = logging.getLogger()

    log.setLevel(logging.NOTSET)    # set logger to pass all messages, then filter in handlers
    suffix = ".log"
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ### Set up the console log to stdout (not stderr since some messages are info)
    consoleHandler = logging.StreamHandler(sys.stdout)

    # add formatter, set logging level and add the handler
    consoleHandler.setFormatter(formatter)
    console_level = logging.INFO
    consoleHandler.setLevel(console_level)

    log.addHandler(consoleHandler)
    log.info("LOG SETUP: Console will display messages of {} or higher".format(logging.getLevelName(consoleHandler.level)))

    ### Set up the experiment log
    experiment_logfile_path = os.path.join(experiment_path, name + suffix)
    experiment_hander = logging.handlers.WatchedFileHandler(experiment_logfile_path)

    # add formatter, set logging level and add the handler
    experiment_hander.setFormatter(formatter)
    experiment_level = logging.INFO
    experiment_hander.setLevel(experiment_level)

    log.addHandler(experiment_hander)
    log.info("LOG SETUP: Experiment log will save messages of {} or higher to {}".format(logging.getLevelName(experiment_hander.level),
                                                                                         experiment_logfile_path))