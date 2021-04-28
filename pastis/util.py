"""
Helper functions for PASTIS.
"""

import glob
import os
import datetime
import importlib
import itertools
import time
from shutil import copy
import sys
from astropy.io import fits
import astropy.units as u
import fpdf
import logging
import logging.handlers
import numpy as np
from PyPDF2 import PdfFileMerger

from pastis.config import CONFIG_PASTIS
#from config import CONFIG_PASTIS
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
    Horribly hacky function to get correct segment number list for an instrument (LUVOIR, or HiCAT and JWST).

    We can assume that all implemented instruments start their numbering at 0, at the center segment.
    LUVOIR doesn't use the center segment though, so we start at 1 and go until 120, for a total of 120 segments.
    HiCAT does use it, so we start at 0 and go to 36 for a total of 37 segments.
    JWST does not have a center segment, but it uses custom segment names anyway, so we start the numbering with zero,
    at the first segment that is actually controllable (A1).

    :param instrument: string, "HiCAT", "LUVOIR" or "JWST"
    :return: seglist, array of segment numbers (names! at least in LUVOIR and HiCAT case. For JWST, it's the segment indices.)
    """
    if instrument not in ['LUVOIR', 'HiCAT', 'JWST', 'RST']:
        raise ValueError('The instrument you requested is not implemented. Try with "LUVOIR", "HiCAT", "JWST"  or "RST" instead.')

    seglist = np.arange(CONFIG_PASTIS.getint(instrument, 'nb_subapertures'))

    # Drop the center segment with label '0' when working with LUVOIR
    if instrument == 'LUVOIR':
        seglist += 1

    return seglist


def apply_mode_to_luvoir(pmode, luvoir):
    """
    Apply a PASTIS mode to the segmented mirror (SM) and return the propagated wavefront "through" the SM.

    This function first flattens the segmented mirror and then applies all segment coefficients from the input mode
    one by one to the segmented mirror.
    :param pmode: array, a single PASTIS mode [nseg] or any other segment phase map in NANOMETERS
    :param luvoir: LuvoirAPLC
    :return: hcipy.Wavefront of the segmented mirror, hcipy.Wavefront of the detector plane
    """

    # Flatten SM to be sure we have no residual aberrations
    luvoir.flatten()

    # Loop through all segments to put them on the segmented mirror one by one
    for seg, val in enumerate(pmode):
        val *= u.nm  # the LUVOIR modes come out in units of nanometers
        luvoir.set_segment(seg + 1, val.to(u.m).value / 2, 0, 0)  # /2 because this SM works in surface, not OPD

    # Propagate the aperture wavefront through the SM
    psf, planes = luvoir.calc_psf(return_intermediate='efield')

    return planes['seg_mirror'], psf


def segment_pairs_all(nseg):
    """
    Return a generator with all possible segment pairs, including repeating ones.

    E.g. if segments are 0, 1, 2, then the returned pairs will be:
    00, 01, 02, 10, 11, 12, 20, 21, 22
    :param nseg: int, number of segments
    :return:
    """
    return itertools.product(np.arange(nseg), np.arange(nseg))


def segment_pairs_non_repeating(nseg):
    """
    Return a generator with all possible non-repeating segment pairs.

    E.g. if segments are 0, 1, 2, then the returned pairs will be:
    00, 01, 02, 11, 12, 22
    :param nseg: int, number of segments
    :return:
    """
    return itertools.combinations_with_replacement(np.arange(nseg), r=2)


def pastis_matrix_measurements(nseg):
    """
    Calculate the total number of measurements needed for a PASTIS matrix with nseg segments
    :param nseg: int, total number of segments
    :return: int, total number of measurements
    """
    total_number = (nseg**2 + nseg) / 2
    return int(total_number)


def symmetrize(array):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.
    Diagonal values are left untouched.
    :param array: square NumPy array, such that a_ij = 0 or a_ji = 0,
    for i != j.

    Source:
    https://stackoverflow.com/a/2573982/10112569
    """
    return array + array.T - np.diag(array.diagonal())


def read_coro_floor_from_txt(datadir):
    """
    Read the coro floor as float from the output file in the data directory.
    :param datadir: str, path to data directory
    :return: coronagraph floor as float
    """
    with open(os.path.join(datadir, 'coronagraph_floor.txt'), 'r') as file:
        full = file.read()
    return float(full[19:])


def rms(ar):
    """
    Manual root-mean-square calculation, assuming a zero-mean
    :param ar: quantity to calculate the rms for
    :return:
    """
    rms = np.sqrt(np.mean(np.square(ar)) - np.square(np.mean(ar)))
    return rms


def create_random_rms_values(nb_seg, total_rms):
    """
    Calculate a set of random segment aberration values scaled to a total WFE rms (input: total_rms).
    Also subtracts global piston.

    :param nb_seg: int, number of segments in the pupil
    :param rms: float, nm (astropy units) of WFE rms that the aberration array will be scaled to
    :return: aber: array of segment aberration values in nm (astropy units) of WFE rms, scaled to input rms value (total_rms)
    """
    # Create own random state
    rms_random_state = np.random.RandomState()

    # Create random aberration coefficients
    aber = rms_random_state.random_sample([nb_seg])  # piston values in input units
    log.info(f'PISTON ABERRATIONS: {aber}')

    # Normalize to the WFE RMS value I want
    rms_init = rms(aber)
    aber *= total_rms.value / rms_init
    calc_rms = rms(aber) * u.nm
    aber *= u.nm  # making sure the aberration has the correct units
    log.info(f"Calculated WFE RMS: {calc_rms}")

    # Remove global piston
    aber -= np.mean(aber)

    return aber


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
    Copy the config_local, or if non-existent, config_pastis.ini to outdir
    :param outdir: string, target location of copied configfile
    :return: 
    """
    print('Saving the configfile to outputs folder.')
    try:
        copy(os.path.join(find_package_location(), 'config_local.ini'), outdir)
    except IOError:
        copy(os.path.join(find_package_location(), 'config_pastis.ini'), outdir)


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


class PDF(fpdf.FPDF):
    """
    Subclass from fpdf.FPDF to be able to add a header and footer.
    :param instrument: str, 'LUVOIR' or 'HiCAT'
    """
    def __init__(self, instrument):
        super().__init__()
        self.instrument = instrument

    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, self.instrument, 1, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number, on the right
        #self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'R')
        # Date, centered
        self.cell(0, 10, str(datetime.date.today()), 0, 0, 'C')


def create_title_page(instrument, datadir, itemlist):
    """
    Write the title page as PDF to disk.
    :param instrument:  str, 'LUVOIR' or 'HiCAT'
    :param datadir: str, data location
    :param itemlist: list of strings to add to the title page
    :return:
    """

    pdf = PDF(instrument=instrument)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    pdf.cell(w=0, h=15, txt=os.path.basename(os.path.normpath(datadir)), border=0, ln=1, align='L')
    for entry in itemlist:
        pdf.multi_cell(0, 10, entry, 0, 1)
    pdf.output(os.path.join(datadir, 'title_page.pdf'), 'F')


def collect_title_page(datadir, c_target):
    """
    Collect all the items from the data directory and return as list of strings for title page.

    Will skip README if none exists.
    :param datadir:  str, data location
    :param c_target: float, target contrast
    :return:
    """

    # Define what txt file contents you want to include in the title page
    txt_files = [os.path.join(datadir, 'README.txt'),
                 os.path.join(datadir, 'coronagraph_floor.txt'),
                 os.path.join(datadir, 'results', f'statistical_contrast_analytical_{c_target}.txt'),
                 os.path.join(datadir, 'results', f'statistical_contrast_empirical_{c_target}.txt')]
    read_list = []

    # Read all files and add their contents as string to read_list
    for one_file in txt_files:
        try:
            with open(one_file, 'r') as file:
                read_this = file.read()
            read_list.append(read_this)
        except FileNotFoundError:
            log.info(f"No {os.path.basename(os.path.normpath(one_file))} found, won't include in PDF title page.")

    return read_list


def create_pdf_report(datadir, c_target):
    """
    Create and write to disk PDF file with all PDF figures and title page.
    :param datadir: str, data directory
    :param c_target: float, target contrast - beware of formatting differences (usually good: e.g. 1e-07, 1e-10, etc.)
    """

    # The hockey stick plot has a variable filename... need to change that at some point
    try:
        hockey_filename_full_path = glob.glob(os.path.join(datadir, 'results', 'hockeystick*'))[0]
    except IndexError:
        hockey_filename_full_path = 'No hockey stick plot available.'

    # Define in what order the PDFs should be merged
    pdfs = [os.path.join(datadir, 'title_page.pdf'),
            os.path.join(datadir, 'unaberrated_dh.pdf'),
            os.path.join(datadir, 'matrix_numerical', 'pastis_matrix.pdf'),
            os.path.join(datadir, 'results', 'modes', 'pupil_plane', 'modes_piston.pdf'),
            os.path.join(datadir, 'results', 'modes', 'focal_plane', 'modes_piston.pdf'),
            os.path.join(datadir, 'results', f'eigenvalues.pdf'),
            hockey_filename_full_path,
            os.path.join(datadir, 'results', f'mode_requirements_{c_target}_uniform.pdf'),
            os.path.join(datadir, 'results', f'monte_carlo_modes_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'cumulative_contrast_accuracy_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'segment_requirements_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'segment_tolerance_map_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'monte_carlo_segments_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'cov_matrix_segments_Ca_{c_target}_segment-based.pdf'),
            os.path.join(datadir, 'results', f'cov_matrix_modes_Cb_{c_target}_segment-based.pdf'),
            os.path.join(datadir, 'results', f'mode_requirements_{c_target}_segment-based.pdf'),
            os.path.join(datadir, 'results', f'mode_requirements_double_axis_{c_target}_segment-based-vs-uniform.pdf'),
            os.path.join(datadir, 'results', f'contrast_per_mode_{c_target}.pdf'),
            os.path.join(datadir, 'results', f'cumulative_contrast_allocation_{c_target}_segment-based-vs-uniform.pdf')
            ]

    merger = PdfFileMerger()

    for pdf in pdfs:
        try:
            merger.append(pdf)
        except FileNotFoundError:
            log.info(f"{pdf} omitted from full report - it doesn't exist.")

    merger.write(os.path.join(datadir, 'full_report.pdf'))
    merger.close()


def read_mean_and_variance(filename):
    """
    Read one of the custom text files that contain a mean value and variance from a custom text file
    and return the mean and variance.
    :param filename: str, full path including file name of the txt file to be read
    :return: floats, mean and variance
    """

    with open(filename, 'r') as file:
        mean_string = file.readline()
        variance_string = file.readline()

    mean_value = float(mean_string.split(': ')[-1])
    variance_value = float(variance_string.split(': ')[-1])

    return mean_value, variance_value


def find_package_location(package='pastis'):
    """Find absolute path to package location on disk
    Taken from hicat-package/hicat/util.py"""
    return importlib.util.find_spec(package).submodule_search_locations[0]


def find_repo_location(package='pastis'):
    """
    Find absolute path to repository location on disk
    Taken from hicat-package/hicat/util.py
    :param package: string, name of package within the repository whose path we are looking for
    """
    return os.path.abspath(os.path.join(find_package_location(package), os.pardir))


def seg_to_dm_xy(actuator_total, segment):
    """
    Convert single index of DM actuator to x|y DM coordinates. This assumes the actuators to be arranged on a square grid
    actuator_total: int, total number of actuators in each line of the (square) DM
    segment: int, single-index actuator number on the DM, to be converted to x|y coordinate
    """
    actuator_pair_x = segment % actuator_total
    actuator_pair_y = (segment-actuator_pair_x)/actuator_total

    return actuator_pair_x, int(actuator_pair_y)
