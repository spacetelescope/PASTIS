"""
This script applies the analytical model in the case of one single Zernike polynomial
on the segments.
We're following the Noll convention starting with index 0
0: piston, 1: tip, 2: tilt, 3: defocus, 4: 45°-astigmatism, and so on...
"""

import os
import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import poppy.zernike as zern
import poppy.matrixDFT as mft
import poppy
import hcipy

from pastis.config import CONFIG_INI
import pastis.util_pastis as util

log = logging.getLogger()


@u.quantity_input(coef=u.nm)
def analytical_model(zernike_pol, coef, cali=False):
    """

    :param zernike_pol:
    :param coef:
    :param cali: bool; True if we already have calibration coefficients to use. False if we still need to create them.
    :return:
    """

    #-# Parameters
    dataDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'active')
    telescope = CONFIG_INI.get('telescope', 'name')
    nb_seg = CONFIG_INI.getint(telescope, 'nb_subapertures')
    tel_size_m = CONFIG_INI.getfloat(telescope, 'diameter') * u.m
    real_size_seg = CONFIG_INI.getfloat(telescope, 'flat_to_flat')     # in m, size in meters of an individual segment flatl to flat
    size_seg = CONFIG_INI.getint('numerical', 'size_seg')              # pixel size of an individual segment tip to tip
    wvln = CONFIG_INI.getint(telescope, 'lambda') * u.nm
    inner_wa = CONFIG_INI.getint(telescope, 'IWA')
    outer_wa = CONFIG_INI.getint(telescope, 'OWA')
    tel_size_px = CONFIG_INI.getint('numerical', 'tel_size_px')        # pupil diameter of telescope in pixels
    im_size_pastis = CONFIG_INI.getint('numerical', 'im_size_px_pastis')             # image array size in px
    sampling = CONFIG_INI.getfloat(telescope, 'sampling')            # sampling
    size_px_tel = tel_size_m / tel_size_px                             # size of one pixel in pupil plane in m
    px_sq_to_rad = (size_px_tel * np.pi / tel_size_m) * u.rad
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')
    sz = CONFIG_INI.getint('ATLAST', 'im_size_lamD_hcipy')              # image size in lam/D, only used in ATLAST case

    # Create Zernike mode object for easier handling
    zern_mode = util.ZernikeMode(zernike_pol)

    #-# Mean subtraction for piston
    if zernike_pol == 1:
        coef -= np.mean(coef)

    #-# Generic segment shapes

    if telescope == 'JWST':
        # Load pupil from file
        pupil = fits.getdata(os.path.join(dataDir, 'segmentation', 'pupil.fits'))

        # Put pupil in randomly picked, slightly larger image array
        pup_im = np.copy(pupil)   # remove if lines below this are active
        #pup_im = np.zeros([tel_size_px, tel_size_px])
        #lim = int((pup_im.shape[1] - pupil.shape[1])/2.)
        #pup_im[lim:-lim, lim:-lim] = pupil
        # test_seg = pupil[394:,197:315]    # this is just so that I can display an individual segment when the pupil is 512
        # test_seg = pupil[:203,392:631]    # ... when the pupil is 1024
        # one_seg = np.zeros_like(test_seg)
        # one_seg[:110, :] = test_seg[8:, :]    # this is the centered version of the individual segment for 512 px pupil

        # Creat a mini-segment (one individual segment from the segmented aperture)
        mini_seg_real = poppy.NgonAperture(name='mini', radius=real_size_seg)   # creating real mini segment shape with poppy
        #test = mini_seg_real.sample(wavelength=wvln, grid_size=flat_diam, return_scale=True)   # fix its sampling with wavelength
        mini_hdu = mini_seg_real.to_fits(wavelength=wvln, npix=size_seg)    # make it a fits file
        mini_seg = mini_hdu[0].data      # extract the image data from the fits file

    elif telescope == 'ATLAST':
        # Create mini-segment
        pupil_grid = hcipy.make_pupil_grid(dims=tel_size_px, diameter=real_size_seg)
        focal_grid = hcipy.make_focal_grid(pupil_grid, sampling, sz, wavelength=wvln.to(u.m).value)       # fov = lambda/D radius of total image
        prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

        mini_seg_real = hcipy.hexagonal_aperture(circum_diameter=real_size_seg, angle=np.pi/2)
        mini_seg_hc = hcipy.evaluate_supersampled(mini_seg_real, pupil_grid, 4)  # the supersampling number doesn't really matter in context with the other numbers
        mini_seg = mini_seg_hc.shaped    # make it a 2D array

        # Redefine size_seg if using HCIPy
        size_seg = mini_seg.shape[0]

        # Make stand-in pupil for DH array
        pupil = fits.getdata(os.path.join(dataDir, 'segmentation', 'pupil.fits'))
        pup_im = np.copy(pupil)

    #-# Generate a dark hole mask
    #TODO: simplify DH generation and usage
    dh_area = util.create_dark_hole(pup_im, inner_wa, outer_wa, sampling)   # this might become a problem if pupil size is not same like pastis image size. fine for now though.
    if telescope == 'ATLAST':
        dh_sz = util.zoom_cen(dh_area, sz*sampling)

    #-# Import information form segmentation script
    Projection_Matrix = fits.getdata(os.path.join(dataDir, 'segmentation', 'Projection_Matrix.fits'))
    vec_list = fits.getdata(os.path.join(dataDir, 'segmentation', 'vec_list.fits'))                    # in pixels
    NR_pairs_list = fits.getdata(os.path.join(dataDir, 'segmentation', 'NR_pairs_list_int.fits'))

    # Figure out how many NRPs we're dealing with
    NR_pairs_nb = NR_pairs_list.shape[0]

    #-# Chose whether calibration factors to do the calibraiton with
    if cali:
        filename = 'calibration_' + zern_mode.name + '_' + zern_mode.convention + str(zern_mode.index)
        ck = fits.getdata(os.path.join(dataDir, 'calibration', filename+'.fits'))
    else:
        ck = np.ones(nb_seg)

    coef = coef * ck

    #-# Generic coefficients
    # the coefficients in front of the non redundant pairs, the A_q in eq. 13 in Leboulleux et al. 2018
    generic_coef = np.zeros(NR_pairs_nb) * u.nm * u.nm    # setting it up with the correct units this will have

    for q in range(NR_pairs_nb):
        for i in range(nb_seg):
            for j in range(i+1, nb_seg):
                if Projection_Matrix[i, j, 0] == q+1:
                    generic_coef[q] += coef[i] * coef[j]

    #-# Constant sum and cosine sum - calculating eq. 13 from Leboulleux et al. 2018
    if telescope == 'JWST':
        i_line = np.linspace(-im_size_pastis/2., im_size_pastis/2., im_size_pastis)
        tab_i, tab_j = np.meshgrid(i_line, i_line)
        cos_u_mat = np.zeros((int(im_size_pastis), int(im_size_pastis), NR_pairs_nb))
    elif telescope == 'ATLAST':
        i_line = np.linspace(-(2 * sz * sampling) / 2., (2 * sz * sampling) / 2., (2 * sz * sampling))
        tab_i, tab_j = np.meshgrid(i_line, i_line)
        cos_u_mat = np.zeros((int((2 * sz * sampling)), int((2 * sz * sampling)), NR_pairs_nb))

    # Calculating the cosine terms from eq. 13.
    # The -1 with each NR_pairs_list is because the segment names are saved starting from 1, but Python starts
    # its indexing at zero, so we have to make it start at zero here too.
    for q in range(NR_pairs_nb):
        # cos(b_q <dot> u): b_q with 1 <= q <= NR_pairs_nb is the basis of NRPS, meaning the distance vectors between
        #                   two segments of one NRP. We can read these out from vec_list.
        #                   u is the position (vector) in the detector plane. Here, those are the grids tab_i and tab_j.
        # We need to calculate the dot product between all b_q and u, so in each iteration (for q), we simply add the
        # x and y component.
        cos_u_mat[:,:,q] = np.cos(px_sq_to_rad * (vec_list[NR_pairs_list[q,0]-1, NR_pairs_list[q,1]-1, 0] * tab_i) +
                                  px_sq_to_rad * (vec_list[NR_pairs_list[q,0]-1, NR_pairs_list[q,1]-1, 1] * tab_j)) * u.dimensionless_unscaled

    sum1 = np.sum(coef**2)   # sum of all a_{k,l} in eq. 13 - this works only for single Zernikes (l fixed), because np.sum would sum over l too, which would be wrong.
    if telescope == 'JWST':
        sum2 = np.zeros((int(im_size_pastis), int(im_size_pastis))) * u.nm * u.nm    # setting it up with the correct units this will have
    elif telescope == 'ATLAST':
        sum2 = np.zeros((int(2 * sz * sampling), int(2 * sz * sampling))) * u.nm * u.nm

    for q in range(NR_pairs_nb):
        sum2 = sum2 + generic_coef[q] * cos_u_mat[:,:,q]

    #-# Local Zernike
    if telescope == 'JWST':
        # Generate a basis of Zernikes with the mini segment being the support
        isolated_zerns = zern.hexike_basis(nterms=zern_max, npix=size_seg, rho=None, theta=None, vertical=False, outside=0.0)

        # Calculate the Zernike that is currently being used and put it on one single subaperture, the result is Zer
        # Apply the currently used Zernike to the mini-segment.
        if zernike_pol == 1:
            Zer = np.copy(mini_seg)
        elif zernike_pol in range(2, zern_max-2):
            Zer = np.copy(mini_seg)
            Zer = Zer * isolated_zerns[zernike_pol-1]

        # Fourier Transform of the Zernike - the global envelope
        mf = mft.MatrixFourierTransform()
        ft_zern = mf.perform(Zer, im_size_pastis/sampling, im_size_pastis)

    elif telescope == 'ATLAST':
        isolated_zerns = hcipy.make_zernike_basis(num_modes=zern_max, D=real_size_seg, grid=pupil_grid, radial_cutoff=False)
        Zer = hcipy.Wavefront(mini_seg_hc * isolated_zerns[zernike_pol - 1], wavelength=wvln.to(u.m).value)

        # Fourier transform the Zernike
        ft_zern = prop(Zer)

    #-# Final image
    if telescope == 'JWST':
        # Generating the final image that will get passed on to the outer scope, I(u) in eq. 13
        intensity = np.abs(ft_zern)**2 * (sum1.value + 2. * sum2.value)
    elif telescope == 'ATLAST':
        intensity = ft_zern.intensity.shaped * (sum1.value + 2. * sum2.value)

    # PASTIS is only valid inside the dark hole, so we cut out only that part
    if telescope == 'JWST':
        tot_dh_im_size = sampling * (outer_wa + 3)
        intensity_zoom = util.zoom_cen(intensity, tot_dh_im_size)       # zoom box is (owa + 3*lambda/D) wide, in terms of lambda/D
        dh_area_zoom = util.zoom_cen(dh_area, tot_dh_im_size)

        dh_psf = dh_area_zoom * intensity_zoom

    elif telescope == 'ATLAST':
        dh_psf = dh_sz * intensity

    """
    # Create plots.
    plt.subplot(1, 3, 1)
    plt.imshow(pupil, origin='lower')
    plt.title('JWST pupil and diameter definition')
    plt.plot([46.5, 464.5], [101.5, 409.5], 'r-')   # show how the diagonal of the pupil is defined

    plt.subplot(1, 3, 2)
    plt.imshow(mini_seg, origin='lower')
    plt.title('JWST individual mini-segment')

    plt.subplot(1, 3, 3)
    plt.imshow(dh_psf, origin='lower')
    plt.title('JWST dark hole')
    plt.show()
    """

    # dh_psf is the image of the dark hole only, the pixels outside of it are zero
    # intensity is the entire final image
    return dh_psf, intensity


if __name__ == '__main__':

    "Testing the uncalibrated analytical model\n"

    ### Define the aberration coeffitients "coef"
    telescope = CONFIG_INI.get('telescope', 'name')
    nb_seg = CONFIG_INI.getint(telescope, 'nb_subapertures')
    zern_max = CONFIG_INI.getint('zernikes', 'max_zern')

    nm_aber = CONFIG_INI.getfloat(telescope, 'calibration_aberration') * u.nm  # [nm] amplitude of aberration
    zern_number = CONFIG_INI.getint('calibration', 'local_zernike')             # Which (Noll) Zernike we are calibrating for
    wss_zern_nb = util.noll_to_wss(zern_number)                           # Convert from Noll to WSS framework

    ### What segmend are we aberrating? ###
    i = 0  # segment 1 --> i=0, seg 2 --> i=1, etc.
    cali = False     # calibrated or not?
    ### ------------------------------- ###

    # Create arrays to hold Zernike aberration coefficients
    Aber_WSS = np.zeros([nb_seg, zern_max])  # The Zernikes here will be filled in the WSS order!!!
    # Because it goes into _apply_hexikes_to_seg().
    Aber_Noll = np.copy(Aber_WSS)  # This is the Noll version for later.

    # Feed the aberration nm_aber into the array position
    # that corresponds to the correct Zernike, but only on segment i
    Aber_WSS[i, wss_zern_nb - 1] = nm_aber.to(u.m).value  # Aberration on the segment we're currenlty working on;
                                                     # convert to meters; -1 on the Zernike because Python starts
                                                     # numbering at 0.
    Aber_Noll[i, zern_number - 1] = nm_aber.value          # Noll version - in input units directly!

    # Make sure the aberration coefficients have correct units
    Aber_WSS *= u.m     # not used here
    Aber_Noll *= u.nm

    # Vector of aberration coefficients takes all segments, but only for the Zernike we currently work with
    coef = Aber_Noll[:, zern_number - 1]

    # Define the (Noll) zernike number
    zernike_pol = zern_number

    log.info('coef: {}'.format(coef))
    log.info('Aberration: {}'.format(nm_aber))
    log.info('On segment: {}'.format(i+1))
    log.info('Zernike (Noll): {}'.format(util.zernike_name(zern_number, framework='Noll')))
    log.info('Zernike (WSS): {}'.format(util.zernike_name(wss_zern_nb, framework='WSS')))
    log.info('Zernike number (Noll): {}'.format(zernike_pol))

    ### Run the analytical model without calibration
    dh_psf, int = analytical_model(zernike_pol, coef, cali=cali)

    plt.figure()
    if cali:
        plt.suptitle("Calibrated")
    else:
        plt.suptitle("NOT calibrated")
    plt.subplot(1, 2, 1)
    plt.imshow(dh_psf, norm=LogNorm())
    plt.title("Dark hole")
    plt.subplot(1, 2, 2)
    plt.imshow(int, norm=LogNorm())
    plt.title("Full image")
    plt.show()
