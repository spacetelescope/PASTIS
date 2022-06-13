"""
This is a module containing classes for telescope and coronagraph (APLC) combinations for designs coming out
of the SCDA working group.
"""
import logging
import os
import hcipy
import numpy as np

from pastis.e2e_simulators.generic_segmented_telescopes import SegmentedAPLC, load_segment_centers

log = logging.getLogger()


class ScdaAPLC(SegmentedAPLC):
    def __init__(self, input_dir, sampling, diameter, seg_flat_to_flat, wvln, imlamD, aplc_params):
        num_seg = aplc_params['num_seg']
        aper_fname = aplc_params['aper_fname']
        aper_ind_fname = aplc_params['aper_ind_fname']
        apod_fname = aplc_params['apod_fname']
        ls_fname = aplc_params['ls_fname']
        pxsize = aplc_params['pxsize']
        fpm_rad = aplc_params['fpm_rad']
        fpm_px = aplc_params['fpm_px']
        iwa = aplc_params['iwa']
        owa = aplc_params['owa']

        pupil_grid = hcipy.make_pupil_grid(dims=pxsize, diameter=diameter)
        lam_over_d = wvln / diameter

        # Load segmented aperture
        pup_read = hcipy.read_fits(os.path.join(input_dir, aper_fname))
        aperture = hcipy.Field(pup_read.ravel(), pupil_grid)

        # Load indexed segmented aperture
        aper_ind_read = hcipy.read_fits(os.path.join(input_dir, aper_ind_fname))
        aper_ind = hcipy.Field(aper_ind_read.ravel(), pupil_grid)
        seg_pos = load_segment_centers(input_dir, aper_ind_fname, num_seg, diameter)
        seg_diameter_circumscribed = 2 / np.sqrt(3) * seg_flat_to_flat    # m

        # Load apodizer
        apod_read = hcipy.read_fits(os.path.join(input_dir, apod_fname))
        apodizer = hcipy.Field(apod_read.ravel(), pupil_grid)

        # Load Lyot stop
        ls_read = hcipy.read_fits(os.path.join(input_dir, ls_fname))
        lyot_stop = hcipy.Field(ls_read.ravel(), pupil_grid)

        # Create a focal plane mask
        samp_foc = fpm_px / (fpm_rad * 2)
        focal_grid_fpm = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=fpm_rad, wavelength=wvln)
        fpm = 1 - hcipy.circular_aperture(2*fpm_rad * lam_over_d)(focal_grid_fpm)

        # Create a focal plane grid for the detector
        focal_det = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=sampling, num_airy=imlamD, wavelength=wvln)

        super().__init__(apod=apodizer, lyot_stop=lyot_stop, fpm=fpm, fpm_rad=fpm_rad,  iwa=iwa, owa=owa, wvln=wvln,
                         diameter=diameter, aper=aperture, indexed_aper=aper_ind, seg_pos=seg_pos,
                         seg_diameter=seg_diameter_circumscribed, focal_grid=focal_det, sampling=sampling, imlamD=imlamD)


class HexRingAPLC(ScdaAPLC):
    """ Segmented telescope with APLC with varying number of hexagonal segment rings.

    Parameters:
    ----------
    input_dir : string
        Path to input files: apodizer, aperture, indexed aperture, Lyot stop.
    num_rings : integer
        Numer of segment rings.
    sampling : float
        Desired image plane sampling of coronagraphic PSF in pixels per lambda/D.
    """
    def __init__(self, input_dir, num_rings, sampling):
        # Diameter for each hex ring solution as provided by GSFC
        # Taken from https://github.com/spacetelescope/aplc_optimization/blob/6bd5a6ecf46a3cf758c23853367d5b18e6a5a1d7/optimization_launchers/SCDA/do_LUVex_survey.py#L67
        pupil_diameter = {1: 7.9445, 2:7.2617, 3:7.7231, 4:7.1522, 5:5.9941}
        self.num_rings = num_rings
        diameter = pupil_diameter[num_rings]

        #TODO: read these from input file headers?
        pxsize = 1024
        fpm_rad =  3.5
        fpm_px = 150
        iwa = 3.4
        owa = 12
        imlamD = 1.2 * owa

        num_seg = None   #TODO from num_rings
        if num_seg in [1,2]:
            aper_fname = f'TelAp_LUVex_{num_seg:02d}-Hex_gy_ovsamp04__N{pxsize:04d}.fits'
        else:
            aper_fname = f'TelAp_LUVex_{num_seg:02d}-Hex_gy_clipped_ovsamp04__N{pxsize:04d}.fits'
        aper_ind_fname = aper_fname.split('.')[0] + '_indexed.fits'
        apod_fname = None   #TODO
        ls_fname = f'LS_LUVex_{num_rings:02d}-Hex_ID0000_OD0982_no_struts_gy_ovsamp04_N{pxsize:04d}.fits'
        seg_flat_to_flat = None   #TODO from num_rings, maybe dict if corners cut off

        aplc_params = {'num_seg': num_seg,
                       'aper_fname': aper_fname,
                       'aper_ind_fname': aper_ind_fname,
                       'apod_fname': apod_fname,
                       'ls_fname': ls_fname,
                       'pxsize': pxsize,
                       'fpm_rad': fpm_rad,
                       'fpm_px': fpm_px,
                       'iwa': iwa,
                       'owa': owa}

        super().__init__(input_dir=input_dir, sampling=sampling, diameter=diameter, seg_flat_to_flat=seg_flat_to_flat,
                         wvln=1, imlamD=imlamD, aplc_params=aplc_params)
