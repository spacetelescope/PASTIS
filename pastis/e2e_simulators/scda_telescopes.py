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
        Path to directories sorted by number of rings that contain the input files: apodizer, aperture, indexed aperture, Lyot stop.
    num_rings : integer
        Numer of segment rings.
    sampling : float
        Desired image plane sampling of coronagraphic PSF in pixels per lambda/D.
    robustness_px : int
        Robustness to Lyot stop misalignments in pixels. None, 2 or 4.
    """
    def __init__(self, input_dir, num_rings, sampling, robustness_px=None):
        self.num_rings = num_rings
        data_in_repo = os.path.join(input_dir, f'{num_rings}-Hex')

        # Diameter for each hex ring solution in meters as provided by GSFC
        # Taken from https://github.com/spacetelescope/aplc_optimization/blob/6bd5a6ecf46a3cf758c23853367d5b18e6a5a1d7/optimization_launchers/SCDA/do_LUVex_survey.py#L67
        pupil_diameter_circumscribed = {1: 7.9445, 2: 7.2617, 3: 7.7231, 4: 7.1522, 5: 5.9941}
        pupil_diameter_inscribed = {1: 6.0023, 2: 5.9994, 3: 5.9899, 4: 5.9937, 5: 6.8526}
        diameter_circumscribed = pupil_diameter_circumscribed[num_rings]
        diameter_inscribed = pupil_diameter_inscribed[num_rings]

        num_seg = 3 * num_rings * (num_rings + 1) + 1
        gap_size = 0.06    # in meters
        if num_rings in [1, 2]:
            seg_flat_to_flat = (diameter_inscribed - 2 * (num_rings - 1) * gap_size) / (2 * (num_rings - 1) + 1)
        elif num_rings in [3, 4, 5]:
            seg_flat_to_flat = (diameter_circumscribed - 2 * num_rings * gap_size) / (2 * num_rings + 1)
        else:
            raise ValueError(f"No telescope/coronagraph solution provided for {num_rings} number of rings.")

        if robustness_px is None:
            robust = 0
        elif robustness_px == 2:
            robust = 1
        elif robustness_px == 4:
            robust = 2
        else:
            raise ValueError(f"An apodizer design with robustness to a LS misalignment of {robustness_px} pixels does not exist for this aperture.")

        # These parameters are contained in the fits header of the apodizer
        pxsize = 1024
        fpm_rad = 3.5
        fpm_px = 150
        iwa = 3.4
        owa = 12
        imlamD = 1.2 * owa

        if num_rings in [1, 2]:
            aper_fname = f'masks/TelAp_LUVex_{num_rings:02d}-Hex_gy_ovsamp04__N{pxsize:04d}.fits'
        elif num_rings in [3, 4, 5]:
            aper_fname = f'masks/TelAp_LUVex_{num_rings:02d}-Hex_gy_clipped_ovsamp04__N{pxsize:04d}.fits'
        aper_ind_fname = aper_fname.split('.')[0] + '_indexed.fits'
        apod_fname = f'solutions/{robust}_SCDA_N1024_FPM350M0150_IWA0340_OWA01200_C10_BW10_Nlam3_LS_IDex_ID_OD0_OD_ls_982_no_strut.fits'
        ls_fname = f'masks/LS_LUVex_{num_rings:02d}-Hex_ID0000_OD0982_no_struts_gy_ovsamp4_N{pxsize:04d}.fits'

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

        super().__init__(input_dir=data_in_repo, sampling=sampling, diameter=diameter_circumscribed,
                         seg_flat_to_flat=seg_flat_to_flat, wvln=1, imlamD=imlamD, aplc_params=aplc_params)
