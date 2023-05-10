"""
This is a module containing classes for telescope and coronagraph (APLC) combinations for designs coming out
of the SCDA working group.
"""
import logging
import os
from astropy.io import fits
import hcipy
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.simulators.generic_segmented_telescopes import SegmentedAPLC, load_segment_centers

log = logging.getLogger()


class ScdaAPLC(SegmentedAPLC):
    """Any segmented telescope with APLC created with optimization scripts from SCDA."""

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

        # For the 3-Hex, 4-Hex and 5-Hex Luvex designs, actually create the LS #TODO: get to the bottom of this problem
        if hasattr(self, 'num_rings') and self.num_rings in [3, 4, 5]:
            all_ls_diams = [5.417565656565657, 5.4495959595959595, 5.4981407035175875]
            ls_diam = all_ls_diams[self.num_rings - 3]
            lyot_stop = hcipy.evaluate_supersampled(hcipy.circular_aperture(diameter=ls_diam), pupil_grid, 4)

        # Create a focal plane mask
        samp_foc = fpm_px / (fpm_rad * 2)
        focal_grid_fpm = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=fpm_rad*1.01, wavelength=wvln)
        fpm = 1 - hcipy.circular_aperture(2 * fpm_rad * lam_over_d)(focal_grid_fpm)

        # Create a focal plane grid for the detector
        focal_det = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=sampling, num_airy=imlamD, wavelength=wvln)

        super().__init__(apod=apodizer, lyot_stop=lyot_stop, fpm=fpm, fpm_rad=fpm_rad, iwa=iwa, owa=owa, wvln=wvln,
                         diameter=diameter, aper=aperture, indexed_aper=aper_ind, seg_pos=seg_pos,
                         seg_diameter=seg_diameter_circumscribed, focal_grid=focal_det, sampling=sampling, imlamD=imlamD)


class HexRingAPLC(ScdaAPLC):
    """Segmented telescope with APLC with varying number of hexagonal segment rings."""

    def __init__(self, input_dir, num_rings, sampling, robustness_px=0):
        """
        Parameters:
        ----------
        input_dir : string
            Path to directories sorted by number of rings that contain the input files: apodizer, aperture, indexed aperture, Lyot stop.
        num_rings : integer
            Numer of segment rings.
        sampling : float
            Desired image plane sampling of coronagraphic PSF in pixels per lambda/D.
        robustness_px : int
            Robustness to Lyot stop misalignments in pixels. 0, 2, 4 or 8.
        """
        if num_rings not in [1, 2, 3, 4, 5]:
            raise ValueError(f"No telescope/coronagraph solution provided for {num_rings} number of rings.")
        self.num_rings = num_rings
        num_seg = 3 * num_rings * (num_rings + 1) + 1
        data_in_repo = os.path.join(input_dir, f'{num_rings}-Hex')

        # The number of pixels is present in the fits headers of both the aperture as well as the apodizer file.
        # But all the files on the repo have the same size, so we are leaving this hard-coded for now.
        # Adjust later if needed.
        pxsize = 1024

        # Find correct aperture file and read parameters from its header
        if num_rings in [1, 2]:
            aper_fname = f'masks/TelAp_LUVex_{num_rings:02d}-Hex_gy_ovsamp04_N{pxsize:04d}.fits'
        elif num_rings in [3, 4, 5]:
            num_seg -= 6    # No corner segments in these designs
            aper_fname = f'masks/TelAp_LUVex_{num_rings:02d}-Hex_gy_clipped_ovsamp04_N{pxsize:04d}.fits'
        aper_ind_fname = aper_fname.split('.')[0] + '_indexed.fits'

        aper_hdr = fits.getheader(os.path.join(data_in_repo, aper_fname))
        # diameter_circumscribed = aper_hdr['D_CIRC']
        all_diameters_outer = [7.952780833, 7.258823341, 7.72593506, 7.136856303, 6.844452477]
        diameter_circumscribed = all_diameters_outer[num_rings - 1]
        seg_flat_to_flat = aper_hdr['SEG_F2F']
        wvln = CONFIG_PASTIS.getfloat('HexRingTelescope', 'lambda') * 1e-9    # m

        # Find correct apodizer file and read parameters from its header
        if robustness_px not in [0, 2, 4, 8]:
            raise ValueError(f"An apodizer design with robustness to a LS misalignment of {robustness_px} pixels does not exist for this aperture.")
        apod_fname = f'solutions/{num_rings:01d}-Hex_robust_{robustness_px}px_N{pxsize:04d}.fits'
        apod_hdr = fits.getheader(os.path.join(data_in_repo, apod_fname))
        imlamD = 1.2 * apod_hdr['OWA']

        # Find correct Lyot stop file
        ls_fname = f'masks/LS_LUVex_{num_rings:02d}-Hex_ID0000_OD0982_no_struts_gy_ovsamp4_N{pxsize:04d}.fits'

        # Bundle all parameters
        aplc_params = {'num_seg': num_seg,
                       'aper_fname': aper_fname,
                       'aper_ind_fname': aper_ind_fname,
                       'apod_fname': apod_fname,
                       'ls_fname': ls_fname,
                       'pxsize': pxsize,
                       'fpm_rad': apod_hdr['FPM_RAD'],
                       'fpm_px': apod_hdr['FPM_PIX'],
                       'iwa': apod_hdr['IWA'],
                       'owa': apod_hdr['OWA']}

        super().__init__(input_dir=data_in_repo, sampling=sampling, diameter=diameter_circumscribed,
                         seg_flat_to_flat=seg_flat_to_flat, wvln=wvln, imlamD=imlamD, aplc_params=aplc_params)
