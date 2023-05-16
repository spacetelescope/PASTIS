import os
import hcipy

from pastis.config import CONFIG_PASTIS
from pastis.simulators.generic_segmented_telescopes import SegmentedAPLC, load_segment_centers


class ELTHarmoniSPC(SegmentedAPLC):
    def __init__(self, input_dir, sampling, wvln, spc_design, fpm_rad):

        # Parameters for the specific SPC designs
        self.spc_dict = {'HSP1': {'pxsize': 1034, 'iwa': 5, 'owa': 11.5, 'fname': 'hsp1_1034px.fits'},
                         'HSP2': {'pxsize': 1034, 'iwa': 7.5, 'owa': 39.5, 'fname': 'hsp2_1034px.fits'}}

        self.spc_design = spc_design
        fpm_px = CONFIG_PASTIS.getfloat('ELT', 'fpm_px')
        num_seg = CONFIG_PASTIS.getint('ELT', 'nb_subapertures')
        imlamD = 1.2 * self.spc_dict[spc_design]['owa']
        diameter = CONFIG_PASTIS.getfloat('ELT', 'diameter')     # m
        aper_fname = CONFIG_PASTIS.get('ELT', 'aperture_path_in_optics')
        aper_ind_fname = CONFIG_PASTIS.get('ELT', 'indexed_aperture_path_in_optics')
        apod_fname = self.spc_dict[spc_design]['fname']
        ls_fname = CONFIG_PASTIS.get('ELT', 'lyot_stop_path_in_optics')
        pxsize = self.spc_dict[spc_design]['pxsize']
        iwa = self.spc_dict[spc_design]['iwa']
        owa = self.spc_dict[spc_design]['owa']
        seg_diameter_circumscribed = 1.45   # m

        pupil_grid = hcipy.make_pupil_grid(dims=pxsize, diameter=diameter)
        lam_over_d = wvln / diameter

        # Load segmented aperture
        pup_read = hcipy.read_fits(os.path.join(input_dir, aper_fname))
        aperture = hcipy.Field(pup_read.ravel(), pupil_grid)

        # Load indexed segmented aperture
        aper_ind_read = hcipy.read_fits(os.path.join(input_dir, aper_ind_fname))
        aper_ind = hcipy.Field(aper_ind_read.ravel(), pupil_grid)
        seg_pos = load_segment_centers(input_dir, aper_ind_fname, num_seg, diameter=0.98)
        # The segment positions saved in the indexed aperture file are already scaled to the overall ELT diameter,
        # so the function above does not have to do that anymore, and we pass it a diameter o ~1. This has been scaled
        # manually so that the local ode bases overlap as good as possible with the aperture segments.

        # Load apodizer
        apod_read = hcipy.read_fits(os.path.join(input_dir, apod_fname))
        apodizer = hcipy.Field(apod_read.ravel(), pupil_grid)

        # Load Lyot stop
        ls_read = hcipy.read_fits(os.path.join(input_dir, ls_fname))
        lyot_stop = hcipy.Field(ls_read.ravel(), pupil_grid)

        # Create a focal plane mask
        samp_foc = fpm_px / (fpm_rad * 2)
        focal_grid_fpm = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=fpm_rad, wavelength=wvln)
        fpm = 1 - hcipy.circular_aperture(2 * fpm_rad * lam_over_d)(focal_grid_fpm)

        # Create a focal plane grid for the detector
        focal_det = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=sampling, num_airy=imlamD, wavelength=wvln)

        super().__init__(apod=apodizer, lyot_stop=lyot_stop, fpm=fpm, fpm_rad=fpm_rad, iwa=iwa, owa=owa, wvln=wvln,
                         diameter=diameter, aper=aperture, indexed_aper=aper_ind, seg_pos=seg_pos,
                         seg_diameter=seg_diameter_circumscribed, focal_grid=focal_det, sampling=sampling, imlamD=imlamD)
