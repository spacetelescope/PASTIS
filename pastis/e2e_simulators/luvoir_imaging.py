"""
This is a module containing functions and classes for imaging propagation with HCIPy, for now LUVOIR A.
"""
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import hcipy

from pastis.config import CONFIG_PASTIS

log = logging.getLogger()

try:
    from hcipy.optics.segmented_mirror import SegmentedMirror
except ImportError:
    log.info('SegmentedMirror simulator from hcipy@980f39c was not imported.')


class SegmentedTelescopeAPLC:
    """ A segmented telescope with an APLC and actuated segments.

    Parameters:
    ----------
    aper : Field
        Telescope aperture.
    indexed_aperture : Field
        The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
        segment identification. Segment gaps must be strictly zero.
    seg_pos : CartesianGrid(UnstructuredCoords)
        Segment positions of the aperture.
    apod : Field
        Apodizer
    lyots : Field
        Lyot stop
    fpm : Field
        Focal plane mask
    focal_grid :
        Focal plane grid to put final image on
    params : dict
        wavelength, telescope diameter, image size in lambda/D, FPM radius in lambda/D, segment circumscribed diameter in m
    """

    def __init__(self, aper, indexed_aperture, seg_pos, apod, lyotst, fpm, focal_grid, params):
        self.sm = SegmentedMirror(indexed_aperture=indexed_aperture, seg_pos=seg_pos)
        self.aperture = aper
        self.apodizer = apod
        self.lyotstop = lyotst
        self.wvln = params['wavelength']
        self.diam = params['diameter']
        self.segment_circumscribed_diameter = params['segment_circumscribed_diameter']
        self.imlamD = params['imlamD']
        self.fpm_rad = params['fpm_rad']
        self.lam_over_d = self.wvln / self.diam
        self.coro = hcipy.LyotCoronagraph(indexed_aperture.grid, fpm, lyotst)
        self.prop = hcipy.FraunhoferPropagator(indexed_aperture.grid, focal_grid)
        self.coro_no_ls = hcipy.LyotCoronagraph(indexed_aperture.grid, fpm)
        self.wf_aper = hcipy.Wavefront(aper, wavelength=self.wvln)
        self.focal_det = focal_grid

    def calc_psf(self, ref=False, display_intermediate=False,  return_intermediate=None):
        """Calculate the PSF of the segmented telescope, normalized to contrast units.

        Parameters:
        ----------
        ref : bool
            Keyword for additionally returning the refrence PSF without the FPM.
        display_intermediate : bool
            Keyword for display of all planes.
        return_intermediate : string
            Either 'intensity', return the intensity in all planes; except phase on the SM (first plane)
            or 'efield', return the E-fields in all planes. Default none.
        Returns:
        --------
        wf_im_coro.intensity : Field
            Coronagraphic image, normalized to contrast units by max of reference image (even when ref
            not returned).
        wf_im_ref.intensity : Field, optional
            Reference image without FPM.
        intermediates : dict of Fields, optional
            Intermediate plane intensity images; except for full wavefront on segmented mirror.
        wf_im_coro : Wavefront
            Wavefront in last focal plane.
        wf_im_ref : Wavefront, optional
            Wavefront of reference image without FPM.
        intermediates : dict of Wavefronts, optional
            Intermediate plane E-fields; except intensity in focal plane after FPM.
        """

        # Create fake FPM for plotting
        fpm_plot = 1 - hcipy.circular_aperture(2 * self.fpm_rad * self.lam_over_d)(self.focal_det)

        # Create apodozer as hcipy.Apodizer() object to be able to propagate through it
        apod_prop = hcipy.Apodizer(self.apodizer)

        # Calculate all wavefronts of the full propagation
        wf_sm = self.sm(self.wf_aper)
        wf_apod = apod_prop(wf_sm)
        wf_lyot = self.coro(wf_apod)
        wf_im_coro = self.prop(wf_lyot)

        # Wavefronts in extra planes
        wf_before_fpm = self.prop(wf_apod)
        int_after_fpm = np.log10(wf_before_fpm.intensity / wf_before_fpm.intensity.max()) * fpm_plot  # this is the intensity straight
        wf_before_lyot = self.coro_no_ls(wf_apod)

        # Wavefronts of the reference propagation
        wf_ref_pup = hcipy.Wavefront(self.aperture * self.apodizer * self.lyotstop, wavelength=self.wvln)
        wf_im_ref = self.prop(wf_ref_pup)

        # Display intermediate planes
        if display_intermediate:

            plt.figure(figsize=(15, 15))

            plt.subplot(331)
            hcipy.imshow_field(wf_sm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Seg aperture phase')

            plt.subplot(332)
            hcipy.imshow_field(wf_apod.intensity, cmap='inferno')
            plt.title('Apodizer')

            plt.subplot(333)
            hcipy.imshow_field(wf_before_fpm.intensity / wf_before_fpm.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Before FPM')

            plt.subplot(334)
            hcipy.imshow_field(int_after_fpm / wf_before_fpm.intensity.max(), cmap='inferno')
            plt.title('After FPM')

            plt.subplot(335)
            hcipy.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno')
            plt.title('Before Lyot stop')

            plt.subplot(336)
            hcipy.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno', mask=self.lyotstop)
            plt.title('After Lyot stop')

            plt.subplot(337)
            hcipy.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(), norm=LogNorm(vmin=1e-10, vmax=1e-3),
                            cmap='inferno')
            plt.title('Final image')
            plt.colorbar()

        if return_intermediate == 'intensity':

            # Return the intensity in all planes; except phase on the SM (first plane)
            intermediates = {'seg_mirror': wf_sm.phase,
                             'apod': wf_apod.intensity,
                             'before_fpm': wf_before_fpm.intensity / wf_before_fpm.intensity.max(),
                             'after_fpm': int_after_fpm / wf_before_fpm.intensity.max(),
                             'before_lyot': wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                             'after_lyot': wf_lyot.intensity / wf_lyot.intensity.max()}

            if ref:
                return wf_im_coro.intensity, wf_im_ref.intensity, intermediates
            else:
                return wf_im_coro.intensity, intermediates

        if return_intermediate == 'efield':

            # Return the E-fields in all planes; except intensity in focal plane after FPM
            intermediates = {'seg_mirror': wf_sm,
                             'apod': wf_apod,
                             'before_fpm': wf_before_fpm,
                             'after_fpm': int_after_fpm,
                             'before_lyot': wf_before_lyot,
                             'after_lyot': wf_lyot}

            if ref:
                return wf_im_coro, wf_im_ref, intermediates
            else:
                return wf_im_coro, intermediates

        if ref:
            return wf_im_coro.intensity, wf_im_ref.intensity

        return wf_im_coro.intensity

    def flatten(self):
        self.sm.flatten()

    def set_segment(self, segid, piston, tip, tilt):
        self.sm.set_segment(segid, piston, tip, tilt)

    def apply_aberrations(self, aber_array):
        for vals in aber_array:
            self.sm.set_segment(vals[0], vals[1], vals[2], vals[3])

    def forward(self, wavefront):
        raise NotImplementedError()

    def backward(self, wavefront):
        raise NotImplementedError()


class LuvoirAPLC(SegmentedTelescopeAPLC):
    """ Simple E2E simulator for LUVOIR A (with APLC).

    All this does is instantiate a SegmentedTelescopeAPLC() by feeding it the appropriate parameters to make it a
    LUVOIR A simulator with one of the three baseline APLC designs.

    Parameters:
    ----------
    input dir : string
        Path to input files: apodizer, aperture, indexed aperture, Lyot stop.
    apod_design : string
        Choice of apodizer design from May 2019 delivery. "small", "medium" or "large".
    samp : float
        Desired image plane sampling of coronagraphic PSF.
    """
    def __init__(self, input_dir, apod_design, samp):

        self.sampling = samp
        self.apod_design = apod_design

        wvln = CONFIG_PASTIS.getfloat('LUVOIR', 'lambda') * 1e-9    # m
        diameter = CONFIG_PASTIS.getfloat('LUVOIR', 'diameter')
        self.lam_over_d = wvln / diameter

        self.apod_dict = {'small': {'pxsize': 1000, 'fpm_rad': 3.5, 'fpm_px': 150, 'iwa': 3.4, 'owa': 12.,
                                    'fname': '0_LUVOIR_N1000_FPM350M0150_IWA0340_OWA01200_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'medium': {'pxsize': 1000, 'fpm_rad': 6.82, 'fpm_px': 250, 'iwa': 6.72, 'owa': 23.72,
                                     'fname': '0_LUVOIR_N1000_FPM682M0250_IWA0672_OWA02372_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'large': {'pxsize': 1000, 'fpm_rad': 13.38, 'fpm_px': 400, 'iwa': 13.28, 'owa': 46.88,
                                    'fname': '0_LUVOIR_N1000_FPM1338M0400_IWA1328_OWA04688_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'}}
        imlamD = 1.2 * self.apod_dict[apod_design]['owa']

        # Create a gridi for pupil plane optics
        pupil_grid = hcipy.make_pupil_grid(dims=self.apod_dict[apod_design]['pxsize'], diameter=diameter)

        # Load segmented aperture
        aper_path = CONFIG_PASTIS.get('LUVOIR', 'aperture_path_in_optics')
        pup_read = hcipy.read_fits(os.path.join(input_dir, aper_path))
        aperture = hcipy.Field(pup_read.ravel(), pupil_grid)

        # Load apodizer
        apod_path = os.path.join('luvoir_stdt_baseline_bw10', apod_design + '_fpm', 'solutions',
                                 self.apod_dict[apod_design]['fname'])
        apod_read = hcipy.read_fits(os.path.join(input_dir, apod_path))
        apodizer = hcipy.Field(apod_read.ravel(), pupil_grid)

        # Load Lyot Stop
        ls_fname = CONFIG_PASTIS.get('LUVOIR', 'lyot_stop_path_in_optics')
        ls_read = hcipy.read_fits(os.path.join(input_dir, ls_fname))
        lyot_stop = hcipy.Field(ls_read.ravel(), pupil_grid)

        # Load indexed segmented aperture
        aper_ind_path = CONFIG_PASTIS.get('LUVOIR', 'indexed_aperture_path_in_optics')
        aper_ind_read = hcipy.read_fits(os.path.join(input_dir, aper_ind_path))
        self.aper_ind = hcipy.Field(aper_ind_read.ravel(), pupil_grid)

        # Load segment positions from fits header
        hdr = fits.getheader(os.path.join(input_dir, aper_ind_path))
        self.nseg = CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures')
        poslist = []
        for i in range(self.nseg):
            segname = 'SEG' + str(i + 1)
            xin = hdr[segname + '_X']
            yin = hdr[segname + '_Y']
            poslist.append((xin, yin))
        poslist = np.transpose(np.array(poslist))
        self.seg_pos = hcipy.CartesianGrid(poslist)

        # Create a focal plane mask
        samp_foc = self.apod_dict[apod_design]['fpm_px'] / (self.apod_dict[apod_design]['fpm_rad'] * 2)
        focal_grid_fpm = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=self.apod_dict[apod_design]['fpm_rad'], wavelength=wvln)
        self.fpm = 1 - hcipy.circular_aperture(2*self.apod_dict[apod_design]['fpm_rad']*self.lam_over_d)(focal_grid_fpm)

        # Create a focal plane grid for the detector
        self.focal_det = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=self.sampling, num_airy=imlamD, wavelength=wvln)

        # Bundle LUVOIR parameters and initialize the general segmented telescope with APLC class; includes the SM.
        luvoir_params = {'wavelength': wvln,
                         'diameter': diameter,
                         'imlamD': imlamD,
                         'fpm_rad': self.apod_dict[apod_design]['fpm_rad'],
                         'segment_circumscribed_diameter': 2 / np.sqrt(3) * 1.2225}    # m
        super().__init__(aper=aperture, indexed_aperture=self.aper_ind, seg_pos=self.seg_pos, apod=apodizer,
                         lyotst=lyot_stop, fpm=self.fpm, focal_grid=self.focal_det, params=luvoir_params)

        # Make dark hole mask
        dh_outer = hcipy.circular_aperture(2 * self.apod_dict[apod_design]['owa'] * self.lam_over_d)(
            self.focal_det)
        dh_inner = hcipy.circular_aperture(2 * self.apod_dict[apod_design]['iwa'] * self.lam_over_d)(
            self.focal_det)
        self.dh_mask = (dh_outer - dh_inner).astype('bool')
