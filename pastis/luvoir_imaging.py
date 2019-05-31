"""
This is a module containing functions and classes for imaging propagation with HCIPy, for now LUVOIR A.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import hcipy as hc
from hcipy.optics.segmented_mirror import SegmentedMirror


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
    fpm : fpm
        Focal plane mask
    params : dict
        wavelength, diameter, image size in lambda/D, FPM radius
    """

    def __init__(self, aper, indexed_aperture, seg_pos, apod, lyotst, fpm, focal_grid, params):
        self.sm = SegmentedMirror(indexed_aperture=indexed_aperture, seg_pos=seg_pos)
        self.aper = aper
        self.apodizer = apod
        self.lyotstop = lyotst
        self.fpm = fpm
        self.wvln = params['wavelength']
        self.diam = params['diameter']
        self.imlamD = params['imlamD']
        self.fpm_rad = params['fpm_rad']
        self.lamDrad = self.wvln / self.diam
        self.coro = hc.LyotCoronagraph(indexed_aperture.grid, fpm, lyotst)
        self.prop = hc.FraunhoferPropagator(indexed_aperture.grid, focal_grid)
        self.coro_no_ls = hc.LyotCoronagraph(indexed_aperture.grid, fpm)
        self.wf_aper = hc.Wavefront(aper, wavelength=self.wvln)
        self.focal_det = focal_grid

    def calc_psf(self, ref=False, display_intermediate=False,  return_intermediate=False):
        """Calculate the PSF of the segmented telescope, normalized to contrast units.

        Parameters:
        ----------
        ref : bool
            Keyword for additinally returning the refrence PSF without the FPM.
        Returns:
        --------
        wf_im_coro.intensity : Field
            Coronagraphic image, normalized to contrast units by max of reference image (even when ref
            not returned).
        wf_im_ref.intensity : Field, optional
            Reference image without FPM.
        intermediates : dict, optional
            Intermediate plane intensity images; except for full wavefront on segmented mirror.
        """

        # Create fake FPM for plotting
        fpm_plot = 1 - hc.circular_aperture(2 * self.fpm_rad * self.lamDrad)(self.focal_det)

        # Create apodozer as hc.Apodizer() object to be able to propagate through it
        apod_prop = hc.Apodizer(self.apodizer)

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
        wf_ref_pup = hc.Wavefront(self.apodizer * self.lyotstop, wavelength=self.wvln)
        wf_im_ref = self.prop(wf_ref_pup)

        # Display intermediate planes
        if display_intermediate:

            plt.figure(figsize=(15, 15))

            plt.subplot(331)
            hc.imshow_field(wf_sm.phase, mask=self.aper, cmap='RdBu')
            plt.title('Seg aperture phase')

            plt.subplot(332)
            hc.imshow_field(wf_apod.intensity, cmap='inferno')
            plt.title('Apodizer')

            plt.subplot(333)
            hc.imshow_field(wf_before_fpm.intensity / wf_before_fpm.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Before FPM')

            plt.subplot(334)
            hc.imshow_field(int_after_fpm / wf_before_fpm.intensity.max(), cmap='inferno')
            plt.title('After FPM')

            plt.subplot(335)
            hc.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno')
            plt.title('Before Lyot stop')

            plt.subplot(336)
            hc.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno', mask=self.lyotstop)
            plt.title('After Lyot stop')

            plt.subplot(337)
            hc.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(), norm=LogNorm(vmin=1e-10, vmax=1e-3),
                            cmap='inferno')
            plt.title('Final image')
            plt.colorbar()

        if return_intermediate:

            intermediates = {'seg_mirror': wf_sm,
                             'apod': wf_apod.intensity,
                             'before_fpm': wf_before_fpm.intensity / wf_before_fpm.intensity.max(),
                             'after_fpm': int_after_fpm / wf_before_fpm.intensity.max(),
                             'before_lyot': wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                             'after_lyot': wf_lyot.intensity / wf_lyot.intensity.max()}

            if ref:
                return wf_im_coro.intensity, wf_im_ref.intensity, intermediates
            else:
                return wf_im_coro.intensity, intermediates

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
