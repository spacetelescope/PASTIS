"""
This is a module containing functions and classes for imaging propagation with HabEx.
"""
import logging
import hcipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pastis.config import CONFIG_PASTIS
from pastis.e2e_simulators.generic_segmented_telescopes import Telescope

log = logging.getLogger()


class Habex_VVC(Telescope):
    def __init__(self, charge):

        # Define or read parameters
        pxsize = CONFIG_PASTIS.getfloat('HabEx', 'pupil_pixels')
        diameter = CONFIG_PASTIS.getfloat('HabEx', 'diameter')     # m
        sampling = CONFIG_PASTIS.getfloat('HabEx', 'sampling')
        imlamD = CONFIG_PASTIS.getfloat('HabEx', 'imlamD')
        wavelength = CONFIG_PASTIS.getfloat('HabEx', 'lambda') * 1e-9    # m
        ls_diam = CONFIG_PASTIS.getfloat('HabEx', 'lyot_stop_ratio')
        self.iwa = CONFIG_PASTIS.getfloat('HabEx', 'iwa')
        self.owa = CONFIG_PASTIS.getfloat('HabEx', 'owa')

        # Create circular telescope primary
        pupil_grid = hcipy.make_pupil_grid(pxsize, diameter)
        aperture = hcipy.aperture.circular_aperture(diameter)
        aperture_field = hcipy.evaluate_supersampled(aperture, pupil_grid, 1)

        # Create focal grid and instantiate Telescope
        focal_grid = hcipy.make_focal_grid(sampling, imlamD, pupil_diameter=diameter, focal_length=1.,
                                           reference_wavelength=wavelength)

        super().__init__(wvln=wavelength, diameter=diameter, aper=aperture_field, focal_grid=focal_grid,
                         sampling=sampling, imlamD=imlamD)

        # Set up coronagraph
        self.charge = charge
        self.coro = hcipy.VortexCoronagraph(pupil_grid, charge)
        lyot_stop_circle = hcipy.aperture.circular_aperture(diameter * ls_diam)
        self.lyotstop = hcipy.evaluate_supersampled(lyot_stop_circle, pupil_grid, 1)
        self.lyot_mask = hcipy.Apodizer(self.lyotstop)

        # DH mask
        dh_outer = hcipy.circular_aperture(2 * self.owa * self.lam_over_d)(self.focal_det)
        dh_inner = hcipy.circular_aperture(2 * self.iwa * self.lam_over_d)(self.focal_det)
        self.dh_mask = (dh_outer - dh_inner).astype('bool')
        
    def calc_psf(self, ref=False, display_intermediate=False,  return_intermediate=None, norm_one_photon=False):
        """ Calculate the PSF of this telescope, and return optionally all E-fields.

        Parameters:
        ----------
        ref : bool
            Keyword for additionally returning the reference PSF without the FPM.
        display_intermediate : bool
            Whether or not to display images of all planes.
        return_intermediate : string
            default None; if "efield", will also return E-fields of each plane and DM
        norm_one_photon : bool
            Whether or not to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns:
        --------
        wf_image.intensity : Field
            returned if return_intermediate=None (default)
        wf_image : hcipy.Wavefront
            returned if return_intermediate='efield'
        intermediates : dict
            Dictionary containing the Wavefronts of all the planes, returned if return_intermediate='efield'
        """
        
        if isinstance(return_intermediate, bool):
            raise TypeError(f"'return_intermediate' needs to be 'efield' or 'intensity' if you want all "
                            f"E-fields returned by 'calc_psf()'.")

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_zm, wf_ripples, wf_dm, _tr = self._propagate_active_pupils(norm_one_photon)

        # All E-field propagations 
        wf_before_lyot = self.coro(wf_active_pupil)
        wf_lyot = self.lyot_mask(wf_before_lyot)
        wf_lyot.wavelength = self.wvln
        wf_im_coro = self.prop(wf_lyot)
        wf_im_ref = self.prop(wf_active_pupil)

        if display_intermediate:
            plt.figure(figsize=(10, 10))

            plt.subplot(3, 3, 1)
            hcipy.imshow_field(self.wf_aper.intensity, mask=self.aperture, cmap='Greys_r')
            plt.title('Primary mirror')

            plt.subplot(3, 3, 2)
            hcipy.imshow_field(wf_zm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Global Zernike phase')

            plt.subplot(3, 3, 3)
            hcipy.imshow_field(wf_dm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Deformable mirror phase')

            plt.subplot(3, 3, 4)
            hcipy.imshow_field(wf_ripples.phase, mask=self.aperture, cmap='RdBu')
            plt.title('High modes mirror phase')

            plt.subplot(3, 3, 5)
            hcipy.imshow_field(wf_active_pupil.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Total phase in entrance pupil')

            plt.subplot(3, 3, 6)
            hcipy.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-8, vmax=1e-1), cmap='inferno')
            plt.title('Before Lyot stop')
            
            plt.subplot(3, 3, 7)
            hcipy.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', mask=self.lyotstop)
            plt.title('After Lyot stop')

            plt.subplot(3, 3, 8)
            hcipy.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(),
                               norm=LogNorm(vmin=1e-12, vmax=1e-5), cmap='inferno')
            plt.title('Coro image')
            plt.colorbar()

        if return_intermediate == 'intensity':

            # Return the intensity in all planes; except phases on all DMs, and combined phase from active pupils
            intermediates = {'zernike_mirror': wf_zm.phase,
                             'dm': wf_dm.phase,
                             'ripple_mirror': wf_ripples.phase,
                             'active_pupil': wf_active_pupil.phase,
                             'before_lyot': wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                             'after_lyot': wf_lyot.intensity / wf_lyot.intensity.max()}

            if ref:
                return wf_im_coro.intensity, wf_im_ref.intensity, intermediates
            else:
                return wf_im_coro.intensity, intermediates

        if return_intermediate == 'efield':

            # Return the E-fields in all planes; except intensity in focal plane after FPM
            intermediates = {'zernike_mirror': wf_zm,
                             'dm': wf_dm,
                             'ripple_mirror': wf_ripples,
                             'active_pupil': wf_active_pupil,
                             'before_lyot': wf_before_lyot,
                             'after_lyot': wf_lyot}

            if ref:
                return wf_im_coro, wf_im_ref, intermediates
            else:
                return wf_im_coro, intermediates

        if ref:
            return wf_im_coro.intensity, wf_im_ref.intensity

        return wf_im_coro.intensity
