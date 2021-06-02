"""
This is a module containing functions and classes for imaging propagation with HCIPy, for now LUVOIR A.
"""
import logging
import os
from astropy.io import fits
import hcipy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pastis.config import CONFIG_PASTIS
from pastis.e2e_simulators.generic_segmented_telescopes import SegmentedTelescope, SegmentedAPLC, load_segment_centers

log = logging.getLogger()


class LuvoirA_APLC(SegmentedAPLC):
    """ LUVOIR A with APLC simulator

    Parameters:
    ----------
    input_dir : string
        Path to input files: apodizer, aperture, indexed aperture, Lyot stop.
    apod_design : string
        Choice of apodizer design from May 2019 delivery. "small", "medium" or "large".
    sampling : float
        Desired image plane sampling of coronagraphic PSF in pixels per lambda/D.
    """
    def __init__(self, input_dir, apod_design, sampling):
        self.apod_design = apod_design
        self.apod_dict = {'small': {'pxsize': 1000, 'fpm_rad': 3.5, 'fpm_px': 150, 'iwa': 3.4, 'owa': 12.,
                                    'fname': '0_LUVOIR_N1000_FPM350M0150_IWA0340_OWA01200_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'medium': {'pxsize': 1000, 'fpm_rad': 6.82, 'fpm_px': 250, 'iwa': 6.72, 'owa': 23.72,
                                     'fname': '0_LUVOIR_N1000_FPM682M0250_IWA0672_OWA02372_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'large': {'pxsize': 1000, 'fpm_rad': 13.38, 'fpm_px': 400, 'iwa': 13.28, 'owa': 46.88,
                                    'fname': '0_LUVOIR_N1000_FPM1338M0400_IWA1328_OWA04688_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'}}
        imlamD = 1.2 * self.apod_dict[apod_design]['owa']

        wvln = CONFIG_PASTIS.getfloat('LUVOIR', 'lambda') * 1e-9    # m
        diameter = CONFIG_PASTIS.getfloat('LUVOIR', 'diameter')     # m
        lam_over_d = wvln / diameter

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
        aper_ind = hcipy.Field(aper_ind_read.ravel(), pupil_grid)

        seg_pos = load_segment_centers(input_dir, aper_ind_path, CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'), diameter)
        seg_diameter_circumscribed = 2 / np.sqrt(3) * 1.2225    # m

        # Create a focal plane mask
        samp_foc = self.apod_dict[apod_design]['fpm_px'] / (self.apod_dict[apod_design]['fpm_rad'] * 2)
        focal_grid_fpm = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=samp_foc, num_airy=self.apod_dict[apod_design]['fpm_rad'], wavelength=wvln)
        fpm = 1 - hcipy.circular_aperture(2*self.apod_dict[apod_design]['fpm_rad'] * lam_over_d)(focal_grid_fpm)

        # Create a focal plane grid for the detector
        focal_det = hcipy.make_focal_grid_from_pupil_grid(pupil_grid=pupil_grid, q=sampling, num_airy=imlamD, wavelength=wvln)

        super().__init__(apod=apodizer, lyot_stop=lyot_stop, fpm=fpm, fpm_rad=self.apod_dict[apod_design]['fpm_rad'],
                         iwa=self.apod_dict[apod_design]['iwa'], owa=self.apod_dict[apod_design]['owa'],
                         wvln=wvln, diameter=diameter, aper=aperture, indexed_aper=aper_ind, seg_pos=seg_pos,
                         seg_diameter=seg_diameter_circumscribed, focal_grid=focal_det, sampling=sampling, imlamD=imlamD)


class SegmentedTelescopeAPLC(SegmentedAPLC):
    """ THIS PIPES DIRECTLY THROUGH TO SegmentedAPLC.
    !!! This class only still exists for back-compatibility. Please use SegmentedTelescope and SegmentedAPLC for new implementations. !!!
    """
    def __init__(self, aper, indexed_aperture, seg_pos, apod, lyotst, fpm, focal_grid, params):
        wvln = params['wavelength']    # m
        diameter = params['diameter']    # m
        seg_diameter = params['segment_circumscribed_diameter']    # m
        sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')    # pixels per lambda/D
        fpm_rad = params['fpm_rad']    # lambda/D
        iwa = fpm_rad - 0.1    # lambda/D
        imlamD = params['imlamD']    # lambda/D
        owa = imlamD / 1.2    # lambda/D
        super().__init__(apod, lyotst, fpm, fpm_rad, iwa, owa, wvln=wvln, diameter=diameter, aper=aper,
                         indexed_aper=indexed_aperture, seg_pos=seg_pos, seg_diameter=seg_diameter,
                         focal_grid=focal_grid, sampling=sampling, imlamD=imlamD)


class LuvoirAPLC(LuvoirA_APLC):
    """ THIS PIPES DIRECTLY THROUGH TO LuvoirA_APLC.
    !!! This class only still exists for back-compatibility. Please use LuvoirA_APLC for new implementations. !!!
    """
    def __init__(self, input_dir, apod_design, samp):
        super().__init__(input_dir, apod_design, samp)


class LuvoirBVortex(SegmentedTelescope):
    """ A segmented Vortex coronagraph

    Parameters:
    ----------
    input_dir : string
        Path to input files: DMs, aperture, indexed aperture, Lyot stop, etc.
    charge : int
        charge of vortex coronagraph
    """

    def __init__(self, input_dir, charge):
        self.input_dir = input_dir
        self.set_up_telescope()
        super().__init__(self.wavelength, self.D_pup, self.aperture, self.indexed_aperture, self.seg_pos,
                         self.segment_circum_diameter, self.focal_grid, self.samp_foc, self.rad_foc, center_segment=False)
        # TODO: center seg is False on purpose, because of the awkward segment numbering we currently have for LUVOIR-B
        # NOTE: self.pupil_grid is already equal to pupil_grid_dms through self.aper

        # Propagators
        self.fresnel = hcipy.propagation.FresnelPropagator(self.pupil_grid, self.zDM, num_oversampling=1)
        self.fresnel_back = hcipy.propagation.FresnelPropagator(self.pupil_grid, -self.zDM, num_oversampling=1)
        self.charge = charge
        self.coro = hcipy.VortexCoronagraph(self.pupil_grid, charge, scaling_factor=4)

        # Set up DH mask
        iwa = CONFIG_PASTIS.getfloat('LUVOIR-B', 'IWA')
        owa = CONFIG_PASTIS.getfloat('LUVOIR-B', 'OWA')
        dh_outer = hcipy.circular_aperture(2 * owa * self.lam_over_d)(self.focal_grid)
        dh_inner = hcipy.circular_aperture(2 * iwa * self.lam_over_d)(self.focal_grid)
        self.dh_mask = (dh_outer - dh_inner).astype('bool')

    def set_up_telescope(self):

        # Read all input data files
        datadir = self.input_dir
        aperture_data = fits.getdata(os.path.join(datadir, 'Pupil1.fits'))
        indexed_aperture_data = fits.getdata(os.path.join(datadir, 'aperture_LUVOIR-B_indexed.fits'))
        apod_stop_data = fits.getdata(os.path.join(datadir, 'APOD.fits'))
        dm2_stop_data = fits.getdata(os.path.join(datadir, 'DM2stop.fits'))
        lyot_stop_data = fits.getdata(os.path.join(datadir, 'LS.fits'))
        dm1_data = fits.getdata(os.path.join(datadir, 'surfDM1.fits'))
        dm2_data = fits.getdata(os.path.join(datadir, 'surfDM2.fits'))

        # Parameters
        nPup = CONFIG_PASTIS.getfloat('LUVOIR-B', 'pupil_pixels')
        self.D_pup = CONFIG_PASTIS.getfloat('LUVOIR-B', 'D_pup')
        self.samp_foc = CONFIG_PASTIS.getfloat('LUVOIR-B', 'sampling')
        self.rad_foc = CONFIG_PASTIS.getfloat('LUVOIR-B', 'imlamD')
        self.wavelength = CONFIG_PASTIS.getfloat('LUVOIR-B', 'lambda') * 1e-9   # m

        nPup_arrays = apod_stop_data.shape[0]
        nPup_dms = dm1_data.shape[0]
        nPup_dm_stop = dm2_stop_data.shape[0]
        self.zDM = (self.D_pup / 2) ** 2 / (self.wavelength * 549.1429)    # last number is Fresnel number for 10% bandpass

        # Pad arrays to correct sizes
        apod_stop_data_pad = np.pad(apod_stop_data, int((nPup_dms - nPup_arrays) / 2), mode='constant')
        DM2Stop_data_pad = np.pad(dm2_stop_data, int((nPup_dms - nPup_dm_stop) / 2), mode='constant')
        lyot_stop_data_pad = np.pad(lyot_stop_data, int((nPup_dms - nPup_arrays) / 2), mode='constant')
        aperture_data_pad = np.pad(aperture_data, int((nPup_dms - nPup_arrays) / 2), mode='constant')
        indexed_aperture_data_pad = np.pad(indexed_aperture_data, int((nPup_dms - nPup_arrays) / 2), mode='constant')

        # Create pupil grids and focal grid
        pupil_grid_arrays = hcipy.make_pupil_grid(nPup * (nPup_arrays / nPup), self.D_pup * (nPup_arrays / nPup))
        pupil_grid_dms = hcipy.make_pupil_grid(nPup * (nPup_dms / nPup), self.D_pup * (nPup_dms / nPup))
        self.focal_grid = hcipy.make_focal_grid(self.samp_foc, self.rad_foc, pupil_diameter=self.D_pup, focal_length=1.,
                                                reference_wavelength=self.wavelength)

        # Create all optical components on DM pupil grids
        self.apod_stop = hcipy.Field(np.reshape(apod_stop_data_pad, nPup_dms ** 2), pupil_grid_dms)
        self.DM2_circle = hcipy.Field(np.reshape(DM2Stop_data_pad, nPup_dms ** 2), pupil_grid_dms)
        self.lyot_mask = hcipy.Field(np.reshape(lyot_stop_data_pad, nPup_dms ** 2), pupil_grid_dms)
        self.lyot_stop = hcipy.Apodizer(self.lyot_mask)
        self.aperture = hcipy.Field(np.reshape(aperture_data_pad, nPup_dms ** 2), pupil_grid_dms)
        self.indexed_aperture = hcipy.Field(np.reshape(indexed_aperture_data_pad, nPup_dms ** 2), pupil_grid_dms)
        self.DM1 = hcipy.Field(np.reshape(dm1_data, nPup_dms ** 2), pupil_grid_dms)
        self.DM2 = hcipy.Field(np.reshape(dm2_data, nPup_dms ** 2), pupil_grid_dms)

        self.seg_pos = load_segment_centers(datadir, 'aperture_LUVOIR-B_indexed.fits',
                                            CONFIG_PASTIS.getint('LUVOIR-B', 'nb_subapertures'), self.D_pup)
        # Calculate segment circumscribed diameter from flat-to-flat distance, and scale from 8m to pupil size used here
        self.segment_circum_diameter = 2 / np.sqrt(3) * 0.955 * (self.D_pup/8)   # m

    def calc_psf(self, ref=False, display_intermediate=False,  return_intermediate=None):

        if isinstance(return_intermediate, bool):
            raise TypeError(f"'return_intermediate' needs to be 'efield' or 'intensity' if you want all "
                            f"E-fields returned by 'calc_psf()'.")

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm = self._propagate_active_pupils()

        # All E-field propagations
        wf_dm1_coro = hcipy.Wavefront(wf_active_pupil.electric_field * np.exp(4 * 1j * np.pi/self.wavelength * self.DM1), self.wavelength)
        wf_dm2_coro_before = self.fresnel(wf_dm1_coro)
        wf_dm2_coro_after = hcipy.Wavefront(wf_dm2_coro_before.electric_field * np.exp(4 * 1j * np.pi / self.wavelength * self.DM2) * self.DM2_circle, self.wavelength)
        wf_back_at_dm1 = self.fresnel_back(wf_dm2_coro_after)
        wf_apod_stop = hcipy.Wavefront(wf_back_at_dm1.electric_field * self.apod_stop, self.wavelength)

        wf_before_lyot = self.coro(wf_apod_stop)
        wf_lyot = self.lyot_stop(wf_before_lyot)
        wf_lyot.wavelength = self.wavelength

        wf_im_coro = self.prop(wf_lyot)
        wf_im_ref = self.prop(wf_back_at_dm1)

        # Display intermediate planes
        if display_intermediate:

            plt.figure(figsize=(15, 15))

            plt.subplot(3, 4, 1)
            hcipy.imshow_field(self.wf_aper.intensity, mask=self.aperture, cmap='Greys_r')
            plt.title('Primary mirror')

            plt.subplot(3, 4, 2)
            hcipy.imshow_field(wf_sm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Segmented mirror phase')

            plt.subplot(3, 4, 3)
            hcipy.imshow_field(wf_zm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Global Zernike phase')

            plt.subplot(3, 4, 4)
            hcipy.imshow_field(wf_dm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Deformable mirror phase')

            plt.subplot(3, 4, 5)
            hcipy.imshow_field(wf_harris_sm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Harris mode mirror phase')

            plt.subplot(3, 4, 6)
            hcipy.imshow_field(wf_ripples.phase, mask=self.aperture, cmap='RdBu')
            plt.title('High modes mirror phase')

            plt.subplot(3, 4, 7)
            hcipy.imshow_field(wf_apod_stop.intensity, cmap='inferno')
            plt.title('Pupil stop after coron DMs')

            plt.subplot(3, 4, 8)
            hcipy.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-3, vmax=1), cmap='inferno')
            plt.title('Before Lyot stop')

            plt.subplot(3, 4, 9)
            hcipy.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-5, vmax=1), cmap='inferno', mask=self.lyot_mask)
            plt.title('After Lyot stop')

            plt.subplot(3, 4, 10)
            hcipy.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(),
                               norm=LogNorm(vmin=1e-10, vmax=1e-3), cmap='inferno')
            plt.title('Coro image')
            plt.colorbar()

        if return_intermediate == 'intensity':

            # Return the intensity in all planes; except phases on all DMs, and combined phase from active pupils
            intermediates = {'seg_mirror': wf_sm.phase,
                             'zernike_mirror': wf_zm.phase,
                             'dm': wf_dm.phase,
                             'harris_seg_mirror': wf_harris_sm.phase,
                             'ripple_mirror': wf_ripples.phase,
                             'active_pupil': wf_active_pupil.phase,
                             'apod': wf_apod_stop.intensity,
                             'before_lyot': wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                             'after_lyot': wf_lyot.intensity / wf_lyot.intensity.max()}

            if ref:
                return wf_im_coro.intensity, wf_im_ref.intensity, intermediates
            else:
                return wf_im_coro.intensity, intermediates

        if return_intermediate == 'efield':

            # Return the E-fields in all planes; except intensity in focal plane after FPM
            intermediates = {'seg_mirror': wf_sm,
                             'zernike_mirror': wf_zm,
                             'dm': wf_dm,
                             'harris_seg_mirror': wf_harris_sm,
                             'ripple_mirror': wf_ripples,
                             'active_pupil': wf_active_pupil,
                             'apod': wf_apod_stop,
                             'before_lyot': wf_before_lyot,
                             'after_lyot': wf_lyot}

            if ref:
                return wf_im_coro, wf_im_ref, intermediates
            else:
                return wf_im_coro, intermediates

        if ref:
            return wf_im_coro.intensity, wf_im_ref.intensity

        return wf_im_coro.intensity
