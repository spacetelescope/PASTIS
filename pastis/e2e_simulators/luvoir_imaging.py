"""
This is a module containing functions and classes for imaging propagation with HCIPy, for now LUVOIR A.
"""
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import griddata
from astropy.io import fits
import hcipy

from pastis.config import CONFIG_PASTIS
from pastis.e2e_simulators.indexed_segmented_mirror import SegmentedMirror

log = logging.getLogger()


class SegmentedTelescopeAPLC:
    """ A segmented telescope with an APLC and actuated segments.

    By default instantiates just with a segmented mirror that can do piston, tip and tilt with the pre-defined methods.
    Use the deformable mirror methods to create more flexible DMs as class attributes:
        self.sm
        self.harris_sm
        self.zernike_mirror
        self.ripple_mirror
        self.dm
    You can retrieve the number of modes/influence functions/actuators of each DM by calling its num_actuators attribute, e.g.
        self.harris_sm.num_actuators
    You can command each DM by passing it an array of length "num_actuators", e.g.
        self.sm.actuators = dm_command

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
    focal_grid : hcipy focal grid
        Focal plane grid to put final image on
    params : dict
        wavelength in m, telescope diameter in m, image radius in lambda/D, FPM radius in lambda/D,
        segment circumscribed diameter in m
    """

    def __init__(self, aper, indexed_aperture, seg_pos, apod, lyotst, fpm, focal_grid, params):
        self.sm = SegmentedMirror(indexed_aperture=indexed_aperture, seg_pos=seg_pos)    # TODO: replace this with None when fully ready to start using create_segmented_mirror()
        self.harris_sm = None
        self.zernike_mirror = None
        self.ripple_mirror = None
        self.dm = None

        self.aperture = aper
        self.apodizer = apod
        self.lyotstop = lyotst
        self.wvln = params['wavelength']
        self.diam = params['diameter']
        self.segment_circumscribed_diameter = params['segment_circumscribed_diameter']
        self.imlamD = params['imlamD']
        self.fpm_rad = params['fpm_rad']
        self.lam_over_d = self.wvln / self.diam
        self.pupil_grid = indexed_aperture.grid
        self.focal_det = focal_grid

        self.coro = hcipy.LyotCoronagraph(self.pupil_grid, fpm, lyotst)
        self.prop = hcipy.FraunhoferPropagator(self.pupil_grid, focal_grid)
        self.coro_no_ls = hcipy.LyotCoronagraph(self.pupil_grid, fpm)
        self.wf_aper = hcipy.Wavefront(aper, wavelength=self.wvln)
        
    def _create_evaluated_segment_grid(self):
        """
        Create a list of segments evaluated on the pupil_grid.

        Returns:
        --------
        seg_evaluated: list
            all segments evaluated individually on self.pupil_grid
        """
        
        # Create single hexagonal segment and full segmented aperture from the single segment, with segment positions
        segment_field_generator = hcipy.hexagonal_aperture(self.segment_circumscribed_diameter, np.pi / 2)
        _aper_in_sm, segs_in_sm = hcipy.make_segmented_aperture(segment_field_generator, self.seg_pos, return_segments=True)

        # Evaluate all segments individually on the pupil_grid
        seg_evaluated = []
        for seg_tmp in segs_in_sm:
            tmp_evaluated = hcipy.evaluate_supersampled(seg_tmp, self.pupil_grid, 1)
            seg_evaluated.append(tmp_evaluated)
        
        return seg_evaluated

    def create_segmented_mirror(self, n_zernikes):
        """
        This creates an actuated segmented mirror from hcipy's DeformableMirror, with n_zernikes Zernike modes per segment.

        Parameters:
        ----------
        n_zernikes : int
            how many Zernikes to create per segment
        """

        seg_evaluated = self._create_evaluated_segment_grid()

        # Create a single segment influence function with all Zernikes n_zernikes
        first_seg = 0    # Create this first influence function on the center segment only
        local_zernike_basis = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                  self.segment_circumscribed_diameter,
                                                                  self.pupil_grid.shifted(-self.seg_pos[first_seg]),
                                                                  starting_mode=1)
        # For all Zernikes, adjust their transformation matrix (by doing what?)
        for zernike_num in range(0, n_zernikes):
            local_zernike_basis._transformation_matrix[:, zernike_num] = seg_evaluated[first_seg]*local_zernike_basis._transformation_matrix[:, zernike_num]

        # Expand the basis of influence functions from one segment to all segments
        for seg_num in range(1, self.nseg):
            local_zernike_basis_tmp = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                          self.segment_circumscribed_diameter,
                                                                          self.pupil_grid.shifted(-self.seg_pos[seg_num]),
                                                                          starting_mode=1)
            # Adjust each transformation matrix again for some reason
            for zernike_num in range(0, n_zernikes):
                local_zernike_basis_tmp._transformation_matrix[:, zernike_num] = seg_evaluated[seg_num] * local_zernike_basis_tmp._transformation_matrix[:, zernike_num]
            local_zernike_basis.extend(local_zernike_basis_tmp)   # extend our basis with this new segment

        self.sm = hcipy.optics.DeformableMirror(local_zernike_basis)

    def create_segmented_harris_mirror(self, filepath, pad_orientation):
        """Generate a basis made of the thermal modes provided by Harris.

        Thermal modes: a, h, i, j, k
        Mechanical modes: e, f, g
        Other modes: b, c, d

        Parameters:
        ----------
        filepath : string
            absolute path to the xls spreadsheet containing the Harris segment modes
        pad_orientation : ndarray
            angles of orientation of the mounting pads of the primary, in rad, one per segment
        """

        # Read the spreadsheet containing the Harris segment modes
        try:
            df = pd.read_excel(filepath)
        except FileNotFoundError:
            log.warning(f"Could not find the Harris spreadsheet under '{filepath}', "
                        f"please double-check your path and that the file exists.")
            return

        # Read all modes as arrays
        valuesA = np.asarray(df.a)
        valuesB = np.asarray(df.b)
        valuesC = np.asarray(df.c)
        valuesD = np.asarray(df.d)
        valuesE = np.asarray(df.e)
        valuesF = np.asarray(df.f)
        valuesG = np.asarray(df.g)
        valuesH = np.asarray(df.h)
        valuesI = np.asarray(df.i)
        valuesJ = np.asarray(df.j)
        valuesK = np.asarray(df.k)
        
        seg_x = np.asarray(df.X)
        seg_y = np.asarray(df.Y)
        harris_seg_diameter = np.max([np.max(seg_x) - np.min(seg_x), np.max(seg_y) - np.min(seg_y)])
        
        pup_dims = self.pupil_grid.dims
        x_grid = np.asarray(df.X) * self.segment_circumscribed_diameter / harris_seg_diameter
        y_grid = np.asarray(df.Y) * self.segment_circumscribed_diameter / harris_seg_diameter
        points = np.transpose(np.asarray([x_grid, y_grid]))

        seg_evaluated = self._create_evaluated_segment_grid()

        harris_base_thermal = []
        for seg_num in range(0, 120):   # TODO: the 120 shouldn't be hard-coded, likely self.nseg, also it's the same like n_segs below

            grid_seg = self.pupil_grid.shifted(-self.seg_pos[seg_num])
            xL1D = np.asarray(grid_seg.x)
            yL1D = np.asarray(grid_seg.y)

            # Rotate the modes grids according to the orientation of the mounting pads
            phi = pad_orientation[seg_num]
            XRot = xL1D * np.cos(phi) + yL1D * np.sin(phi)
            YRot = -xL1D * np.sin(phi) + yL1D * np.cos(phi)

            # TODO: stick the following block in a loop over all modes
            ZA = griddata(points, valuesA, (XRot, YRot), method='linear')
            ZA[np.isnan(ZA)] = 0
            ZA = ZA.ravel() * seg_evaluated[seg_num]
            ZH = griddata(points, valuesH, (XRot, YRot), method='linear')
            ZH[np.isnan(ZH)] = 0
            ZH = ZH.ravel() * seg_evaluated[seg_num]
            ZI = griddata(points, valuesI, (XRot, YRot), method='linear')
            ZI[np.isnan(ZI)] = 0
            ZI = ZI.ravel() * seg_evaluated[seg_num]
            ZJ = griddata(points, valuesJ, (XRot, YRot), method='linear')
            ZJ[np.isnan(ZJ)] = 0
            ZJ = ZJ.ravel() * seg_evaluated[seg_num]
            ZK = griddata(points, valuesK, (XRot, YRot), method='linear')
            ZK[np.isnan(ZK)] = 0
            ZK = ZK.ravel() * seg_evaluated[seg_num]
            harris_base_thermal.append([ZA, ZH, ZI, ZJ, ZK])

        harris_base_thermal = np.asarray(harris_base_thermal)
        n_segs = harris_base_thermal.shape[0]
        n_single_modes = harris_base_thermal.shape[1]
        harris_base_thermal = harris_base_thermal.reshape(n_segs * n_single_modes, pup_dims[0] ** 2)
        harris_thermal_mode_basis = hcipy.ModeBasis(np.transpose(harris_base_thermal), grid=self.pupil_grid)

        self.harris_sm = hcipy.optics.DeformableMirror(harris_thermal_mode_basis)

    def create_global_zernike_mirror(self, n_zernikes):
        """
        Create a Zernike mirror in the pupil plane, with a global Zenrike modal basis of n_zernikes modes.

        Parameters:
        ----------
        n_zernikes : int
            number of Zernikes to enable the Zernike mirror for.
        """
        global_zernike_basis = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                   self.diam,
                                                                   self.pupil_grid,
                                                                   starting_mode=1)
        self.zernike_mirror = hcipy.optics.DeformableMirror(global_zernike_basis)

    def create_ripple_mirror(self, n_fourier):
        """
        Create a Dm that applies Fourier sine and cosine modes in the entrance pupil plane, up to n_fourier cycles per aperture.
        Parameters:
        ----------
        n_fourier : int
            Maximum number for cycles per apertures, use an odd number (!)
        """
        fourier_grid = hcipy.make_pupil_grid(dims=n_fourier, diameter=n_fourier)
        fourier_basis = hcipy.mode_basis.make_fourier_basis(self.pupil_grid, fourier_grid, sort_by_energy=True)
        self.ripple_mirror = hcipy.optics.DeformableMirror(fourier_basis)

    def create_continuous_deformable_mirror(self, n_actuators_across):
        """
        Create a continuous deformable mirror in the pupil plane, with n_actuators_across across the pupil.

        Parameters:
        ----------
        n_actuators_across : int
            number of actuators across the pupil plane
        """
        actuator_spacing = self.diam / n_actuators_across
        influence_functions = hcipy.make_xinetics_influence_functions(self.pupil_grid,
                                                                      n_actuators_across,
                                                                      actuator_spacing)
        self.dm = hcipy.DeformableMirror(influence_functions)

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

        # Create empty field for components that are None
        values = np.ones_like(self.pupil_grid.x)
        transparent_field = hcipy.Field(values, self.pupil_grid)

        # Calculate wavefront after all active pupil components depending on which of the DMs exist
        wf_active_pupil = self.wf_aper

        if self.sm is not None:
            wf_active_pupil = self.sm(wf_active_pupil)
            wf_sm = self.sm(self.wf_aper)
        else:
            wf_sm = hcipy.Wavefront(transparent_field, wavelength=self.wvln)
        if self.harris_sm is not None:
            wf_active_pupil = self.harris_sm(wf_active_pupil)
            wf_harris_sm = self.harris_sm(self.wf_aper)
        else:
            wf_harris_sm = hcipy.Wavefront(transparent_field, wavelength=self.wvln)
        if self.zernike_mirror is not None:
            wf_active_pupil = self.zernike_mirror(wf_active_pupil)
            wf_zm = self.zernike_mirror(self.wf_aper)
        else:
            wf_zm = hcipy.Wavefront(transparent_field, wavelength=self.wvln)
        if self.ripple_mirror is not None:
            wf_active_pupil = self.ripple_mirror(wf_active_pupil)
            wf_ripples = self.ripple_mirror(self.wf_aper)
        else:
            wf_ripples = hcipy.Wavefront(transparent_field, wavelength=self.wvln)
        if self.dm is not None:
            wf_active_pupil = self.dm(wf_active_pupil)
            wf_dm = self.dm(self.wf_aper)
        else:
            wf_dm = hcipy.Wavefront(transparent_field, wavelength=self.wvln)

        # Calculate wavefront after apodizer plane
        wf_apod = apod_prop(wf_active_pupil)

        # Calculate wavefronts of the full coronagraphic propagation
        wf_lyot = self.coro(wf_apod)
        wf_im_coro = self.prop(wf_lyot)

        # Calculate wavefronts in extra planes
        wf_before_fpm = self.prop(wf_apod)
        int_after_fpm = np.log10(wf_before_fpm.intensity / wf_before_fpm.intensity.max()) * fpm_plot  # this is the intensity straight
        wf_before_lyot = self.coro_no_ls(wf_apod)

        # Calculate wavefronts of the reference propagation (no FPM)
        wf_ref_pup = hcipy.Wavefront(self.aperture * self.apodizer * self.lyotstop, wavelength=self.wvln)
        wf_im_ref = self.prop(wf_ref_pup)

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
            hcipy.imshow_field(wf_apod.intensity, cmap='inferno')
            plt.title('Apodizer')

            plt.subplot(3, 4, 8)
            hcipy.imshow_field(wf_before_fpm.intensity / wf_before_fpm.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Before FPM')

            plt.subplot(3, 4, 9)
            hcipy.imshow_field(int_after_fpm / wf_before_fpm.intensity.max(), cmap='inferno')
            plt.title('After FPM')

            plt.subplot(3, 4, 10)
            hcipy.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-3, vmax=1), cmap='inferno')
            plt.title('Before Lyot stop')

            plt.subplot(3, 4, 11)
            hcipy.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(),
                               norm=LogNorm(vmin=1e-3, vmax=1), cmap='inferno', mask=self.lyotstop)
            plt.title('After Lyot stop')

            plt.subplot(3, 4, 12)
            hcipy.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(),
                               norm=LogNorm(vmin=1e-10, vmax=1e-3), cmap='inferno')
            plt.title('Coro image')
            plt.colorbar()

        if return_intermediate == 'intensity':

            # TODO make sure to also return all additional deformable mirrors

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

            # TODO make sure to also return all additional deformable mirrors

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
        """
        Flatten all deformable mirrors in this simulator instance, if they exist.
        """
        if self.sm is not None:
            self.sm.flatten()
        if self.harris_sm is not None:
            self.harris_sm.flatten()
        if self.zernike_mirror is not None:
            self.zernike_mirror.flatten()
        if self.ripple_mirror is not None:
            self.ripple_mirror.flatten()
        if self.dm is not None:
            self.dm.flatten()

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

        # Create a grid for pupil plane optics
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
        self.seg_pos = hcipy.CartesianGrid(hcipy.UnstructuredCoords(poslist))
        self.seg_pos = self.seg_pos.scaled(diameter)

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
