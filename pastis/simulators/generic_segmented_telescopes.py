import logging
import os
from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import hcipy

from hcipy.field import Field
from hcipy.plotting import imshow_field

log = logging.getLogger()


class SegmentedMirror(hcipy.OpticalElement):
    """A segmented mirror from a segmented aperture."""

    def __init__(self, indexed_aperture, seg_pos):
        """
        Parameters
        ----------
        indexed_aperture : Field
            The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
            segment identification. Segment gaps must be strictly zero.
        seg_pos : CartesianGrid(UnstructuredCoords)
            Segment positions of the aperture.
        """
        self.ind_aper = indexed_aperture
        self.segnum = len(seg_pos.x)
        self.segmentlist = np.arange(1, self.segnum + 1)
        self._coef = np.zeros((self.segnum, 3))
        self.seg_pos = seg_pos
        self.input_grid = indexed_aperture.grid
        self._last_npix = np.nan  # see _setup_grids for this
        self._surface = None

    def forward(self, wavefront):
        """Propagate a wavefront through the segmented mirror.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The reflected wavefront.
        """

        wf = wavefront.copy()
        wf.electric_field *= np.exp(2j * self.surface * wavefront.wavenumber)
        return wf

    def backward(self, wavefront):
        """Propagate a wavefront backwards through the deformable mirror.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The reflected wavefront.
        """

        wf = wavefront.copy()
        wf.electric_field *= np.exp(-2j * self.surface * wavefront.wavenumber)
        return wf

    @property
    def surface(self):
        """The surface of the segmented mirror in meters, the full surface as a Field."""
        if self._surface is None:
            self._surface = self.apply_coef()
        return self._surface

    @property
    def coef(self):
        """The surface shape of the deformable mirror, in meters and radians; PTT segment coefficients."""
        self._surface = None
        return self._coef

    def show_numbers(self):
        """ Display the mirror pupil with numbered segments."""
        imshow_field(self.ind_aper)
        for i, par in enumerate(self.seg_pos):
            plt.annotate(s=i + 1, xy=par, xytext=par, color='white', fontweight='bold')  # TODO: scale text size by segment size

    def flatten(self):
        """Flatten the DM by setting all segment coefficients to zero."""
        self._surface = None
        self._coef[:] = 0

    def set_segment(self, segid, piston, tip, tilt):
        """Set an individual segment of the DM.

        Piston in meter of surface, tip and tilt in radians of surface.

        Parameters
        -------------
        segid : integer
            Index of the segment you wish to control, starting at 1 (center would  be 0, but doesn't exist)
        piston, tip, tilt : floats, meters and radians
            Piston (in meters) and tip and tilt (in radians)
        """

        self._surface = None
        self._coef[segid - 1] = [piston, tip, tilt]

    def _setup_grids(self):
        """Set up the grids to compute the segmented mirror surface into.

        This is relatively slow, but we only need to do this once for
        each size of input grids.
        """

        npix = self.ind_aper.shaped.shape[0]
        if npix == self._last_npix:
            return
        else:
            self._last_npix = npix

        x, y = self.input_grid.coords

        self._seg_x = np.zeros_like(x)
        self._seg_y = np.zeros_like(y)
        self._seg_indices = dict()

        for i in self.segmentlist:
            wseg = np.where(self.ind_aper == i)
            self._seg_indices[i] = wseg

            cenx, ceny = self.seg_pos.points[i - 1]

            self._seg_x[wseg] = x[wseg] - cenx
            self._seg_y[wseg] = y[wseg] - ceny

    def apply_coef(self):
        """Apply the DM shape from its own segment coefficients to make segmented mirror surface."""

        self._setup_grids()
        keep_surf = np.zeros_like(self._seg_x)
        for i in self.segmentlist:
            wseg = self._seg_indices[i]
            keep_surf[wseg] = (self._coef[i - 1, 0] +
                               self._coef[i - 1, 1] * self._seg_x[wseg] +
                               self._coef[i - 1, 2] * self._seg_y[wseg])
        return Field(keep_surf, self.input_grid)

    def phase_for(self, wavelength):
        """Get the phase that is added to a wavefront with a specified wavelength.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the phase deformation.

        Returns
        -------
        Field
            The calculated phase deformation.
        """
        return 2 * self.surface * 2 * np.pi / wavelength


def load_segment_centers(input_dir, aper_ind_path, nseg, diameter):
    """Load segment positions from fits header

    Parameters
    ----------
    input_dir : string
        absolute path to input directory
    aper_ind_path : string
        relative path and filename of indexed aperture file
    nseg : int
        total number of segments in the pupil
    diameter : float
        pupil diameter

    Returns
    -------
    seg_pos : hcipy.CartesianGrid
        grid of segment centers
    """

    hdr = fits.getheader(os.path.join(input_dir, aper_ind_path))
    poslist = []
    for i in range(nseg):
        segname = 'SEG' + str(i + 1)
        xin = hdr[segname + '_X']
        yin = hdr[segname + '_Y']
        poslist.append((xin, yin))
    poslist = np.transpose(np.array(poslist))
    seg_pos = hcipy.CartesianGrid(hcipy.UnstructuredCoords(poslist))
    seg_pos = seg_pos.scaled(diameter)

    return seg_pos


class Telescope:
    """A simple telescope with active components in the pupil plane (DMs), but without actively controlled segments.

    This class can take an arbitrary telescope aperture as input and will create a telescope object out of it. This can
    be monolithic or segmented or other apertures, but the DMs that the class contains will always be actinvn on the
    global pupil.
    By default, it instantiates with none of these DMs, they can each be created with their respective "create_...()" functions:
        self.zernike_mirror
        self.ripple_mirror
        self.dm
    You can retrieve the number of total modes/influence functions/actuators of each DM by calling its num_actuators attribute, e.g.
        self.zernike_mirror.num_actuators
    You can command each DM by passing it an array of length "num_actuators", e.g.
        self.zernike_mirror.actuators = dm_command

    Attributes:
    ----------
    wvln : float
        Wavelength in meters
    diam : float
        Telescope diameter in meters
    aperture : hcipy.Field
        Telescope aperture
    focal_det : hcipy.Grid
        Grid of focal plane coordinates that the final detector image gets sampled on.
    sampling : float
        Sampling in focal plane in pixels per lambda/D
    imlamD : float
        Detector image half-size in lambda/D
    lam_over_d :
        lambda/D of the system in radians
    pupil_grid : hcipy.Grid
        Grid of pupil plane coordinates that all pupil plane optics get sampled on.
    prop : hcipy.FraunhoferPropagator
        Propagation method for Fraunhofer propagation between a pupil plane and a focal plane
    wf_aper : hcipy.Wavefront
        E-field on the segmented primary
    norm_phot : float
        Method that performs slef.prop, but normalized ot one photon in the pupil.
    """

    def __init__(self, wvln, diameter, aper, focal_grid, sampling, imlamD):
        """
        Parameters:
        ----------
        wvln : float
            Wavelength in meters
        diameter : float
            Telescope diameter in meters
        aper : Field
            Telescope aperture
        focal_grid : hcipy focal grid
            Focal plane grid to put final image on
        sampling : float
            Sampling in focal plane in pixels per lambda/D
        imlamD : float
            Detector image half-size in lambda/D
        """

        self.wvln = wvln
        self.diam = diameter
        self.aperture = aper
        self.focal_det = focal_grid
        self.sampling = sampling
        self.imlamD = imlamD
        self.lam_over_d = wvln / diameter
        self.pupil_grid = aper.grid

        self.prop = hcipy.FraunhoferPropagator(self.pupil_grid, self.focal_det)
        self.wf_aper = hcipy.Wavefront(self.aperture, wavelength=self.wvln)
        self.norm_phot = 1 / np.sqrt(np.sum(self.wf_aper.intensity))

        self.zernike_mirror = None
        self.ripple_mirror = None
        self.dm = None
        self.zwfs = None

    def prop_norm_one_photon(self, input_efield):
        """Perform a Fraunhofer propagation, normalized to one photon.

        Parameters:
        ----------
        input_efield : hcipy.Wavefront

        Returns
        --------
        normalized_efield : hcipy.Wavefront
        """

        norm_fac = np.max(self.focal_det.x) * self.pupil_grid.dims[0] / np.max(self.pupil_grid.x) / self.focal_det.dims[0]
        prop_before_norm = self.prop(input_efield)
        normalize = norm_fac * prop_before_norm.electric_field
        normalized_efield = hcipy.Wavefront(normalize, self.wvln)
        return normalized_efield

    def create_global_zernike_mirror(self, n_zernikes):
        """Create a Zernike mirror in the pupil plane, with a global Zenrike modal basis of n_zernikes modes.

        Parameters:
        ----------
        n_zernikes : int
            Number of Zernikes to enable the Zernike mirror for.
        """

        global_zernike_basis = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                   self.diam,
                                                                   self.pupil_grid,
                                                                   starting_mode=1)
        self.zernike_mirror = hcipy.optics.DeformableMirror(global_zernike_basis)

    def create_ripple_mirror(self, n_fourier):
        """Create a DM that applies Fourier sine and cosine modes in the entrance pupil plane, up to n_fourier cycles per aperture.

        Parameters:
        ----------
        n_fourier : int
            Maximum number for cycles per aperture, use an odd number (!)
        """

        fourier_grid = hcipy.make_pupil_grid(dims=n_fourier, diameter=n_fourier)   # TODO: should it be diameter=self.diam instead?
        fourier_basis = hcipy.mode_basis.make_fourier_basis(self.pupil_grid, fourier_grid, sort_by_energy=True)
        self.ripple_mirror = hcipy.optics.DeformableMirror(fourier_basis)

    def create_continuous_deformable_mirror(self, n_actuators_across):
        """Create a continuous deformable mirror in the pupil plane, with n_actuators_across across the pupil.

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

    def remove_global_zernike_mirror(self):
        """Remove the global Zernike mirror as class attribute."""
        self.zernike_mirror = None

    def remove_ripple_mirror(self):
        """Remove the high-spatial frequency ripple mirror as class attribute."""
        self.ripple_mirror = None

    def remove_continuous_deformable_mirror(self):
        """Remove the continuous deformable mirror as class attribute."""
        self.dm = None

    def flatten(self):
        """Flatten all deformable mirrors in this simulator instance, if they exist."""
        if self.zernike_mirror is not None:
            self.zernike_mirror.flatten()
        if self.ripple_mirror is not None:
            self.ripple_mirror.flatten()
        if self.dm is not None:
            self.dm.flatten()

    def _create_transparent_plane_and_active_pupil(self, norm_one_photon):
        """Create empty field and WF on active pupil.

        Create a transparent field object that can be used in propagation planes that are not set up.
        Also create the WF on the full active pupil, depending on normalization choice.

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        transparent_field : hcipy.Field
            Transparent field (all ones).
        wf_active_pupil : hcipy.Wavefront
            Wavefront of all active conjugate pupils combined into one.
        """

        # Create empty field for components that are None
        values = np.ones_like(self.pupil_grid.x)
        transparent_field = hcipy.Field(values, self.pupil_grid)

        # Create E-field on primary mirror
        if norm_one_photon:
            wf_active_pupil = hcipy.Wavefront(self.norm_phot * self.wf_aper.electric_field, self.wvln)
        else:
            wf_active_pupil = self.wf_aper

        return transparent_field, wf_active_pupil

    def _propagate_active_pupils(self, norm_one_photon=False):
        """Propagate aperture wavefront "through" all active entrance pupil elements (DMs).

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        wf_active_pupil, wf_zm, wf_ripples, wf_dm : hcipy.Wavefronts
            E-field after each respective DM individually; all DMs in the case of wf_active_pupil.
       """

        transparent_field, wf_active_pupil = self._create_transparent_plane_and_active_pupil(norm_one_photon)

        # Calculate wavefront after all active pupil components depending on which of the DMs exist
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

        return wf_active_pupil, wf_zm, wf_ripples, wf_dm, transparent_field

    def calc_psf(self, display_intermediate=False, return_intermediate=None, norm_one_photon=False):
        """Calculate the PSF of this telescope, and return optionally all E-fields.

        Parameters
        ----------
        display_intermediate : bool
            Whether to display images of all planes.
        return_intermediate : string
            default None; if "efield", will also return E-fields of each plane and DM
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
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

        if norm_one_photon:
            prop_method = self.prop_norm_one_photon
        else:
            prop_method = self.prop

        wf_image = prop_method(wf_active_pupil)

        if display_intermediate:
            plt.figure(figsize=(10, 15))

            plt.subplot(3, 2, 1)
            hcipy.imshow_field(self.wf_aper.intensity, mask=self.aperture, cmap='Greys_r')
            plt.title('Primary mirror')

            plt.subplot(3, 2, 2)
            hcipy.imshow_field(wf_zm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Global Zernike phase')

            plt.subplot(3, 2, 3)
            hcipy.imshow_field(wf_dm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Deformable mirror phase')

            plt.subplot(3, 2, 4)
            hcipy.imshow_field(wf_ripples.phase, mask=self.aperture, cmap='RdBu')
            plt.title('High modes mirror phase')

            plt.subplot(3, 2, 5)
            hcipy.imshow_field(wf_active_pupil.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Total phase in entrance pupil')

            plt.subplot(3, 2, 6)
            hcipy.imshow_field(wf_image.intensity / wf_image.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Focal plane image')

        if return_intermediate == 'efield':
            # Return the E-fields in all planes; except intensity in focal plane after FPM
            intermediates = {'zernike_mirror': wf_zm,
                             'dm': wf_dm,
                             'ripple_mirror': wf_ripples,
                             'active_pupil': wf_active_pupil}
            return wf_image, intermediates

        return wf_image.intensity

    def create_zernike_wfs(self, step=None, spot_diam=None, spot_points=None):
        """Create a Zernike wavefront sensor object. """

        if step is None:
            step = np.pi / 2    # original value for LUVOIR-A
        if spot_diam is None:
            spot_diam = 1.06    # original value for LUVOIR-A
        if spot_points is None:
            spot_points = 128    # original value for LUVOIR-A

        self.zwfs = hcipy.wavefront_sensing.ZernikeWavefrontSensorOptics(self.pupil_grid,
                                                                         phase_step=step,
                                                                         phase_dot_diameter=spot_diam,
                                                                         num_pix=spot_points,
                                                                         pupil_diameter=self.diam,
                                                                         reference_wavelength=self.wvln)

    def calc_out_of_band_wfs(self, norm_one_photon=False):
        """Propagate pupil through an out-of-band wavefront sensor.

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        ob_wfs : hcipy.Wavefront
            E-field on OBWFS detector
        """

        # If ZWFS hasn't been created yet, do it now
        if self.zwfs is None:
            self.create_zernike_wfs()

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_zm, wf_ripples, wf_dm, _tr = self._propagate_active_pupils(norm_one_photon)

        ob_wfs = self.zwfs(wf_active_pupil)
        return ob_wfs


class SegmentedTelescope(Telescope):
    """A segmented telescope with active components in the pupil plane (DMs).

    By default, instantiates just with a segmented mirror that can do piston, tip and tilt with the pre-defined methods.
    Use the deformable mirror methods to create more flexible DMs as class attributes, on top of DMs inherited from "Telescope":
        self.sm
        self.harris_sm
    You can retrieve the number of total modes/influence functions/actuators of each DM by calling its num_actuators attribute, e.g.
        self.harris_sm.num_actuators
    You can command each DM by passing it an array of length "num_actuators", e.g.
        self.sm.actuators = dm_command

    The segments are numbered following their respective index in the indexed aperture input file.

    Attributes
    ----------
    aper_ind : hcipy.Field
        Indexed telescope aperture
    seg_pos : CartesianGrid(UnstructuredCoords)
        Segment positions of the aperture
    segment_circumscribed_diameter : float
        Circumscribed diameter of an individual segment in meters
    nseg : int
        Number of active segments on telescope primary mirror
    center_segment : bool
        Whether the center segment is actively controllable (unobscured) or not
    """

    def __init__(self, indexed_aper, seg_pos, seg_diameter, center_segment=False, **kwargs):
        """
        Parameters
        ----------
        indexed_aper : Field
            The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
            segment identification. Segment gaps must be strictly zero.
        seg_pos : CartesianGrid(UnstructuredCoords)
            Segment positions of the aperture
        seg_diameter : float
            Circumscribed diameter of an individual segment in meters
        center_segment : Bool
            ...
        *kwargs :
            keyword arguments passed through to "Telescope" parent class, see docs there
        """

        super().__init__(**kwargs)

        self.aper_ind = indexed_aper
        self.seg_pos = seg_pos
        self.segment_circumscribed_diameter = seg_diameter
        self.nseg = seg_pos.size
        self.center_segment = center_segment

        self.sm = SegmentedMirror(indexed_aperture=indexed_aper, seg_pos=seg_pos)    # TODO: replace this with None when fully ready to start using create_segmented_mirror()
        self.harris_sm = None

    def set_segment(self, segid, piston, tip, tilt):
        """Set an individual segment of the SegmentedMirror to a piston/tip/tilt command.

        This method only works with a segmented DM of type pastis.simulators.generic_segmented_telescopes.SegmentedMirror
        and only exists for back-compatibility.

        Parameters
        ----------
        segid : int
            Id number of the segment you want to set. Center segment is always 0, whether it is obscured or not.
        piston : float
            Piston aberration amplitude in meters rms of surface.
        tip : float
            Tip aberration amplitude in meters rms of surface.
        tilt : float
            Tilt aberration amplitude in meters rms of surface.
        """

        if not isinstance(self.sm, SegmentedMirror):
            raise TypeError("This function is only for usage with a segmented mirror of type "
                            "'pastis.simulators.generic_segmented_telescopes.SegmentedMirror'. You are currently "
                            "using a multi-mode Zernike mirror. Please use `set_sm_segment()` instead.")
        self.sm.set_segment(segid, piston, tip, tilt)

    def _create_evaluated_segment_grid(self):
        """Create a list of segments evaluated on the pupil_grid.

        Returns
        --------
        seg_evaluated: list
            all segments evaluated individually on self.pupil_grid
        """

        # Create single hexagonal segment and full segmented aperture from the single segment, with segment positions
        segment_field_generator = hcipy.hexagonal_aperture(self.segment_circumscribed_diameter, np.pi / 2)
        _aper_in_sm, segs_in_sm = hcipy.make_segmented_aperture(segment_field_generator, self.seg_pos,
                                                                return_segments=True)

        # Evaluate all segments individually on the pupil_grid
        seg_evaluated = []
        for seg_tmp in segs_in_sm:
            tmp_evaluated = hcipy.evaluate_supersampled(seg_tmp, self.pupil_grid, 1)
            seg_evaluated.append(tmp_evaluated)

        return seg_evaluated

    def create_segmented_mirror(self, n_zernikes):
        """Create an actuated segmented mirror from hcipy's DeformableMirror, with n_zernikes Zernike modes per segment.

        Parameters
        ----------
        n_zernikes : int
            how many Zernikes to create per segment
        """

        self.seg_n_zernikes = n_zernikes
        seg_evaluated = self._create_evaluated_segment_grid()

        # Create a single segment influence function with all Zernikes n_zernikes
        first_seg = 0  # Create this first influence function on the first segment only
        local_zernike_basis = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                  self.segment_circumscribed_diameter,
                                                                  self.pupil_grid.shifted(-self.seg_pos[first_seg]),
                                                                  starting_mode=1, radial_cutoff=False)
        # # For all Zernikes on this first segment, cut them to the actual segment support
        for zernike_num in range(0, n_zernikes):
            local_zernike_basis._transformation_matrix[:, zernike_num] = seg_evaluated[first_seg] * local_zernike_basis._transformation_matrix[:, zernike_num]

        # Expand the basis of influence functions from one segment to all segments
        for seg_num in range(1, self.nseg):
            local_zernike_basis_tmp = hcipy.mode_basis.make_zernike_basis(n_zernikes,
                                                                          self.segment_circumscribed_diameter,
                                                                          self.pupil_grid.shifted(-self.seg_pos[seg_num]),
                                                                          starting_mode=1, radial_cutoff=False)
            # Adjust each transformation matrix again for some reason
            for zernike_num in range(0, n_zernikes):
                local_zernike_basis_tmp._transformation_matrix[:, zernike_num] = seg_evaluated[seg_num] * local_zernike_basis_tmp._transformation_matrix[:, zernike_num]
            local_zernike_basis.extend(local_zernike_basis_tmp)  # extend our basis with this new segment

        self.sm = hcipy.optics.DeformableMirror(local_zernike_basis)

    def set_sm_segment(self, segid, zernike_number, amplitude, override=False):
        """Set an individual segment of the multi-mode segmented DM to a single Zernike mode.

        This works only on the new multi-mode segmented DM that is a hcipy.optics.DeformableMirror. You can use
        Zernikes up to the number you created the SM with. For commanding individual segments of the internal
        SegmentedMirror, use self.set_segment(), which exists purely for back-compatibility.

        Parameters
        ----------
        segid : int
            Id number of the segment you want to set. Center segment is always 0, whether it is obscured or not.
        zernike_number : int
            Which local Zernike mode to apply to segment ID segid. Ordered after Noll and they start with 0 (piston).
        amplitude : float
            Aberration amplitude in meters rms of surface.
        override : bool
            Whether to override all other segment commands with zero, default False, which means the new segment
            aberration will be added to what is already on the segmented mirror.
        """

        if not isinstance(self.sm, hcipy.optics.DeformableMirror):
            raise TypeError("This function is only for usage with the multi-mode segmented DM. You are currently "
                            "using a 'pastis.simulators.generic_segmented_telescopes.SegmentedMirror'.")

        if segid == 0 and not self.center_segment:
            raise NotImplementedError("'self.center_segment' is set to 'False', so there is not center segment to command.")

        segment_cutoff_id = self.nseg if self.center_segment is False else self.nseg - 1
        if segid > segment_cutoff_id:
            raise NotImplementedError(f"Your telescope has {self.nseg} active segments and the highest existing "
                                      f"segment index is segment number {segment_cutoff_id}; you requested {segid}.")

        if zernike_number >= self.seg_n_zernikes:
            raise NotImplementedError(f"'self.sm' has only been instantiated for {self.seg_n_zernikes} Zernike "
                                      f"modes per segment, indexed with 0.")

        if not self.center_segment:
            segid -= 1

        if override:
            new_command = np.zeros(self.sm.num_actuators)
        else:
            new_command = np.copy(self.sm.actuators)
        new_command[self.seg_n_zernikes * segid + zernike_number] = amplitude
        self.sm.actuators = new_command

    def create_segmented_harris_mirror(self, filepath, pad_orientation, thermal=True, mechanical=True, other=True):
        """Create an actuated segmented mirror with a modal basis made of the thermal modes provided by Harris.

        Thermal modes: a, h, i, j, k
        Mechanical modes: e, f, g
        Other modes: b, c, d

        If all modes are created, they will be ordered as:
        a, h, i, j, k, e, f, g, b, c, d
        If only a subset is created, the ordering will be retained but the non-chosen modes dropped.

        Parameters
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
        self.harris_seg_diameter = np.max([np.max(seg_x) - np.min(seg_x), np.max(seg_y) - np.min(seg_y)])

        pup_dims = self.pupil_grid.dims
        x_grid = np.asarray(df.X) * self.segment_circumscribed_diameter / self.harris_seg_diameter
        y_grid = np.asarray(df.Y) * self.segment_circumscribed_diameter / self.harris_seg_diameter
        points = np.transpose(np.asarray([x_grid, y_grid]))

        seg_evaluated = self._create_evaluated_segment_grid()

        def _transform_harris_mode(values, xrot, yrot, points, seg_evaluated, seg_num):
            """ Take imported Harris mode data and transform into a segment mode on our aperture. """
            zval = griddata(points, values, (xrot, yrot), method='linear')
            zval[np.isnan(zval)] = 0
            zval = zval.ravel() * seg_evaluated[seg_num]
            return zval

        harris_base = []
        for seg_num in range(0, self.nseg):
            mode_set_per_segment = []

            grid_seg = self.pupil_grid.shifted(-self.seg_pos[seg_num])
            x_line_grid = np.asarray(grid_seg.x)
            y_line_grid = np.asarray(grid_seg.y)

            # Rotate the modes grids according to the orientation of the mounting pads
            phi = pad_orientation[seg_num]
            x_rotation = x_line_grid * np.cos(phi) + y_line_grid * np.sin(phi)
            y_rotation = -x_line_grid * np.sin(phi) + y_line_grid * np.cos(phi)

            # Transform all needed Harris modes from data to modes on our segmented aperture
            # Use only the sets of modes that have been specified in the input parameters
            if thermal:
                ZA = _transform_harris_mode(valuesA, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZH = _transform_harris_mode(valuesH, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZI = _transform_harris_mode(valuesI, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZJ = _transform_harris_mode(valuesJ, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZK = _transform_harris_mode(valuesK, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                mode_set_per_segment.extend([ZA, ZH, ZI, ZJ, ZK])
            if mechanical:
                ZE = _transform_harris_mode(valuesE, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZF = _transform_harris_mode(valuesF, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZG = _transform_harris_mode(valuesG, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                mode_set_per_segment.extend([ZE, ZF, ZG])
            if other:
                ZB = _transform_harris_mode(valuesB, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZC = _transform_harris_mode(valuesC, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                ZD = _transform_harris_mode(valuesD, x_rotation, y_rotation, points, seg_evaluated, seg_num)
                mode_set_per_segment.extend([ZB, ZC, ZD])

            harris_base.append(mode_set_per_segment)

        # Create full mode basis of selected Harris modes on all segments
        harris_base = np.asarray(harris_base)
        self.n_harris_modes = harris_base.shape[1]
        harris_base = harris_base.reshape(self.nseg * self.n_harris_modes, pup_dims[0] ** 2)
        harris_mode_basis = hcipy.ModeBasis(np.transpose(harris_base), grid=self.pupil_grid)

        self.harris_sm = hcipy.optics.DeformableMirror(harris_mode_basis)

    def set_harris_segment(self, segid, mode_number, amplitude, override=False):
        """Set an individual segment of the Harris segmented mirror to a single Harris mode.

        Parameters
        ----------
        segid : int
            Id number of the segment you want to set. Center segment is always 0, whether it is obscured or not.
        mode_number : int
            Which local Harris mode to apply to segment with ID segid. Ordering:
                Thermal modes: 0, 7, 8, 9, 10
                Mechanical modes: 4, 5, 6
                Other modes: 1, 2, 3
        amplitude : float
            Aberration amplitude in  ? meters of surface.   # FIXME: meters? surface? rms or ptv?
        override : bool
            Whether to override all other segment commands with zero, default False, which means the new segment
            aberration will be added to what is already on the segmented mirror.
        """

        if mode_number >= self.n_harris_modes:
            raise NotImplementedError(f"'self.harris_sm' has only been instantiated for {self.n_harris_modes} modes "
                                      f"per segment, indexed with 0.")

        if segid == 0 and not self.center_segment:
            raise NotImplementedError("'self.center_segment' is set to 'False', so there is not center segment to command.")

        segment_cutoff_id = self.nseg if self.center_segment is False else self.nseg - 1
        if segid > segment_cutoff_id:
            raise NotImplementedError(f"Your telescope has {self.nseg} active segments and the highest existing "
                                      f"segment index is segment number {segment_cutoff_id}; you requested {segid}.")

        if not self.center_segment:
            segid -= 1

        if override:
            new_command = np.zeros(self.harris_sm.num_actuators)
        else:
            new_command = np.copy(self.harris_sm.actuators)
        new_command[self.n_harris_modes * segid + mode_number] = amplitude
        self.harris_sm.actuators = new_command

    def remove_segmented_mirror(self):
        """Remove the segmented mirror with local Zernikes as class attribute, replace with PTT segmented mirror."""
        self.sm = SegmentedMirror(indexed_aperture=self.aper_ind, seg_pos=self.seg_pos)

    def remove_segmented_harris_mirror(self):
        """Remove the segmented Harris mirror as class attribute."""
        self.harris_sm = None

    def flatten(self):
        """Flatten all deformable mirrors in this simulator instance, if they exist."""

        if self.sm is not None:
            self.sm.flatten()
        if self.harris_sm is not None:
            self.harris_sm.flatten()
        super().flatten()

    def _propagate_active_pupils(self, norm_one_photon=False):
        """Propagate aperture wavefront "through" all active entrance pupil elements (DMs).

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm : hcipy.Wavefronts
            E-field after each respective DM individually; all DMs in the case of wf_active_pupil.
       """

        wf_active_pupil, wf_zm, wf_ripples, wf_dm, transparent_field = super()._propagate_active_pupils(norm_one_photon)

        # Calculate wavefront after all active pupil components depending on which of the DMs exist
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

        return wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm

    def calc_psf(self, display_intermediate=False, return_intermediate=None, norm_one_photon=False):
        """Calculate the PSF of this segmented telescope, and return optionally all E-fields.

        Parameters
        ----------
        display_intermediate : bool
            Whether to display images of all planes.
        return_intermediate : string
            default None; if "efield", will also return E-fields of each plane and DM
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        wf_image.intensity : Field
            returned if return_intermediate=None (default)
        wf_image : hcipy.Wavefront
            returned if return_intermediate='efield'
        intermediates : dict
            Dictionary containing the Wavefronts of all the planes, returned if return_intermediate='efield'
        """

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm = self._propagate_active_pupils(norm_one_photon)

        if norm_one_photon:
            prop_method = self.prop_norm_one_photon
        else:
            prop_method = self.prop

        wf_image = prop_method(wf_active_pupil)

        if display_intermediate:
            plt.figure(figsize=(15, 15))

            plt.subplot(3, 3, 1)
            hcipy.imshow_field(self.wf_aper.intensity, mask=self.aperture, cmap='Greys_r')
            plt.title('Primary mirror')

            plt.subplot(3, 3, 2)
            hcipy.imshow_field(wf_sm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Segmented mirror phase')

            plt.subplot(3, 3, 3)
            hcipy.imshow_field(wf_zm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Global Zernike phase')

            plt.subplot(3, 3, 4)
            hcipy.imshow_field(wf_dm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Deformable mirror phase')

            plt.subplot(3, 3, 5)
            hcipy.imshow_field(wf_harris_sm.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Harris mode mirror phase')

            plt.subplot(3, 3, 6)
            hcipy.imshow_field(wf_ripples.phase, mask=self.aperture, cmap='RdBu')
            plt.title('High modes mirror phase')

            plt.subplot(3, 3, 7)
            hcipy.imshow_field(wf_active_pupil.phase, mask=self.aperture, cmap='RdBu')
            plt.title('Total phase in entrance pupil')

            plt.subplot(3, 3, 8)
            hcipy.imshow_field(wf_image.intensity / wf_image.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Focal plane image')

        if return_intermediate == 'efield':
            # Return the E-fields in all planes; except intensity in focal plane after FPM
            intermediates = {'seg_mirror': wf_sm,
                             'zernike_mirror': wf_zm,
                             'dm': wf_dm,
                             'harris_seg_mirror': wf_harris_sm,
                             'ripple_mirror': wf_ripples,
                             'active_pupil': wf_active_pupil}
            return wf_image, intermediates

        return wf_image.intensity

    def calc_out_of_band_wfs(self, norm_one_photon=False):
        """Propagate pupil through an out-of-band wavefront sensor.

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        ob_wfs : hcipy.Wavefront
            E-field on OBWFS detector
        """

        # If ZWFS hasn't been created yet, do it now
        if self.zwfs is None:
            self.create_zernike_wfs()

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm = self._propagate_active_pupils(norm_one_photon)

        ob_wfs = self.zwfs(wf_active_pupil)
        return ob_wfs


class SegmentedAPLC(SegmentedTelescope):
    """A segmented Apodized Pupil Lyot Coronagraph (APLC)

    Attributes
    ----------
    apodizer : hcipy.Field
        Apodizer
    lyotstop : hcipy.Field
        Lyot Stop
    fpm : hcipy.Field
        Focal Plane Mask
    fpm_rad : float
        Radius of your FPM in lambda/D
    coro : hcipy.LyotCoronagraph
        Lyot-style coronagraph propagator, includes multiplication by Lyot stop
    coro_no_ls : hcipy.LyotCoronagraph
        Lyot-style coronagraph propagator, excludes multiplication by Lyot stop
    iwa : float
        Inner working angle of the APLC
    owa : float
        Outer working angle of the APLC
    dh_mask : hcipy.Field
        DH mask of your APLC as a boolean array
    and all attributes of SegmentedTelescope
    """

    def __init__(self, apod, lyot_stop, fpm, fpm_rad, iwa, owa, **kwargs):
        """
        Parameters
        ----------
        apod : Field
            Apodizer
        lyot_stop : Field
            Lyot Stop
        fpm : Field
            Focal plane mask (FPM)
        fpm_rad : float
            FPM radius in lambda/D
        iwa : float
            Inner working angle in lambda/D
        owa : float
            Outer working angle in lambda/D
        **kwargs :
            Keyword arguments passed through to SegmentedTelescope, see documentation there
        """

        self.apodizer = apod
        self.lyotstop = lyot_stop
        self.fpm = fpm
        self.fpm_rad = fpm_rad
        super().__init__(**kwargs)

        self.coro = hcipy.LyotCoronagraph(self.pupil_grid, fpm, lyot_stop)
        self.coro_no_ls = hcipy.LyotCoronagraph(self.pupil_grid, fpm)
        self.iwa = iwa
        self.owa = owa
        dh_outer = hcipy.circular_aperture(2 * owa * self.lam_over_d)(self.focal_det)
        dh_inner = hcipy.circular_aperture(2 * iwa * self.lam_over_d)(self.focal_det)
        self.dh_mask = (dh_outer - dh_inner).astype('bool')

    def calc_psf(self, ref=False, display_intermediate=False, return_intermediate=None, norm_one_photon=False):
        """Calculate the PSF of the segmented APLC, normalized to contrast units. Optionally return reference (direct
        PSF) and/or E-fields in all planes.

        Parameters
        ----------
        ref : bool
            Keyword for additionally returning the reference PSF without the FPM.
        display_intermediate : bool
            Whether to display images of all planes.
        return_intermediate : string
            Either 'intensity', return the intensity in all planes; except phase on the SM (first plane)
            or 'efield', return the E-fields in all planes. Default none.
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        wf_im_coro.intensity : Field
            Coronagraphic image, normalized to contrast units by max of reference image (even when ref
            not returned).
        wf_im_ref.intensity : Field, optional
            Reference image without FPM.
        intermediates : dict of Fields, optional
            Intermediate plane intensity images; except for phases on DMs
        wf_im_coro : Wavefront
            Wavefront in last focal plane.
        wf_im_ref : Wavefront, optional
            Wavefront of reference image without FPM.
        intermediates : dict of Wavefronts, optional
            Intermediate plane E-fields; except intensity in focal plane after FPM.
        """

        if isinstance(return_intermediate, bool):
            raise TypeError(f"'return_intermediate' needs to be 'efield' or 'intensity' if you want all "
                            f"E-fields returned by 'calc_psf()'.")

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm = self._propagate_active_pupils(norm_one_photon)

        if norm_one_photon:
            prop_method = self.prop_norm_one_photon
            norm_factor = self.norm_phot
        else:
            prop_method = self.prop
            norm_factor = 1

        # Create fake FPM for plotting
        fpm_plot = 1 - hcipy.circular_aperture(2 * self.fpm_rad * self.lam_over_d)(self.focal_det)

        # Create apodizer as hcipy.Apodizer() object to be able to propagate through it
        apod_prop = hcipy.Apodizer(self.apodizer)

        # Calculate wavefront after apodizer plane
        wf_apod = apod_prop(wf_active_pupil)

        # Calculate wavefronts of the full coronagraphic propagation
        wf_lyot = self.coro(wf_apod)
        wf_im_coro = prop_method(wf_lyot)

        # Calculate wavefronts in extra planes
        wf_before_fpm = prop_method(wf_apod)
        int_after_fpm = np.log10(wf_before_fpm.intensity / wf_before_fpm.intensity.max()) * fpm_plot  # this is the intensity straight
        wf_before_lyot = self.coro_no_ls(wf_apod)

        # Calculate wavefronts of the reference propagation (no FPM)
        wf_ref_pup = hcipy.Wavefront(norm_factor * self.aperture * self.apodizer * self.lyotstop, wavelength=self.wvln)
        wf_im_ref = prop_method(wf_ref_pup)

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

            # Return the intensity in all planes; except phases on all DMs, and combined phase from active pupils
            intermediates = {'seg_mirror': wf_sm.phase,
                             'zernike_mirror': wf_zm.phase,
                             'dm': wf_dm.phase,
                             'harris_seg_mirror': wf_harris_sm.phase,
                             'ripple_mirror': wf_ripples.phase,
                             'active_pupil': wf_active_pupil.phase,
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
                             'zernike_mirror': wf_zm,
                             'dm': wf_dm,
                             'harris_seg_mirror': wf_harris_sm,
                             'ripple_mirror': wf_ripples,
                             'active_pupil': wf_active_pupil,
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

    def calc_low_order_wfs(self, norm_one_photon=False):
        """Propagate pupil through a low-order wavefront sensor.

        Parameters
        ----------
        norm_one_photon : bool
            Whether to normalize the returned E-fields and intensities to one photon in the entrance pupil.

        Returns
        --------
        lowfs : hcipy.Wavefront
            E-field on LOWFS detector
        """

        # If ZWFS hasn't been created yet, do it now
        if self.zwfs is None:
            self.create_zernike_wfs()

        # Propagate aperture wavefront "through" all active entrance pupil elements (DMs)
        wf_active_pupil, wf_sm, wf_harris_sm, wf_zm, wf_ripples, wf_dm = self._propagate_active_pupils(norm_one_photon)

        # Create apodizer as hcipy.Apodizer() object to be able to propagate through it
        apod_prop = hcipy.Apodizer(self.apodizer)

        # Apply spatial filter
        apod_plane = apod_prop(wf_active_pupil)
        through_fpm = apod_plane.electric_field - self.coro_no_ls(apod_plane).electric_field
        wf_pre_lowfs = hcipy.Wavefront(through_fpm, self.wvln)

        lowfs = self.zwfs(wf_pre_lowfs)
        return lowfs
