"""
This is a module containing functions to generate the ATLAST pupil and simple coronagraphs from HCIPy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import hcipy

from config import CONFIG_INI
import util_pastis as util


# Configfile imports
which_tel = CONFIG_INI.get('telescope', 'name')
pupil_size = CONFIG_INI.getint('numerical', 'tel_size_px')
PUP_DIAMETER = CONFIG_INI.getfloat(which_tel, 'diameter')


def get_atlast_aperture(normalized=False, with_segment_gaps=True, segment_transmissions=1, write_to_disk=False, outDir=None):
    """Make the ATLAST/HiCAT pupil mask.

    This function is a copy of make_hicat_aperture(), except that it also returns the segment positions.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission will
        be used for all segments.

    Returns
    -------
    Field generator
        The ATLAST aperture.
    CartesianGrid
        The segment positions.
    """
    pupil_diameter = PUP_DIAMETER
    segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 7
    num_rings = 3
    segment_gap = CONFIG_INI.getfloat(which_tel, 'gaps')

    if not with_segment_gaps:
        segment_gap = 0

    if normalized:
        segment_circum_diameter /= pupil_diameter
        segment_gap /= pupil_diameter
        pupil_diameter = 1.0

    segment_positions = hcipy.make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(hcipy.circular_aperture(segment_circum_diameter)(grid) > 0))

    hexagon = hcipy.hexagonal_aperture(segment_circum_diameter - segment_gap)

    def segment(grid):
        return hexagon(grid.rotated(np.pi/2))

    segmented_aperture = hcipy.make_segmented_aperture(segment, segment_positions, segment_transmissions)

    def func(grid):
        res = segmented_aperture(grid)

        return hcipy.Field(res, grid)

    # Save pupil to disk, as pdf and fits
    if write_to_disk:
        pupil_grid = hcipy.make_pupil_grid(dims=pupil_size, diameter=pupil_diameter)
        atlast = hcipy.evaluate_supersampled(func, pupil_grid, 8)

        hcipy.imshow_field(atlast)
        for i in range(36):
            plt.annotate(str(i + 1), size='x-large', xy=(segment_positions.x[i]-pupil_diameter*0.03, segment_positions.y[i]-pupil_diameter*0.02))
            # -0.03/-0.02 is for shifting the numbers closer to the segment centers. Scaling that by pupil_diameter
            # keeps them in place.
        plt.savefig(os.path.join(outDir, 'ATLAST_pupil.pdf'))

        util.write_fits(atlast.shaped, os.path.join(outDir, 'pupil.fits'))

    return func, segment_positions


class SegmentedMirror(hcipy.OpticalElement):
    """A segmented mirror from a segmented aperture.

    Parameters:
    ----------
    aperture : Field
        The segmented aperture of the mirror.
    seg_pos : CartesianGrid(UnstructuredCoords)
        Segment positions of the aperture.
    """

    def __init__(self, aperture, seg_pos):
        self.aperture = aperture
        self.segnum = len(seg_pos.x)
        self.segmentlist = np.arange(1, self.segnum + 1)
        self._coef = np.zeros((self.segnum, 3))
        self.seg_pos = seg_pos
        self.input_grid = aperture.grid
        self._last_npix = np.nan    # see _setup_grids for this

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
        """ The surface of the segmented mirror in meters, the full surface as a Field.
        """
        surf = self.apply_coef()
        return surf

    @property
    def coef(self):
        """ The surface shape of the deformable mirror, in meters and radians; PTT segment coefficients.
        """
        return self._coef

    def flatten(self):
        """ Flatten the DM by setting all segment coefficients to zero."""
        self._coef[:] = 0

    def set_segment(self, segid, piston, tip, tilt):
        """ Set an individual segment of the DM.

        Piston in meter of surface, tip and tilt in radians of surface.

        Parameters
        -------------
        segid : integer
            Index of the segment you wish to control, starting at 1 (center would  be 0, but doesn't exist)
        piston, tip, tilt : floats, meters and radians
            Piston (in meters) and tip and tilt (in radians)
        """
        self._coef[segid - 1] = [piston, tip, tilt]

    def _setup_grids(self):
        """ Set up the grids to compute the segmented mirror surface into.
        This is relatively slow, but we only need to do this once for
        each size of input grids.
        """
        npix = self.aperture.shaped.shape[0]
        if npix == self._last_npix:
            return
        else:
            self._last_npix = npix

        x, y = self.input_grid.coords

        self._seg_mask = np.zeros_like(x)
        self._seg_x = np.zeros_like(x)
        self._seg_y = np.zeros_like(y)
        self._seg_indices = dict()

        pupil_grid = hcipy.make_pupil_grid(dims=npix, diameter=PUP_DIAMETER)
        aper_num, seg_positions = get_atlast_aperture(normalized=False,
                                                      segment_transmissions=np.arange(1, self.segnum + 1))
        aper_num = hcipy.evaluate_supersampled(aper_num, pupil_grid, 2)

        self._seg_mask = np.copy(aper_num)

        for i in self.segmentlist:
            wseg = np.where(self._seg_mask == i)
            self._seg_indices[i] = wseg

            cenx, ceny = self.seg_pos.points[i - 1]

            self._seg_x[wseg] = x[wseg] - cenx
            self._seg_y[wseg] = y[wseg] - ceny

            # Set gaps to zero
            bad_gaps_x = np.where(np.abs(self._seg_x) > 0.1*PUP_DIAMETER)    #*PUP_DIAMETER generalizes it for any size pupil field
            self._seg_x[bad_gaps_x] = 0
            bad_gaps_y = np.where(np.abs(self._seg_y) > 0.1*PUP_DIAMETER)
            self._seg_y[bad_gaps_y] = 0

    def apply_coef(self):
        """ Apply the DM shape from its own segment coefficients to make segmented mirror surface.
        """
        self._setup_grids()

        keep_surf = np.zeros_like(self._seg_x)
        for i in self.segmentlist:
            wseg = self._seg_indices[i]
            keep_surf[wseg] = (self._coef[i - 1, 0] +
                              self._coef[i - 1, 1] * self._seg_x[wseg] +
                              self._coef[i - 1, 2] * self._seg_y[wseg])
        return hcipy.Field(keep_surf, self.input_grid)

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
