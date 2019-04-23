"""
This is a module containing functions to generate the ATLAST pupil and simple coronagraphs from HCIPy.
"""
import os
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import hcipy

from config import CONFIG_INI
import util_pastis as util


# Configfile imports
which_tel = CONFIG_INI.get('telescope', 'name')
pupil_size = CONFIG_INI.getint('numerical', 'tel_size_px')


def get_atlast_aperture(outDir, normalized=False, with_segment_gaps=True, segment_transmissions=1):
    """Make the ATLAST pupil mask.

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
    pupil_grid = hcipy.make_pupil_grid(dims=pupil_size)
    pupil_diameter = CONFIG_INI.getfloat(which_tel, 'diameter')
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
    atlast = hcipy.evaluate_supersampled(func, pupil_grid, 2)

    #TODO: add segment numbering to PDF pupil image
    hcipy.imshow_field(atlast)
    plt.savefig(os.path.join(outDir, 'ATLAST_pupil.pdf'))
    util.write_fits(atlast.shaped, os.path.join(outDir, 'pupil.fits'))

    return segment_positions
