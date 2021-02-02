"""
Defining some aperture functions analogous to hcipy apertures.
"""

import numpy as np
import hcipy as hc


# This dictionary maps a Poppy 36-segment mirror to an HCIPy 36-segment mirror
POPPY_TO_HCIPY_INDEX = {
    1: 2, 2: 1, 3: 0, 4: 5, 5: 4, 6: 3,
    7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 17, 13: 16, 14: 15, 15: 14, 16: 13, 17: 12, 18: 11,
    19: 24, 20: 23, 21: 22, 22: 21, 23: 20, 24: 19, 25: 18,
    26: 35, 27: 34, 28: 33, 29: 32, 30: 31, 31: 30, 32: 29, 33: 28, 34: 27, 35: 26, 36: 25}


def make_luvoir_a_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, segment_transmissions=1,
                           return_segments=False):
    '''Make the LUVOIR A aperture.

    This aperture changes frequently. This one is based on [1]_. Spiders and segment gaps
    can be included or excluded, and the transmission for each of the segments can also be changed.

    .. [1] Ruane et al. 2018 "Fast linearized coronagraph optimizer (FALCO) IV: coronagraph design
    	survey for obstructed and segmented apertures." Space Telescopes and Instrumentation 2018:
    	Optical, Infrared, and Millimeter Wave. Vol. 10698. International Society for Optics and Photonics, 2018.

    Parameters
    ----------
    normalized : boolean
    	If this is True, the outer diameter will be scaled to 1. Otherwise, the
    	diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
    	Include the secondary mirror support structure in the aperture.
    with_segment_gaps : boolean
    	Include the gaps between individual segments in the aperture.
    segment_transmissions : scalar or array_like
    	The transmission for each of the segments. If this is a scalar, this transmission will
    	be used for all segments.
    return_segments : boolean
    	If this is True, also return a ModeBasis with all segments.

    Returns
    -------
    Field generator
    	The LUVOIR A aperture.
    ModeBasis
    	The segment mode basis. Only returned when `return_segments` is True.
    '''
    pupil_diameter = 15.0  # m
    segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 12
    num_rings = 6
    segment_gap = 0.001 * pupil_diameter
    spider_width = 0.005 * pupil_diameter
    spider_relative_offset = 0.9  # as estimated from the paper, not an actual value

    if not with_segment_gaps:
        segment_gap = 0

    if normalized:
        segment_circum_diameter /= pupil_diameter
        segment_gap /= pupil_diameter
        spider_width /= pupil_diameter
        pupil_diameter = 1.0

    segment_positions = hc.make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
    segment_positions = segment_positions.subset(hc.circular_aperture(pupil_diameter * 0.98))
    segment_positions = segment_positions.subset(lambda grid: ~(hc.circular_aperture(segment_circum_diameter)(grid) > 0))

    hexagon = hc.hexagonal_aperture(segment_circum_diameter - segment_gap, np.pi / 2)

    def segment(grid):
        return hexagon(grid)

    segmented_aperture = hc.make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments)

    if return_segments:
        segmented_aperture, segments = segmented_aperture

    if with_spiders:
        spider1 = hc.make_spider_infinite([0, 0], 90, spider_width)

        p1 = np.array([-segment_circum_diameter * 0.5 * spider_relative_offset, 0])
        p2 = np.array([p1[0], -np.sqrt(3) * segment_circum_diameter + (
                segment_circum_diameter * 0.5 * (1 - spider_relative_offset)) * np.sqrt(3)])
        spider2 = hc.make_spider(p1, p2, spider_width)

        p3 = p2 - np.array([pupil_diameter / 2, pupil_diameter * np.sqrt(3) / 2])
        spider3 = hc.make_spider(p2, p3, spider_width)

        p4 = p1 * np.array([-1, 1])
        p5 = p2 * np.array([-1, 1])
        p6 = p3 * np.array([-1, 1])

        spider4 = hc.make_spider(p4, p5, spider_width)
        spider5 = hc.make_spider(p5, p6, spider_width)

    if return_segments:
        for i, s in enumerate(segments):
            s = lambda grid: s(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * spider5(grid)

    def aperture(grid):
        res = segmented_aperture(grid)

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * spider5(grid)

        return hc.Field(res, grid)

    if return_segments:
        return aperture, segments
    else:
        return aperture


def make_atlast_aperture(normalized=False, with_spiders=True, with_segment_gaps=True, segment_transmissions=1,
                         return_segments=False):
    '''Make the ATLAST pupil mask.

    This function is a WIP. It should NOT be used for actual designs. Current pupil should be taken as
    representative only.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 15.0 meters.
    with_spiders : boolean
        Include the secondary mirror support structure in the aperture.
    with_segment_gaps : boolean
        Include the gaps between individual segments in the aperture.
    segment_transmissions : scalar or array_like
        The transmission for each of the segments. If this is a scalar, this transmission will
        be used for all segments.
    return_segments : boolean
        If this is True, also return a ModeBasis with all segments.

    Returns
    -------
    Field generator
        The HiCAT aperture.
    ModeBasis
        The segment mode basis. Only returned when `return_segments` is True.
    '''
    pupil_diameter = 15  # m
    segment_circum_diameter = 2 / np.sqrt(3) * pupil_diameter / 7
    num_rings = 3
    segment_gap = 0.02
    spider_width = 350e-6

    if not with_segment_gaps:
        segment_gap = 0

    if normalized:
        segment_circum_diameter /= pupil_diameter
        segment_gap /= pupil_diameter
        spider_width /= pupil_diameter
        pupil_diameter = 1.0

    segment_positions = hc.make_hexagonal_grid(segment_circum_diameter / 2 * np.sqrt(3), num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(hc.circular_aperture(segment_circum_diameter)(grid) > 0))

    hexagon = hc.hexagonal_aperture(segment_circum_diameter - segment_gap, np.pi / 2)

    def segment(grid):
        return hexagon(grid)

    segmented_aperture = hc.make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments)

    if return_segments:
        segmented_aperture, segments = segmented_aperture

    if with_spiders:
        spider1 = hc.make_spider_infinite([0, 0], 60, spider_width)
        spider2 = hc.make_spider_infinite([0, 0], 120, spider_width)
        spider3 = hc.make_spider_infinite([0, 0], 240, spider_width)
        spider4 = hc.make_spider_infinite([0, 0], 300, spider_width)

        if return_segments:
            for i, s in enumerate(segments):
                s = lambda grid: s(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)

    def aperture(grid):
        res = segmented_aperture(grid)

        if with_spiders:
            res *= spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)

        return hc.Field(res, grid)

    if return_segments:
        return aperture, segments
    else:
        return aperture
