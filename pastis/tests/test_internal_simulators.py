import os
import hcipy
import numpy as np

from pastis.e2e_simulators.scda_telescopes import HexRingAPLC
from pastis.util import find_repo_location


SAMPLING = 4
OPTICS_DIR = os.path.join(find_repo_location(), 'data', 'SCDA')


def test_num_rings_variation():
    for num_rings in [1, 2, 3, 4, 5]:
        tel = HexRingAPLC(OPTICS_DIR, num_rings, SAMPLING)
        coro, ref, inter = tel.calc_psf(ref=True, return_intermediate='efield')


def test_return_type():
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)

    coro = tel.calc_psf()
    assert type(coro) == hcipy.field.field.Field

    coro, ref = tel.calc_psf(ref=True)
    assert type(coro) == hcipy.field.field.Field
    assert type(ref) == hcipy.field.field.Field

    coro, ref, inter = tel.calc_psf(ref=True, return_intermediate='intensity')
    assert type(coro) == hcipy.field.field.Field
    assert type(ref) == hcipy.field.field.Field
    assert type(inter) == dict

    coro, ref, inter = tel.calc_psf(ref=True, return_intermediate='efield')
    assert type(coro) == hcipy.optics.wavefront.Wavefront
    assert type(ref) == hcipy.optics.wavefront.Wavefront
    assert type(inter) == dict

    coro, inter = tel.calc_psf(return_intermediate='intensity')
    assert type(coro) == hcipy.field.field.Field
    assert type(inter) == dict

    coro, inter = tel.calc_psf(return_intermediate='efield')
    assert type(coro) == hcipy.optics.wavefront.Wavefront
    assert type(inter) == dict


def test_simple_segmented_dm():
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)
    coro_perfect, ref = tel.calc_psf(ref=True)

    tel.set_segment(2, 1e-2, 0, 0)
    tel.set_segment(4, 0, 1e-2, 0)
    tel.set_segment(7, 0, 0, 1e-2)
    coro_aberrated = tel.calc_psf()
    assert np.sum(coro_perfect/ref.max()) < np.sum(coro_aberrated/ref.max())


def test_multi_mode_segmented_dm():
    n_zernikes = 6
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)
    coro_perfect, ref = tel.calc_psf(ref=True)

    tel.create_segmented_mirror(n_zernikes)
    tel.set_sm_segment(segid=2, zernike_number=2, amplitude=1e-2)
    tel.set_sm_segment(segid=4, zernike_number=4, amplitude=1e-2)
    tel.set_sm_segment(segid=7, zernike_number=5, amplitude=1e-2)
    coro_aberrated = tel.calc_psf()
    assert np.sum(coro_perfect/ref.max()) < np.sum(coro_aberrated/ref.max())


def test_global_zernike_dm():
    n_zernikes = 10
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)
    coro_perfect, ref = tel.calc_psf(ref=True)

    tel.create_global_zernike_mirror(n_zernikes)
    num_act = tel.zernike_mirror.num_actuators
    dm_command = np.zeros(num_act)

    dm_command[2] = 1e-4
    dm_command[7] = 1e-4
    tel.zernike_mirror.actuators = dm_command
    coro_aberrated = tel.calc_psf()
    assert np.sum(coro_perfect/ref.max()) < np.sum(coro_aberrated/ref.max())


def test_ripple_mirror():
    n_modes = 5
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)
    coro_perfect, ref = tel.calc_psf(ref=True)

    tel.create_ripple_mirror(n_modes)
    num_act = tel.ripple_mirror.num_actuators
    dm_command = np.zeros(num_act)

    dm_command[2] = 1e-4
    dm_command[7] = 1e-4
    tel.ripple_mirror.actuators = dm_command
    coro_aberrated = tel.calc_psf()
    assert np.sum(coro_perfect/ref.max()) < np.sum(coro_aberrated/ref.max())


def test_continuous_dm():
    n_act_across = 5
    tel = HexRingAPLC(OPTICS_DIR, num_rings=1, sampling=SAMPLING)
    coro_perfect, ref = tel.calc_psf(ref=True)

    tel.create_continuous_deformable_mirror(n_act_across)
    num_act = tel.dm.num_actuators
    dm_command = np.zeros(num_act)

    dm_command[3] = 1e-4
    dm_command[11] = 1e-4
    dm_command[19] = 1e-4
    dm_command[23] = 1e-4
    tel.dm.actuators = dm_command
    coro_aberrated = tel.calc_psf()
    assert np.sum(coro_perfect/ref.max()) < np.sum(coro_aberrated/ref.max())
