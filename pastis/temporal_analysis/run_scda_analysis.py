import os
import numpy as np
from astropy.io import fits
import pastis.util as util
import matplotlib.pyplot as plt
from pastis.config import CONFIG_PASTIS
from pastis.simulators.scda_telescopes import HexRingAPLC
from pastis.pastis_analysis import calculate_segment_constraints
from pastis.matrix_generation.matrix_from_efields import MatrixEfieldHex


def plot_thermal_mus(mus, nmodes, nsegments, c0, out_dir, save=False):
    harris_coeffs_table = np.zeros([nmodes, nsegments])
    for qq in range(nmodes):
        for kk in range(nsegments):
            harris_coeffs_table[qq, kk] = mus[qq + (kk) * nmodes]

    plt.figure(figsize=(10, 10))
    plt.title("Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=20)
    plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
    plt.xlabel("Segment Number", fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.plot(harris_coeffs_table[0]*1e3, label="Faceplates Silvered")
    plt.plot(harris_coeffs_table[1]*1e3, label="Bulk")
    plt.plot(harris_coeffs_table[2]*1e3, label="Gradiant Radial")
    plt.plot(harris_coeffs_table[3]*1e3, label="Gradiant X lateral")
    plt.plot(harris_coeffs_table[4]*1e3, label="Gradient Z axial")
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, 'mus_1d_multi_modes_%s.png' % c0))


if __name__ == '__main__':

    # Instantiate the HeXRingAPLC telescope
    ini_path = CONFIG_PASTIS.get('local', 'local_data_path')
    optics_dir = os.path.join(util.find_repo_location(), 'data', 'SCDA')
    num_rings = 5
    robust = 4
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    tel = HexRingAPLC(optics_dir, num_rings, sampling, robustness_px=robust)

    # Needed for Harris mirror
    DM = 'harris_seg_mirror'  # Possible: "seg_mirror", "harris_seg_mirror", "zernike_mirror"
    fpath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')  # path to Harris spreadsheet
    pad_orientations = np.pi / 2 * np.ones(CONFIG_PASTIS.getint('LUVOIR', 'nb_subapertures'))
    DM_SPEC = (fpath, pad_orientations, True, False, False)

    # Create harris deformable mirror
    pad_orientation = np.pi / 2 * np.ones(tel.nseg)
    filepath = CONFIG_PASTIS.get('LUVOIR', 'harris_data_path')
    tel.create_segmented_harris_mirror(filepath, pad_orientation, thermal=True, mechanical=False, other=False)

    # get number of poking modes
    num_actuators = tel.harris_sm.num_actuators
    num_modes = 5

    # First generate a PASTIS matrix
    run_matrix = MatrixEfieldHex(which_dm=DM, dm_spec=DM_SPEC, num_rings=num_rings,
                                 calc_science=True, calc_wfs=True,
                                 initial_path=ini_path,
                                 saveefields=True, saveopds=True,
                                 norm_one_photon=True)

    run_matrix.calc()
    data_dir = run_matrix.overall_dir
    print(f'Matrix and Efields saved to {data_dir}.')

    pastis_matrix = fits.getdata(os.path.join(data_dir, 'matrix_numerical', 'pastis_matrix.fits'))

    # Calculate the static tolerances
    c_target_log = -11
    c_target = 10 ** (c_target_log)
    mus_1d = calculate_segment_constraints(pastis_matrix, c_target=1e-11, coronagraph_floor=0)
    np.savetxt(os.path.join(data_dir, 'mus_%s_%d.csv' % (c_target, num_rings)), mus_1d, delimiter=',')

    plot_thermal_mus(mus_1d, num_modes, tel.nseg, c_target, data_dir, save=True)

