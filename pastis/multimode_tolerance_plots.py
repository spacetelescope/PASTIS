import os
import numpy as np
import matplotlib.pyplot as plt


def plot_mus_all_hexrings(mu1, mu2, mu3, mu4, mu5, c0, out_dir, save=False):
    """
    Parameters
    ----------
    mu1 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 1-HexRingTelescope.
    mu2 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 2-HexRingTelescope.
    mu3 : numpy.ndarray
        Each element  represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 3-HexRingTelescope.
    mu4 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 4-HexRingTelescope.
    mu5 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 5-HexRingTelescope.
    c0 : float
        The set-target contrast for which the above tolerances were calculated.
    out_dir : str
        path where the plot will be saved
    save : bool
        whether to save the plot
    """
    plt.figure(figsize=(10, 10))
    plt.title("Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=20)
    plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
    plt.xlabel("Mode Index", fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.plot(mu1*1e3, label="1-HexRingTelescope")
    plt.plot(mu2*1e3, label="2-HexRingTelescope")
    plt.plot(mu3*1e3, label="3-HexRingTelescope")
    plt.plot(mu4*1e3, label="4-HexRingTelescope")
    plt.plot(mu5*1e3, label="5-HexRingTelescope")
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, 'mus_1d_multi_modes_%s.png' % c0))


def plot_single_thermal_mode_all_hex(mu1, mu2, mu3, mu4, mu5, c0, mode, out_dir, save=False, inner_segments=False):
    """
    Parameters
    ----------
    mu1 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 1-HexRingTelescope.
    mu2 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 2-HexRingTelescope.
    mu3 : numpy.ndarray
        Each element  represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 3-HexRingTelescope.
    mu4 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 4-HexRingTelescope.
    mu5 : numpy.ndarray
        Each element represents tolerance (in units of nm) for segment per a segment level aberration mode
        for the 5-HexRingTelescope.
    c0 : float
        The set-target contrast for which the above tolerances were calculated.
    mode : str
        name of the segment level zernike/harris thermal aberration
        "Faceplates Silvered" or "Piston", "Bulk" or "Tip", "Gradiant Radial" or "Tilt",
        "Gradiant X lateral" or "Defocus", or "Gradiant Z axial" or "Astig"
    out_dir : str
        path where the plot will be saved
    save : bool
        whether to save the plot
    inner_segments : bool
        whether to plot tolerances for the inner segments only
    """

    # for 1-HexRingTelescope
    mus1_table = np.zeros([5, 7])
    for qq in range(5):
        for kk in range(7):
            mus1_table[qq, kk] = mu1[qq + kk * 5]

    # for 2-HexRingTelescope
    mus2_table = np.zeros([5, 19])
    for qq in range(5):
        for kk in range(19):
            mus2_table[qq, kk] = mu2[qq + kk * 5]

    # for 3-HexRingTelescope
    mus3_table = np.zeros([5, 31])
    for qq in range(5):
        for kk in range(31):
            mus3_table[qq, kk] = mu3[qq + kk * 5]

    # for 4-HexRingTelescope
    mus4_table = np.zeros([5, 55])
    for qq in range(5):
        for kk in range(55):
            mus4_table[qq, kk] = mu4[qq + kk * 5]

    # for 5-HexRingTelescope
    mus5_table = np.zeros([5, 85])
    for qq in range(5):
        for kk in range(85):
            mus5_table[qq, kk] = mu5[qq + kk * 5]

    if mode == "Faceplates Silvered" or mode == "Piston":
        num = 0
    elif mode == "Bulk" or mode == "Tip":
        num = 1
    elif mode == "Gradiant Radial" or mode == "Tilt":
        num = 2
    elif mode == "Gradiant X lateral" or mode == "Defocus":
        num = 3
    elif mode == "Gradiant Z axial" or mode == "Astig":
        num = 4

    plt.figure(figsize=(10, 10))
    plt.title(str(mode)+" modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
    plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
    plt.xlabel("Segment Number", fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.plot(mus1_table[num] * 1e3, label="1-HexRingTelescope", marker="o")
    plt.plot(mus2_table[num] * 1e3, label="2-HexRingTelescope", marker="s")
    plt.plot(mus3_table[num] * 1e3, label="3-HexRingTelescope", marker="p")
    plt.plot(mus4_table[num] * 1e3, label="4-HexRingTelescope", marker="P")
    plt.plot(mus5_table[num] * 1e3, label="5-HexRingTelescope", marker="H")
    if inner_segments:
        plt.xlabel("Inner Segment Number", fontsize=20)
        plt.yticks(np.arange(np.min(mus3_table[num] * 1e3), np.max(mus4_table[num] * 1e3), 0.5))
        plt.ylim(1, 5)
        plt.xlim(0, 15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(out_dir, str(mode) + '_mus_%s.png' % c0))


if __name__ == '__main__':

    # Thermal tolerance coefficients
    mus5 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_thermal/mus_1e-11_5.csv', delimiter=',')
    mus4 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_thermal/mus_1e-11_4.csv', delimiter=',')
    mus3 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_thermal/mus_1e-11_3.csv', delimiter=',')
    mus2 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_thermal/mus_1e-11_2.csv', delimiter=',')
    mus1 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_thermal/mus_1e-11_1.csv', delimiter=',')

    # Segment level zernike coefficients
    z5 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_mid_zernike/mus_1e-11_5.csv', delimiter=',')
    z4 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_mid_zernike/mus_1e-11_4.csv', delimiter=',')
    z3 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_mid_zernike/mus_1e-11_3.csv', delimiter=',')
    z2 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_mid_zernike/mus_1e-11_2.csv', delimiter=',')
    z1 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots_mid_zernike/mus_1e-11_1.csv', delimiter=',')

    resdir = '/Users/asahoo/Desktop/data_repos/plots_mid_zernike'
    # plot_mus_all_hexrings(mus1, mus2, mus3, mus4, mus5, 1e-11, resdir, save=False)
    plot_single_thermal_mode_all_hex(z1, z2, z3, z4, z5, 1e-11,
                                     mode="Astig", out_dir=resdir, save=True, inner_segments=True)