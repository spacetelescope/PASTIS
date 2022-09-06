import os
import numpy as np
import matplotlib.pyplot as plt


def plot_mus_all_hexrings(mu1, mu2, mu3, mu4, mu5, c0, out_dir, save=False):
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

    if mode == "Faceplates Silvered":
        plt.figure(figsize=(10, 10))
        plt.title("Faceplates Silvered Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
        plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
        plt.xlabel("Segment Number", fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.plot(mus1_table[0] * 1e3, label="1-HexRingTelescope")
        plt.plot(mus2_table[0] * 1e3, label="2-HexRingTelescope")
        plt.plot(mus3_table[0] * 1e3, label="3-HexRingTelescope")
        plt.plot(mus4_table[0] * 1e3, label="4-HexRingTelescope")
        plt.plot(mus5_table[0] * 1e3, label="5-HexRingTelescope")
        plt.yticks(np.arange(np.min(mus5_table[0]*1e3), np.max(mus5_table[0]*1e3), 0.1))
        if inner_segments:
            plt.xlabel("Inner Segment Number", fontsize=20)
            plt.ylim(0, 2)
            plt.xlim(0, 15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_dir, 'mus_faceplate_%s.png' % c0))

    if mode == "Bulk":
        plt.figure(figsize=(10, 10))
        plt.title("Bulk Modal constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
        plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
        plt.xlabel("Segment Number", fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.plot(mus1_table[1] * 1e3, label="1-HexRingTelescope")
        plt.plot(mus2_table[1] * 1e3, label="2-HexRingTelescope")
        plt.plot(mus3_table[1] * 1e3, label="3-HexRingTelescope")
        plt.plot(mus4_table[1] * 1e3, label="4-HexRingTelescope")
        plt.plot(mus5_table[1] * 1e3, label="5-HexRingTelescope")
        plt.yticks(np.arange(np.min(mus2_table[1] * 1e3), np.max(mus5_table[1] * 1e3), 2))
        if inner_segments:
            plt.xlabel("Inner Segment Number", fontsize=20)
            plt.yticks(np.arange(np.min(mus2_table[1] * 1e3), np.max(mus5_table[1] * 1e3), 0.5))
            plt.ylim(1, 10)
            plt.xlim(0, 15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_dir, 'mus_bulk_%s.png' % c0))

    if mode == "Gradiant Radial":
        plt.figure(figsize=(10, 10))
        plt.title("Gradiant Radial constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
        plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
        plt.xlabel("Segment Number", fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.plot(mus1_table[2] * 1e3, label="1-HexRingTelescope")
        plt.plot(mus2_table[2] * 1e3, label="2-HexRingTelescope")
        plt.plot(mus3_table[2] * 1e3, label="3-HexRingTelescope")
        plt.plot(mus4_table[2] * 1e3, label="4-HexRingTelescope")
        plt.plot(mus5_table[2] * 1e3, label="5-HexRingTelescope")
        plt.yticks(np.arange(np.min(mus2_table[2] * 1e3), np.max(mus5_table[2] * 1e3), 5))
        if inner_segments:
            plt.xlabel("Inner Segment Number", fontsize=20)
            plt.yticks(np.arange(np.min(mus2_table[1] * 1e3), np.max(mus5_table[1] * 1e3), 0.5))
            plt.ylim(1, 13.5)
            plt.xlim(0, 15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_dir, 'mus_gradient_radial_%s.png' % c0))

    if mode == "Gradiant X lateral":
        plt.figure(figsize=(10, 10))
        plt.title("Gradiant X lateral constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
        plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
        plt.xlabel("Segment Number", fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.plot(mus1_table[3] * 1e3, label="1-HexRingTelescope")
        plt.plot(mus2_table[3] * 1e3, label="2-HexRingTelescope")
        plt.plot(mus3_table[3] * 1e3, label="3-HexRingTelescope")
        plt.plot(mus4_table[3] * 1e3, label="4-HexRingTelescope")
        plt.plot(mus5_table[3] * 1e3, label="5-HexRingTelescope")
        plt.yticks(np.arange(np.min(mus3_table[3] * 1e3), np.max(mus5_table[3] * 1e3), 0.5))
        if inner_segments:
            plt.xlabel("Inner Segment Number", fontsize=20)
            plt.yticks(np.arange(np.min(mus3_table[3] * 1e3), np.max(mus5_table[3] * 1e3), 0.1))
            plt.ylim(0.8, 2.5)
            plt.xlim(0, 15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_dir, 'mus_gradient_xlateral_%s.png' % c0))

    if mode == "Gradiant Y lateral":
        plt.figure(figsize=(10, 10))
        plt.title("Gradiant Y lateral constraints to achieve a dark hole contrast of "r"$10^{%d}$" % np.log10(c0), fontsize=10)
        plt.ylabel("Weight per segment (in units of pm)", fontsize=15)
        plt.xlabel("Segment Number", fontsize=20)
        plt.tick_params(top=True, bottom=True, left=True, right=True, labelleft=True, labelbottom=True, labelsize=20)
        plt.plot(mus1_table[4] * 1e3, label="1-HexRingTelescope")
        plt.plot(mus2_table[4] * 1e3, label="2-HexRingTelescope")
        plt.plot(mus3_table[4] * 1e3, label="3-HexRingTelescope")
        plt.plot(mus4_table[4] * 1e3, label="4-HexRingTelescope")
        plt.plot(mus5_table[4] * 1e3, label="5-HexRingTelescope")
        plt.yticks(np.arange(np.min(mus3_table[4] * 1e3), np.max(mus5_table[4] * 1e3), 1))
        if inner_segments:
            plt.xlabel("Inner Segment Number", fontsize=20)
            plt.yticks(np.arange(np.min(mus3_table[4] * 1e3), np.max(mus5_table[4] * 1e3), 0.1))
            plt.ylim(0.7, 3.0)
            plt.xlim(0, 15)
        plt.grid()
        plt.legend(fontsize=15)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_dir, 'mus_gradient_inner_%s.png' % c0))


if __name__ == '__main__':

    mus5 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots/mus_1e-11_5.csv', delimiter=',')
    mus4 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots/mus_1e-11_4.csv', delimiter=',')
    mus3 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots/mus_1e-11_3.csv', delimiter=',')
    mus2 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots/mus_1e-11_2.csv', delimiter=',')
    mus1 = np.genfromtxt('/Users/asahoo/Desktop/data_repos/plots/mus_1e-11_1.csv', delimiter=',')

    resdir = '/Users/asahoo/Desktop/data_repos/plots'
    plot_mus_all_hexrings(mus1, mus2, mus3, mus4, mus5, 1e-11, resdir, save=False)
    plot_single_thermal_mode_all_hex(mus1, mus2, mus3, mus4, mus5, 1e-11,
                                     mode="Gradiant Y lateral", out_dir=resdir, save=False, inner_segments=False)