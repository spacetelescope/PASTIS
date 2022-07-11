import os
import numpy as np
from pastis.temporal_analysis.close_loop_analysis import req_closedloop_calc_batch
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from pastis.config import CONFIG_PASTIS
from pastis.simulators.luvoir_imaging import LuvoirA_APLC
import pastis.util as util


if __name__ == '__main__':
    detector_noise = 0.0
    tscale = 316227.7660168379
    flux_Starfactor = 8245167299.524352
    niter = 10
    norm = 0.010242195657579547
    TimeMinus = -2
    TimePlus = 5.5
    Ntimes = 20

    resDir = '/Users/asahoo/Desktop/data_repos/harris_data/2022-07-11T18-05-38_luvoir'
    past_dir = '/Users/asahoo/Desktop/data_repos/harris_data/2022-07-11T08-34-22_luvoir_small'
    G_coron = fits.getdata(os.path.join(resDir, 'G_coron.fits'))
    G_OBWFS = fits.getdata(os.path.join(resDir, 'G_OBWFS.fits'))
    E0_coron = fits.getdata(os.path.join(resDir, 'E0_coron.fits'))
    E0_OBWFS = fits.getdata(os.path.join(resDir, 'E0_OBWFS.fits'))
    Qharris = fits.getdata(os.path.join(past_dir, 'Qharris.fits'))

    optics_input = os.path.join(util.find_repo_location(), CONFIG_PASTIS.get('LUVOIR', 'optics_path_in_repo'))
    coronagraph_design = CONFIG_PASTIS.get('LUVOIR','coronagraph_design')
    sampling = CONFIG_PASTIS.getfloat('LUVOIR', 'sampling')
    luvoir = LuvoirA_APLC(optics_input, coronagraph_design, sampling)
    dh_mask = luvoir.dh_mask


    c_target = 10**(-11)
    contrast_floor = 4.237636070056418e-11

    result_mv = []
    for StarMag in range(2, 7, 1):
        print('Harris modes closeloop batch estimation, StarMag %f' % StarMag)
        wavescale = 1.
        niter = 10
        timer1 = time.time()
        for tscale in np.logspace(TimeMinus, TimePlus, Ntimes):
            Starfactor = 10 ** (-StarMag / 2.5)
            print(tscale)
            tmp0 = req_closedloop_calc_batch(G_coron, G_OBWFS, E0_coron, E0_OBWFS, detector_noise,
                                             detector_noise, tscale, flux_Starfactor,
                                             wavescale ** 2 * Qharris,
                                             niter, dh_mask, norm)
            tmp1 = tmp0['averaged_hist']
            n_tmp1 = len(tmp1)
            result_mv.append(tmp1[n_tmp1 - 1])

    texp = np.logspace(TimeMinus, TimePlus, Ntimes)

    plt.figure(figsize=(15, 10))
    plt.plot(texp, result_mv[0:20] - contrast_floor, label=r'$m_{v}=2$')
    plt.plot(texp, result_mv[20:40] - contrast_floor, label=r'$m_{v}=3$')
    plt.plot(texp, result_mv[40:60] - contrast_floor, label=r'$m_{v}=4$')
    plt.plot(texp, result_mv[60:80] - contrast_floor, label=r'$m_{v}=5$')
    plt.plot(texp, result_mv[80:100] - contrast_floor, label=r'$m_{v}=6$')
    plt.xlabel("$t_{WFS}$ in secs", fontsize=20)
    plt.ylabel("$\Delta$ contrast", fontsize=20)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(top=True, bottom=True, left=True,
                    right=True, labelleft=True, labelbottom=True,
                    labelsize=20)
    plt.tick_params(axis='both', which='major', length=10, width=2)
    plt.tick_params(axis='both', which='minor', length=6, width=2)
    plt.grid()
    plt.show()
    plt.savefig(os.path.join(resDir, 'cont_mv_%s.png' % c_target))