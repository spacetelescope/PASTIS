import os
from astropy.io import fits
import numpy as np

from pastis.pastis_analysis import calculate_segment_constraints
import pastis.plotting as ppl
import pastis.util as util


class MultiModeAnalysis:
    """Analysis for multi-mode tolerancing on one of the internal simulators.

    Attributes
    ----------
    c_target : float
    simulator : simulator instance
    which_dm : string
    data_dir : string
    num_modes : int
    nseg : int
    matrix_pastis : ndarray
    dh_mask : ndarray
    contrast_floor : float
    mus : ndarray
    """

    def __init__(self, c_target, simulator, which_dm, num_modes, data_dir):
        """
        Parameters
        ----------
        c_target : float
            target contrast to tolerance to
        simulator : simulator instance
            simulator instance of one of the internal simulators
        which_dm : string
            which DM the tolerancing is done for, "harris_seg_mirror", "seg_mirror"
        num_modes : int
            number of modes (on a single segment)
        data_dir : string
            overall data directory that contains further subdirectories (e.g., "matrix_numerical")
        """

        self.c_target = c_target
        self.simulator = simulator
        self.which_dm = which_dm
        self.data_dir = data_dir
        self.num_modes = num_modes
        self.nseg = self.simulator.nseg

        os.makedirs(os.path.join(self.data_dir, 'results', 'mu_maps'), exist_ok=True)

    def run(self):
        """Main class method to call to perform the analysis."""
        self.read_matrix_results()
        self.calc_segment_reqs()

    def read_matrix_results(self):
        """Save results from matrix calculation to class attributes."""
        self.matrix_pastis = fits.getdata(os.path.join(self.data_dir, 'matrix_numerical', 'pastis_matrix.fits'))
        unaber_psf = fits.getdata(os.path.join(self.data_dir, 'unaberrated_coro_psf.fits'))  # already normalized to max of direct pdf
        self.dh_mask = self.simulator.dh_mask.shaped
        self.contrast_floor = util.dh_mean(unaber_psf, self.dh_mask)

    def calc_segment_reqs(self):
        self.mus = calculate_segment_constraints(self.matrix_pastis, c_target=self.c_target,
                                                 coronagraph_floor=self.contrast_floor)
        np.savetxt(os.path.join(self.data_dir, 'results', f'segment_requirements_{self.c_target:.2e}.csv'), self.mus, delimiter=',')

        mus_per_segment = util.sort_1d_mus_per_segment(self.mus, self.num_modes, self.nseg)
        mu_list = []
        label_list = []
        for i in range(mus_per_segment.shape[0]):
            mu_list.append(mus_per_segment[i])
            if self.which_dm == 'seg_mirror':
                label_list.append(f'Zernike mode {i}')
        if self.which_dm == 'harris_seg_mirror':
            label_list = ['Faceplates Silvered', 'Bulk', 'Gradient Radial', 'Gradient X lateral', 'Gradient Z axial']

        ppl.plot_segment_weights(mu_list, os.path.join(self.data_dir, 'results'), self.c_target, labels=label_list,
                                 save=True)
        ppl.plot_multimode_surface_maps(self.simulator, self.mus, self.num_modes, mirror=self.which_dm, cmin=-5, cmax=5,
                                        data_dir=os.path.join(self.data_dir, 'results'), fname='stat_mu_maps')
