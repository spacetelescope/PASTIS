"""
This is a module containing functions and classes for imaging propagation with HCIPy, for now LUVOIR A.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import hcipy as hc
import pandas as pd
from scipy.interpolate import griddata


# class AberratedPrimary:
#     """ A segmented primary mirror with influence functions.
#
#         Parameters:
#         ----------
#         aper : Field
#             Telescope aperture.
#         indexed_aperture : Field
#             The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
#             segment identification. Segment gaps must be strictly zero.
#         seg_pos : CartesianGrid(UnstructuredCoords)
#             Segment positions of the aperture.
#         apod : Field
#             Apodizer
#         lyots : Field
#             Lyot stop
#    ,     fpm : fpm
#             Focal plane mask
#         focal_grid :
#             Focal plane grid to put final image on
#         params : dict
#             wavelength, diameter, image size in lambda/D, FPM radius
#         """


class SegmentedTelescopeAPLC:
    """ A segmented telescope with an APLC and actuated segments.

    Parameters:
    ----------
    aper : Field
        Telescope aperture.
    indexed_aperture : Field
        The *indexed* segmented aperture of the mirror, all pixels each segment being filled with its number for
        segment identification. Segment gaps must be strictly zero.
    seg_pos : CartesianGrid(UnstructuredCoords)
        Segment positions of the aperture.
    apod : Field
        Apodizer
    lyots : Field
        Lyot stop
    fpm : fpm
        Focal plane mask
    focal_grid :
        Focal plane grid to put final image on
    params : dict
        wavelength, diameter, image size in lambda/D, FPM radius
    """

    def __init__(self, aper, indexed_aperture, seg_pos, apod, lyotst, fpm, focal_grid, params):
        self.sm = []
        self.fm = []
        self.zm = []
        self.dm = []
        self.aper = aper
        self.apodizer = apod
        self.lyotstop = lyotst
        self.fpm = fpm   #TODO: this is not actually used inside this class
        self.wvln = params['wavelength']
        self.diam = params['diameter']
        self.imlamD = params['imlamD']
        self.fpm_rad = params['fpm_rad']
        self.lamDrad = self.wvln / self.diam
        self.coro = hc.LyotCoronagraph(indexed_aperture.grid, fpm, lyotst)
        self.prop = hc.FraunhoferPropagator(indexed_aperture.grid, focal_grid)
        self.coro_no_ls = hc.LyotCoronagraph(indexed_aperture.grid, fpm)
        self.wf_aper = hc.Wavefront(aper, wavelength=self.wvln)
        self.focal_det = focal_grid
        self.seg_pos = seg_pos
        self.zernike_step = np.pi / 2
        self.zernike_spot_diam = 1.06
        self.zernike_spot_points = 128
        self.zernike_pupil_downsample = 10
        self.zernike_wfs = hc.wavefront_sensing.ZernikeWavefrontSensorOptics(self.pupil_grid, phase_step=self.zernike_step,
                                                            phase_dot_diameter=self.zernike_spot_diam, num_pix=self.zernike_spot_points,
                                                            pupil_diameter=1/self.diam, reference_wavelength=1/self.wvln)
    def prop_norm(self,tmp):
        fact = np.max(self.focal_det.x)*self.pupil_grid.dims[0]/np.max(self.pupil_grid.x)/self.focal_det.dims[0]
        tmp1 = self.prop(tmp)
        tmp2 = fact*tmp1.electric_field
        tmp3 =hc.Wavefront(tmp2,self.wvln)
        return tmp3

    def prop_OBWFS(self):

        norm = 1/np.sqrt(np.sum(self.wf_aper.intensity))

        wf_coro_pup = hc.Wavefront(norm*self.wf_aper.electric_field,self.wvln)

        if (self.sm != []):
            wf_coro_pup = self.sm(wf_coro_pup)

        if (self.fm != []):
            wf_coro_pup = self.fm(wf_coro_pup)

        if (self.zm != []):
            wf_coro_pup = self.zm(wf_coro_pup)

        if (self.dm != []):
            wf_coro_pup = self.dm(wf_coro_pup)

        res = self.zernike_wfs.forward(wf_coro_pup)
        return res

    def prop_LOWFS(self):

        norm = 1 / np.sqrt(np.sum(self.wf_aper.intensity))
        wf_coro_pup = hc.Wavefront(norm * self.wf_aper.electric_field, self.wvln)

        if (self.sm != []):
            wf_coro_pup = self.sm(wf_coro_pup)

        if (self.fm != []):
            wf_coro_pup = self.fm(wf_coro_pup)

        if (self.zm != []):
            wf_coro_pup = self.zm(wf_coro_pup)

        if (self.dm != []):
            wf_coro_pup = self.dm(wf_coro_pup)

        apod_prop = hc.Apodizer(self.apodizer)
        tmp1 = apod_prop(wf_coro_pup)
        tmp2 = tmp1.electric_field - self.coro_no_ls(tmp1).electric_field
        tmp3 = hc.Wavefront(tmp2,self.wvln)
        res = self.zernike_wfs.forward(tmp3)
        return res



    def make_LO_Modes(self, Nzernike_global):
        global_zernike_basis = hc.mode_basis.make_zernike_basis(Nzernike_global,self.diam, self.pupil_grid,
                                                                starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True)
        self.zm = hc.optics.DeformableMirror(global_zernike_basis)

    def make_DM(self,num_actuators_across):
        actuator_spacing = self.diam / num_actuators_across
        influence_functions = hc.make_xinetics_influence_functions(self.pupil_grid, num_actuators_across,
                                                                   actuator_spacing)
        self.dm = hc.DeformableMirror(influence_functions)

    def make_HI_Modes(self, NFourier):

        """Generate a fourier sine and cosine , up to NFourier cycles per aperture.

                        Parameters:
                        ----------
                        NFourier : int
                            Maximum number for cycles per apertures, use an odd number

                        --------
                        self.fdm: DeformableMirror
                            Fourier deformable mirror (primary) as a DM object
                        """

        fourier_grid = hc.make_pupil_grid(dims=NFourier, diameter=NFourier)
        fourier_basis = hc.mode_basis.make_fourier_basis(self.pupil_grid, fourier_grid, sort_by_energy=True)
        self.fm = hc.optics.DeformableMirror(fourier_basis)

    def make_segment_zernike_primary(self,Nzernike):
        """Generate a zernike basis, up to Nzernike, for each segment.

                Parameters:
                ----------
                Nzernike : int
                    Maximum order of each zernike on each segment

                --------
                self.sm: DeformableMirror
                    Segmented mirror (primary) as a DM object
                """

        segment = hc.hexagonal_aperture(self.segment_circum_diameter, np.pi / 2)
        segment_sampled = hc.evaluate_supersampled(segment,self.pupil_grid, 1)
        aper2, segs2 = hc.make_segmented_aperture(segment,self.seg_pos, segment_transmissions=1, return_segments=True)
        luvoir_segmented_pattern = hc.evaluate_supersampled(aper2, self.pupil_grid, 1)
        seg_evaluated = []
        for seg_tmp in segs2:
            tmp_evaluated = hc.evaluate_supersampled(seg_tmp, self.pupil_grid, 1)
            seg_evaluated.append(tmp_evaluated)


        seg_num = 0
        mode_basis_local_zernike = hc.mode_basis.make_zernike_basis(Nzernike, self.segment_circum_diameter,self.pupil_grid.shifted(-self.seg_pos[seg_num]),
                                                                    starting_mode=1,
                                                                    ansi=False, radial_cutoff=True, use_cache=True)
        for qq in range(0, Nzernike):
            mode_basis_local_zernike._transformation_matrix[:, qq] = seg_evaluated[seg_num]*mode_basis_local_zernike._transformation_matrix[:, qq]
        for seg_num in range(1, 120):
            # print(seg_num)
            mode_basis_local_zernike_tmp = hc.mode_basis.make_zernike_basis(Nzernike, self.segment_circum_diameter,self.pupil_grid.shifted(-self.seg_pos[seg_num]),
                                                                            starting_mode=1,
                                                                            ansi=False, radial_cutoff=True,
                                                                            use_cache=True)
            for qq in range(0, Nzernike):
                mode_basis_local_zernike_tmp._transformation_matrix[:, qq] = seg_evaluated[seg_num] * mode_basis_local_zernike_tmp._transformation_matrix[:, qq]
            mode_basis_local_zernike.extend(mode_basis_local_zernike_tmp)

        self.sm = hc.optics.DeformableMirror(mode_basis_local_zernike)


    def make_segment_Harris_thermal_primary(self,Filepath,Pad_orients):
        """Generate a basis made of the thermal modes provided by harris

                Parameters:
                ----------
                Filepath: path of the xls spreadshet with the modes.

                Pad_orients: orientation of the mounting pads of the primary

                --------
                self.sm: DeformableMirror
                    Segmented mirror (primary) as a DM object
                """

        df = pd.read_excel(Filepath)
        A = np.asarray(df.a)
        B = np.asarray(df.b)
        C = np.asarray(df.c)
        D = np.asarray(df.d)
        E = np.asarray(df.e)
        F = np.asarray(df.f)
        G = np.asarray(df.g)
        H = np.asarray(df.h)
        I = np.asarray(df.i)
        J = np.asarray(df.j)
        K = np.asarray(df.k)
        X = np.asarray(df.X)
        Y = np.asarray(df.Y)
        HarrisDiam = np.max([np.max(X) - np.min(X), np.max(Y) - np.min(Y)])
        pup_dims = self.pupil_grid.dims
        X = np.asarray(df.X) * self.segment_circum_diameter / HarrisDiam
        Y = np.asarray(df.Y) * self.segment_circum_diameter / HarrisDiam
        ti = np.linspace(-0.5, 0.5, pup_dims[0]) * self.diam
        XI, YI = np.meshgrid(ti, ti)
        points = np.transpose(np.asarray([X, Y]))
        valuesA = A
        valuesB = B
        valuesC = C
        valuesD = D
        valuesE = E
        valuesF = F
        valuesG = G
        valuesH = H
        valuesI = I
        valuesJ = J
        valuesK = K




        segment = hc.hexagonal_aperture(self.segment_circum_diameter, np.pi / 2)
        segment_sampled = hc.evaluate_supersampled(segment,self.pupil_grid, 1)
        aper2, segs2 = hc.make_segmented_aperture(segment,self.seg_pos, segment_transmissions=1, return_segments=True)
        luvoir_segmented_pattern = hc.evaluate_supersampled(aper2, self.pupil_grid, 1)
        seg_evaluated = []
        for seg_tmp in segs2:
            tmp_evaluated = hc.evaluate_supersampled(seg_tmp, self.pupil_grid, 1)
            seg_evaluated.append(tmp_evaluated)

        HarrisBase_Thermal = []
        for seg_num in range(0, 120):
            # Thermal: a, h, i, j, k
            # Mechnical: e, f, g
            # Other: b, c, d
            print(seg_num)
            grid_seg = self.pupil_grid.shifted(-self.seg_pos[seg_num])
            xL1D = grid_seg.x
            yL1D = grid_seg.y
            xL1D = np.asarray(xL1D)
            yL1D = np.asarray(yL1D)
            phi = Pad_orients[seg_num]
            XRot = xL1D * np.cos(phi) + yL1D * np.sin(phi)
            YRot = -xL1D * np.sin(phi) + yL1D * np.cos(phi)
            ZA = griddata(points, valuesA, (XRot, YRot), method='linear')
            ZA[np.isnan(ZA)] = 0
            ZA = ZA.ravel() * seg_evaluated[seg_num]
            ZH = griddata(points, valuesH, (XRot, YRot), method='linear')
            ZH[np.isnan(ZH)] = 0
            ZH = ZH.ravel() * seg_evaluated[seg_num]
            ZI = griddata(points, valuesI, (XRot, YRot), method='linear')
            ZI[np.isnan(ZI)] = 0
            ZI = ZI.ravel() * seg_evaluated[seg_num]
            ZJ = griddata(points, valuesJ, (XRot, YRot), method='linear')
            ZJ[np.isnan(ZJ)] = 0
            ZJ = ZJ.ravel() * seg_evaluated[seg_num]
            ZK = griddata(points, valuesK, (XRot, YRot), method='linear')
            ZK[np.isnan(ZK)] = 0
            ZK = ZK.ravel() * seg_evaluated[seg_num]
            HarrisBase_Thermal.append([ZA, ZH, ZI, ZJ, ZK])
        HarrisBase_Thermal = np.asarray(HarrisBase_Thermal)
        N_segs = HarrisBase_Thermal.shape[0]
        N_single_modes = HarrisBase_Thermal.shape[1]
        HarrisBase_Thermal = HarrisBase_Thermal.reshape(N_segs * N_single_modes, pup_dims[0] ** 2)
        Harris_Thermal_ModeBasis = hc.ModeBasis(np.transpose(HarrisBase_Thermal), grid=self.pupil_grid)

        self.sm = hc.optics.DeformableMirror(Harris_Thermal_ModeBasis)

    def calc_psf(self, ref=False, display_intermediate=False,  return_intermediate=None):
        """Calculate the PSF of the segmented telescope, normalized to contrast units.

        Parameters:
        ----------
        ref : bool
            Keyword for additionally returning the refrence PSF without the FPM.
        display_intermediate : bool
            Keyword for display of all planes.
        return_intermediate : string
            Either 'intensity', return the intensity in all planes; except phase on the SM (first plane)
            or 'efield', return the E-fields in all planes. Default none.
        Returns:
        --------
        wf_im_coro.intensity : Field
            Coronagraphic image, normalized to contrast units by max of reference image (even when ref
            not returned).
        wf_im_ref.intensity : Field, optional
            Reference image without FPM.
        intermediates : dict of Fields, optional
            Intermediate plane intensity images; except for full wavefront on segmented mirror.
        wf_im_coro : Wavefront
            Wavefront in last focal plane.
        wf_im_ref : Wavefront, optional
            Wavefront of reference image without FPM.
        intermediates : dict of Wavefronts, optional
            Intermediate plane E-fields; except intensity in focal plane after FPM.
        """

        # Create fake FPM for plotting
        fpm_plot = 1 - hc.circular_aperture(2 * self.fpm_rad * self.lamDrad)(self.focal_grid_fpm)
        # This was hackd and does not work.

        # Create apodozer as hc.Apodizer() object to be able to propagate through it
        apod_prop = hc.Apodizer(self.apodizer)

        # Calculate all wavefronts of the full propagation

        norm = 1 / np.sqrt(np.sum(self.wf_aper.intensity))
        wf_coro_pup = hc.Wavefront(norm * self.wf_aper.electric_field, self.wvln)

        if (self.sm != []):
            wf_coro_pup = self.sm(wf_coro_pup)

        if (self.fm != []):
            wf_coro_pup = self.fm(wf_coro_pup)

        if (self.zm != []):
            wf_coro_pup = self.zm(wf_coro_pup)

        if (self.dm != []):
            wf_coro_pup = self.dm(wf_coro_pup)

        wf_apod = apod_prop(wf_coro_pup)
        wf_lyot = self.coro(wf_apod)
        wf_im_coro = self.prop_norm(wf_lyot)

        # Wavefronts in extra planes
        wf_before_fpm = self.prop(wf_apod)
        int_after_fpm = np.log10(wf_before_fpm.intensity / wf_before_fpm.intensity.max())   # this is the intensity straight I fixed this
        wf_before_lyot = self.coro_no_ls(wf_apod)

        # Wavefronts of the reference propagation
        wf_ref_pup = hc.Wavefront(norm * self.aper * self.apodizer * self.lyotstop, wavelength=self.wvln)
        wf_im_ref = self.prop_norm(wf_ref_pup)

        # Display intermediate planes
        if display_intermediate:

            plt.figure(figsize=(15, 15))

            plt.subplot(331)
            hc.imshow_field(wf_sm.phase, mask=self.aper, cmap='RdBu')
            plt.title('Seg aperture phase')

            plt.subplot(332)
            hc.imshow_field(wf_apod.intensity, cmap='inferno')
            plt.title('Apodizer')

            plt.subplot(333)
            hc.imshow_field(wf_before_fpm.intensity / wf_before_fpm.intensity.max(), norm=LogNorm(), cmap='inferno')
            plt.title('Before FPM')

            plt.subplot(334)
            hc.imshow_field(int_after_fpm / wf_before_fpm.intensity.max(), cmap='inferno')
            plt.title('After FPM')

            plt.subplot(335)
            hc.imshow_field(wf_before_lyot.intensity / wf_before_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno')
            plt.title('Before Lyot stop')

            plt.subplot(336)
            hc.imshow_field(wf_lyot.intensity / wf_lyot.intensity.max(), norm=LogNorm(vmin=1e-3, vmax=1),
                            cmap='inferno', mask=self.lyotstop)
            plt.title('After Lyot stop')

            plt.subplot(337)
            hc.imshow_field(wf_im_coro.intensity / wf_im_ref.intensity.max(), norm=LogNorm(vmin=1e-10, vmax=1e-3),
                            cmap='inferno')
            plt.title('Final image')
            plt.colorbar()

        if return_intermediate == 'intensity':

            # Return the intensity in all planes; except phase on the SM (first plane)
            intermediates = {'seg_mirror': wf_coro_pup.phase,
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
            intermediates = {'seg_mirror': wf_coro_pup,
                             'apod': wf_apod,
                             'before_fpm': wf_before_fpm,
                             'after_fpm': int_after_fpm,
                             'before_lyot': wf_before_lyot,
                             'after_lyot': wf_lyot,
                             'at_science_focus': wf_im_coro}

            if ref:
                return wf_im_coro, wf_im_ref, intermediates
            else:
                return wf_im_coro, intermediates

        if ref:
            return wf_im_coro.intensity, wf_im_ref.intensity

        return wf_im_coro.intensity

    def flatten(self):
        self.sm.flatten()

    def set_segment(self, segid, piston, tip, tilt):
        self.sm.set_segment(segid, piston, tip, tilt)

    def apply_aberrations(self, aber_array):
        for vals in aber_array:
            self.sm.set_segment(vals[0], vals[1], vals[2], vals[3])

    def forward(self, wavefront):
        raise NotImplementedError()

    def backward(self, wavefront):
        raise NotImplementedError()


class LuvoirAPLC(SegmentedTelescopeAPLC):
    """ Simple E2E simulator for LUVOIR A (with APLC).

    Parameters:
    ----------
    input dir : string
        Path to input files: apodizer, aperture, indexed aperture, Lyot stop.
    apod_design : string
        Choice of apodizer design from May 2019 delivery. "small", "medium" or "large".
    """
    def __init__(self, input_dir, apod_design, samp):
        self.nseg = 120
        self.wvln = 638e-9  # m
        self.diam = 15.  # m
        self.segment_circum_diameter = 2 / np.sqrt(3) * 1.2225
        self.sampling = samp
        self.lam_over_d = self.wvln / self.diam
        self.apod_dict = {'small': {'pxsize': 1000, 'fpm_rad': 3.5, 'fpm_px': 150, 'iwa': 3.4, 'owa': 12.,
                                    'fname': '0_LUVOIR_N1000_FPM350M0150_IWA0340_OWA01200_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'medium': {'pxsize': 1000, 'fpm_rad': 6.82, 'fpm_px': 250, 'iwa': 6.72, 'owa': 23.72,
                                     'fname': '0_LUVOIR_N1000_FPM682M0250_IWA0672_OWA02372_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'},
                          'large': {'pxsize': 1000, 'fpm_rad': 13.38, 'fpm_px': 400, 'iwa': 13.28, 'owa': 46.88,
                                    'fname': '0_LUVOIR_N1000_FPM1338M0400_IWA1328_OWA04688_C10_BW10_Nlam5_LS_IDD0120_OD0982_no_ls_struts.fits'}}
        self.imlamD = 1.2*self.apod_dict[apod_design]['owa']

        # Pupil plane optics
        aper_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000.fits'
        aper_ind_path = 'inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000_indexed.fits'
        apod_path = os.path.join(input_dir, 'luvoir_stdt_baseline_bw10', apod_design + '_fpm', 'solutions',
                                 self.apod_dict[apod_design]['fname'])
        ls_fname = 'inputs/LS_LUVOIR_ID0120_OD0982_no_struts_gy_ovsamp4_N1000.fits'

        pup_read = hc.read_fits(os.path.join(input_dir, aper_path))
        aper_ind_read = hc.read_fits(os.path.join(input_dir, aper_ind_path))
        apod_read = hc.read_fits(os.path.join(input_dir, apod_path))
        ls_read = hc.read_fits(os.path.join(input_dir, ls_fname))

        self.pupil_grid = hc.make_pupil_grid(dims=self.apod_dict[apod_design]['pxsize'], diameter=self.diam)

        self.aperture = hc.Field(pup_read.ravel(), self.pupil_grid)
        self.aper_ind = hc.Field(aper_ind_read.ravel(), self.pupil_grid)
        self.apod = hc.Field(apod_read.ravel(), self.pupil_grid)
        self.ls = hc.Field(ls_read.ravel(), self.pupil_grid)

        # Load segment positions from fits header
        hdr = fits.getheader(os.path.join(input_dir, aper_ind_path))

        poslist = []
        for i in range(self.nseg):
            segname = 'SEG' + str(i + 1)
            xin = hdr[segname + '_X']
            yin = hdr[segname + '_Y']
            poslist.append((xin, yin))

        poslist = np.transpose(np.array(poslist))
        self.seg_pos = hc.CartesianGrid(hc.UnstructuredCoords(poslist))
        self.seg_pos = self.seg_pos.scaled(self.diam)

        # Focal plane mask
        samp_foc = self.apod_dict[apod_design]['fpm_px'] / (self.apod_dict[apod_design]['fpm_rad'] * 2)
        #focal_grid_fpm = hc.make_focal_grid(pupil_grid=pupil_grid, q=samp_foc,num_airy=self.apod_dict[apod_design]['fpm_rad'], wavelength=self.wvln)
        self.focal_grid_fpm = hc.make_focal_grid(
            samp_foc,
            self.apod_dict[apod_design]['fpm_rad'],
            pupil_diameter=self.diam,
            focal_length=1,
            reference_wavelength=self.wvln,
        )

        self.fpm = 1 - hc.circular_aperture(2*self.apod_dict[apod_design]['fpm_rad']*self.wvln/self.diam)(self.focal_grid_fpm)

        # Final focal plane grid (detector)
        #self.focal_det = hc.make_focal_grid(pupil_grid=pupil_grid, q=self.sampling, num_airy=self.imlamD, wavelength=self.wvln)
        self.focal_det = hc.make_focal_grid(
            self.sampling,
            self.imlamD,
            pupil_diameter=self.diam,
            focal_length=1,
            reference_wavelength=self.wvln,
        )


        luvoir_params = {'wavelength': self.wvln, 'diameter': self.diam, 'imlamD': self.imlamD,
                         'fpm_rad': self.apod_dict[apod_design]['fpm_rad']}

        # Initialize the general segmented telescope with APLC class, includes the SM
        super().__init__(aper=self.aperture, indexed_aperture=self.aper_ind, seg_pos=self.seg_pos, apod=self.apod,
                         lyotst=self.ls, fpm=self.fpm, focal_grid=self.focal_det, params=luvoir_params)

        # Propagators
        self.coro = hc.LyotCoronagraph(self.pupil_grid, self.fpm, self.ls)
        self.prop = hc.FraunhoferPropagator(self.pupil_grid, self.focal_det)
        self.coro_no_ls = hc.LyotCoronagraph(self.pupil_grid, self.fpm)
        #TODO: these three propagators should actually happen in the super init
        # -> how are self.aper_ind and pupil_grid connected?