import numpy as np


def matrix_subsample(matrix, n, m):
    """
    Reduces the order of a matrix by taking mean over a block in the matrix.

    Parameters
    ----------
    matrix : numpy 2d array
        the input matrix to be reduced
    n : int
        desired number of rows of the reduced matrix
    m : int
        desired number of columns of the reduced matrix

    Returns
    -------
    matrix_reduced : numpy 2d array
    """

    length = matrix.shape[0] // n  # block length
    breadth = matrix.shape[1] // m  # block breadth

    new_shape = (n, length, m, breadth)
    reshaped_array = matrix.reshape(new_shape)
    matrix_reduced = np.sum(reshaped_array, axis=(1, 3))

    return matrix_reduced


def calculate_sensitivity_matrices(e0_coron, e0_obwfs, efield_coron_real,
                                   efield_coron_imag, efield_obwfs_real,
                                   efield_obwfs_imag, subsample_factor):

    total_sci_pix = np.square(e0_coron.shape[1])
    total_pupil_pix = np.square(e0_obwfs.shape[1])

    ref_coron_real = np.reshape(e0_coron[0], total_sci_pix)
    ref_coron_imag = np.reshape(e0_coron[1], total_sci_pix)

    ref_obwfs_real = np.reshape(e0_obwfs[0], total_pupil_pix)
    ref_obwfs_imag = np.reshape(e0_obwfs[1], total_pupil_pix)

    ref_coron = np.zeros([total_sci_pix, 1, 2])
    ref_coron[:, 0, 0] = ref_coron_real
    ref_coron[:, 0, 1] = ref_coron_imag

    n_sub_pix = int(np.sqrt(total_pupil_pix) // subsample_factor)
    ref_wfs_real_sub = np.reshape(matrix_subsample(e0_obwfs[0], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_imag_sub = np.reshape(matrix_subsample(e0_obwfs[1], n_sub_pix, n_sub_pix), int(np.square(n_sub_pix)))
    ref_wfs_sub = (ref_wfs_real_sub + 1j * ref_wfs_imag_sub) / subsample_factor

    ref_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 1, 2])
    ref_obwfs_downsampled[:, 0, 0] = ref_wfs_sub.real
    ref_obwfs_downsampled[:, 0, 1] = ref_wfs_sub.imag

    num_all_modes = efield_coron_real.shape[0]
    g_coron = np.zeros([total_sci_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_coron[:, 0, i] = np.reshape(efield_coron_real[i], total_sci_pix) - ref_coron_real
        g_coron[:, 1, i] = np.reshape(efield_coron_imag[i], total_sci_pix) - ref_coron_imag

    g_obwfs = np.zeros([total_pupil_pix, 2, num_all_modes])
    for i in range(num_all_modes):
        g_obwfs[:, 0, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_real
        g_obwfs[:, 1, i] = np.reshape(efield_obwfs_real[i], total_pupil_pix) - ref_obwfs_imag

    g_obwfs_downsampled = np.zeros([int(np.square(n_sub_pix)), 2, num_all_modes])
    for i in range(num_all_modes):
        efields_per_mode_wfs_real_sub = np.reshape(matrix_subsample(efield_obwfs_real[i],
                                                                    n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        efields_per_mode_wfs_imag_sub = np.reshape(matrix_subsample(efield_obwfs_imag[i],
                                                                    n_sub_pix, n_sub_pix),
                                                   int(np.square(n_sub_pix))) / subsample_factor
        g_obwfs_downsampled[:, 0, i] = efields_per_mode_wfs_real_sub - ref_wfs_sub.real
        g_obwfs_downsampled[:, 1, i] = efields_per_mode_wfs_imag_sub - ref_wfs_sub.imag

    matrix = {"ref_image_plane": ref_coron,
              "ref_wfs_plane": ref_obwfs_downsampled,
              "senitivity_image_plane": g_coron,
              "sensitvity_wfs_plane": g_obwfs_downsampled}

    return matrix

def req_closedloop_calc_recursive(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor,
                                  t_exp, flux, Q, Niter, dh_mask, norm):
    """
    Parameters
    ----------
    Gcoro : numpy ndarray
        Sensitivity matrix calculated at the coronagraphic plane. It contains list of electric fields gathered by
        poking one segment with one mode at a time and subtracting the un-aberrated coronagraphic field.
    Gsensor : numpy ndarray
        sensitivity matrix calculated at the wavefront sensor plane: list of electric field at the wfs plane
        gathered by poking one segment with one mode at a time and subtracting the unaberrated electric field
        at the wfs plane.
    E0coro : numpy ndarray
        unaberrated coronagraphic electric field
    E0sensor : numpy ndarray
        reference un-aberrated electric field seen by wavefront sensor
    Dcoro : float
        Detector noise at the coronagraphic plane
    Dsensor : float
        Dectector noise at the wavefront sensor plane
    t_exp : float
        exposure time of the wavefront sensor in secs
    flux : scalar
        star magnitude in units number of photons
    Q : ndarray
        diagonal matrix, where each diagonal element is a mu coefficient
    Niter : int
        number of iterations
    dh_mask : ndarray
        dark hole mask made up of 1 and 0
    norm : float
        peak value of the ideal direct psf in focal plane

    Returns
    -------
    dict of intensity_WFS_hist, cal_I_hist, eps_hist, averaged_hist, contrasts
    """
    # TODO: Rewrite Gcoro and Gsensor's definition to include lo and hi spatial frequency aberration

    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    intensity_WFS_hist = np.zeros(Niter)
    cal_I_hist = np.zeros(Niter)
    eps_hist = np.zeros([Niter, r])
    averaged_hist = np.zeros(Niter)
    contrasts = []
    for pp in range(Niter):
        eps = np.random.multivariate_normal(np.zeros(r), P + Q * t_exp).reshape((1, 1, r))  # random modes
        G_eps = np.sum(Gsensor * eps, axis=2).reshape((N, 1, 2 * c)) + E0sensor  # electric field
        G_eps_squared = np.sum(G_eps * G_eps, axis=2, keepdims=True)
        G_eps_G = np.matmul(G_eps, Gsensor)
        G_eps_G_scaled = G_eps_G / np.sqrt(G_eps_squared + Dsensor / flux / t_exp)  # trick to save RAM
        cal_I = 4 * flux * t_exp * np.einsum("ijk,ijl->kl", G_eps_G_scaled, G_eps_G_scaled)  # information matrix
        P = np.linalg.inv(np.linalg.inv(P + Q * t_exp / 2) + cal_I)

        # Coronagraph
        G_eps_coron = np.sum(Gcoro * eps, axis=2).reshape((N_img, 1, 2 * c)) + E0coro
        G_eps_coron_squared = np.sum(G_eps_coron * G_eps_coron, axis=2, keepdims=True)
        intensity = G_eps_coron_squared * flux * t_exp + Dcoro

        # Wavefront sensor
        intensity_WFS = G_eps_squared * flux * t_exp + Dsensor

        # Archive
        test_DH0 = intensity[:, 0, 0] * dh_mask
        test_DH = np.mean(test_DH0[np.where(test_DH0 != 0)])
        contrasts.append(test_DH / flux / t_exp / norm)
        intensity_WFS_hist[pp] = np.sum(intensity_WFS) / flux
        cal_I_hist[pp] = np.mean(cal_I) / flux
        eps_hist[pp] = eps
        averaged_hist[pp] = np.mean(contrasts)

        outputs = {'intensity_WFS_hist': intensity_WFS_hist,
                   'cal_I_hist': cal_I_hist,
                   'eps_hist': eps_hist,
                   'averaged_hist': averaged_hist,
                   'contrasts': contrasts}
    return outputs


def req_closedloop_calc_batch(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor,
                              t_exp, flux, Q, Niter, dh_mask, norm):
    """
    Parameters
    ----------
    Gcoro : numpy ndarray
        sensitivity matrix calculated at the coronagraphic plane: list of electric fields gathered by
        poking one segment with one mode at a time and subtracting the un-aberrated coronagraphic field.
    Gsensor : numpy ndarray
        sensitivity matrix calculated at the wavefront sensor plane: list of electric field at the wfs plane
        gathered by poking one segment with one mode at a time and subtracting the unaberrated electric field
        at the wfs plane
    E0coro : numpy ndarray
        unaberrated coronagraphic electric field
    E0sensor : numpy ndarray
        reference un-aberrated electric field seen by wavefront sensor
    Dcoro : float
        Detector noise at the coronagraphic plane
    Dsensor : float
        Dectector noise at the wavefront sensor plane
    t_exp : float
        exposure time of the wavefront sensor in secs
    flux : scalar
        star magnitude in units number of photons
    Q : ndarray
        diagonal matrix, where each diagonal element is a tolerance (mu) coefficient.
    Niter : int
        number of iterations
    dh_mask : ndarray
        dark hole mask made up of 1 and 0
    norm : float
        peak value of the ideal direct psf in focal plane

    Returns
    -------
    dict of intensity_WFS_hist, cal_I_hist, eps_hist, averaged_hist, contrasts
    """

    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    intensity_WFS_hist = np.zeros(Niter)
    cal_I_hist = np.zeros(Niter)
    eps_hist = np.zeros([Niter, r])
    averaged_hist = np.zeros(Niter)
    contrasts = []
    for pp in range(Niter):
        eps = np.random.multivariate_normal(np.zeros(r), P + Q * t_exp).reshape((1, 1, r))  # random modes
        G_eps = np.sum(Gsensor * eps, axis=2).reshape((N, 1, 2 * c)) + E0sensor  # electric field
        G_eps_squared = np.sum(G_eps * G_eps, axis=2, keepdims=True)
        G_eps_G = np.matmul(G_eps, Gsensor)
        G_eps_G_scaled = G_eps_G / np.sqrt(G_eps_squared + Dsensor / flux / t_exp)  # trick to save RAM
        cal_I = 4 * flux * t_exp * np.einsum("ijk,ijl->kl", G_eps_G_scaled, G_eps_G_scaled)  # information matrix
        P = np.linalg.pinv(cal_I)

        # Coronagraph
        G_eps_coron = np.sum(Gcoro * eps, axis=2).reshape((N_img, 1, 2 * c)) + E0coro
        G_eps_coron_squared = np.sum(G_eps_coron * G_eps_coron, axis=2, keepdims=True)
        intensity = G_eps_coron_squared * flux * t_exp + Dcoro

        # Wavefront sensor
        intensity_WFS = G_eps_squared * flux * t_exp + Dsensor

        # Archive
        test_DH0 = intensity[:, 0, 0] * dh_mask
        test_DH = np.mean(test_DH0[np.where(test_DH0 != 0)])
        contrasts.append(test_DH / flux / t_exp / norm)
        intensity_WFS_hist[pp] = np.sum(intensity_WFS) / flux
        cal_I_hist[pp] = np.mean(cal_I) / flux
        eps_hist[pp] = eps
        averaged_hist[pp] = np.mean(contrasts)

        outputs = {'intensity_WFS_hist': intensity_WFS_hist,
                   'cal_I_hist': cal_I_hist,
                   'eps_hist': eps_hist,
                   'averaged_hist': averaged_hist,
                   'contrasts': contrasts}

    return outputs
