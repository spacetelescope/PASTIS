import numpy as np

def req_closedloop_calc_recursive(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor, t_exp, flux, Q, Niter,dh_mask,norm):
    """
    Parameters:
    ----------
    Gcoro: numpy ndarray
        sensitivity matrix calculated at the coronagraphic plane: list of electric fields gathered by
        poking one segment with one mode at a time and subtracting the un-aberrated coronagraphic field.
    Gsensor: numpy ndarray
        sensitivity matrix calculated at the wavefront sensor plane: list of electric field at the wfs plane
        gathered by poking one segment with one mode at a time ans subtracting the unaberrated electric field
        at the wfs plane
    E0coro: numpy ndarray
        unaberrated coronagraphic electric field
    E0sensor: numpy ndarray
        reference un-aberrated electric field seen by wavefront sensor
    Dcoro:
    """

    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    contrast_hist = np.zeros(Niter)
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
        #         P = np.linalg.inv(cal_I)

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
        #         print("est. contrast", np.mean(contrasts))

        outputs = {'intensity_WFS_hist': intensity_WFS_hist,
                   'cal_I_hist': cal_I_hist,
                   'eps_hist': eps_hist,
                   'averaged_hist': averaged_hist,
                   'contrasts': contrasts}
    return outputs

def req_closedloop_calc_batch(Gcoro, Gsensor, E0coro, E0sensor, Dcoro, Dsensor, t_exp, flux, Q, Niter, dh_mask, norm):


    P = np.zeros(Q.shape)  # WFE modes covariance estimate
    r = Gsensor.shape[2]
    N = Gsensor.shape[0]
    N_img = Gcoro.shape[0]
    c = 1
    # Iterations of ALGORITHM 1
    contrast_hist = np.zeros(Niter)
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
        #         P = np.linalg.inv(np.linalg.inv(P+Q*t_exp/2) + cal_I)
        P = np.linalg.pinv(cal_I)

        # Coronagraph
        G_eps_coron = np.sum(Gcoro * eps, axis=2).reshape((N_img, 1, 2 * c)) + E0coro
        G_eps_coron_squared = np.sum(G_eps_coron * G_eps_coron, axis=2, keepdims=True)
        intensity = G_eps_coron_squared * flux * t_exp + Dcoro

        # Wavefront sensor
        intensity_WFS = G_eps_squared * flux * t_exp + Dsensor

        # Archive
        test_DH0 = intensity[:, 0, 0] *dh_mask
        test_DH = np.mean(test_DH0[np.where(test_DH0 != 0)])
        contrasts.append(test_DH / flux / t_exp / norm)
        intensity_WFS_hist[pp] = np.sum(intensity_WFS) / flux
        cal_I_hist[pp] = np.mean(cal_I) / flux
        eps_hist[pp] = eps
        averaged_hist[pp] = np.mean(contrasts)
    #         print("est. contrast", np.mean(contrasts))
    #         print("est. contrast", np.mean(contrasts))

    outputs = {'intensity_WFS_hist': intensity_WFS_hist,
               'cal_I_hist': cal_I_hist,
               'eps_hist': eps_hist,
               'averaged_hist': averaged_hist,
               'contrasts': contrasts}

    return outputs
