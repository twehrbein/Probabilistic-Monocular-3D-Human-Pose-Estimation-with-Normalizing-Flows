import torch
import numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed
import multiprocessing
import config as c


def gauss2d_angle(xy, amp, x0, y0, sigma_x, theta, sigma_y):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    out = amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                          + c * ((y - y0) ** 2)))
    return out.ravel()


def fit_single_gaussian(esti_hm, p2d):
    # esti_hm.shape (16, 64, 64)
    # p2d.shape (32,)  x0x1x2...y0y1y2
    n_joints = esti_hm.shape[0]
    solution = []
    for j in range(esti_hm.shape[0]):  # iterate over each joint heatmap
        single_hm = esti_hm[j]
        w_guess = p2d[j]
        h_guess = p2d[j + n_joints]

        max_confi = single_hm.max().item()
        # initial guess:
        guess = [max_confi, w_guess, h_guess, c.gt_sigma, 0, c.gt_sigma]

        x = np.linspace(0, c.hm_w-1, c.hm_w)
        y = np.linspace(0, c.hm_h-1, c.hm_h)
        x, y = np.meshgrid(x, y)

        # first input is function to fit, second is array of x and y values (coordinates) and third is heatmap array
        try:
            pred_params, uncert_cov = opt.curve_fit(gauss2d_angle, (x, y), single_hm.flatten(), p0=guess)
        except:
            print("Runtime error in curve fitting")
            pred_params = guess
            data_fitted = gauss2d_angle((x, y), *pred_params).reshape(c.hm_w, c.hm_h)
            rmse_fit = np.sqrt(np.mean((single_hm.numpy() - data_fitted) ** 2))
            print(rmse_fit)

        data_fitted = gauss2d_angle((x, y), *pred_params).reshape(c.hm_w, c.hm_h)
        rmse_fit = np.sqrt(np.mean((single_hm.numpy() - data_fitted) ** 2))

        solution.extend(pred_params)
        solution.append(rmse_fit)
    # returns a single list of fit parameters:
    # A_0, mux_0, muy_0, sigmax_0, theta_0, sigmay_0, fiterr_0, A_1, ...
    return solution


def fit_gaussian_full_parallel(esti_hm, p2d):
    # input is batch of heatmaps
    # and corresponding estimated 2d joint coordinates needed for initial guess
    # (bs, 16, 64, 64), (bs, 32)
    esti_hm = torch.clamp(esti_hm, min=0, max=1)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(fit_single_gaussian)(esti_hm[i], p2d[i]) for i in range(len(esti_hm)))
    # returns an array (bs, #fit_parameters)
    return np.array(results)
