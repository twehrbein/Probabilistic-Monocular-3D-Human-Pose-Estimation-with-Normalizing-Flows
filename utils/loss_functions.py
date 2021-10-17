import config as c
import torch
from torch import autograd
from utils.data_utils import align_gauss_fit_and_p3d
from torch.nn.functional import relu_


def mmd_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))

    # kernel computation
    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def calc_gradient_penalty(netD, real_data, fake_data, gp_lambda=10):
    bs = real_data.shape[0]
    # alpha contains one scalar between [0, 1] for each element in batch
    alpha = torch.rand(bs, 1)
    alpha = alpha.expand(bs, int(real_data.nelement() / bs)).contiguous()
    alpha = alpha.view(bs, 3*16)
    alpha = alpha.to(c.device)

    fake_data = fake_data.view(bs, 3*16)
    # interpolate between real and fake data on a straight line, depending on alpha [0, 1]
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(c.device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    # compute gradients of the critic at the interpolated points! (not w.r.t. to
    # network weights!)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(c.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    # gradient penalty penalizes the model if its gradient norm moves away from 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def covariance_loss(poses_3d, gt_var_x=0, gt_var_y=0, gt_cov=0):
    # gets as input the poses and the corresponding covariance parameters
    assert len(poses_3d.shape) == 4  # (n_hypo, bs, 3, 16)
    poses_3d_spatial = poses_3d[:, :, 0:2, :]

    # need to root center poses
    poses_3d_spatial -= torch.mean(poses_3d_spatial[:, :, :, (0, 3)], dim=3, keepdim=True)

    # remove root joint from 2d poses/heatmaps and nose joint from p3d
    poses_3d_spatial, gt_var_x, gt_var_y, gt_cov = align_gauss_fit_and_p3d(poses_3d_spatial, gt_var_x,
                                                                           gt_var_y, gt_cov)

    # estimate covariance matrix of poses
    def estimate_cov(x, y):
        xbar, ybar = x.mean(dim=0), y.mean(dim=0)
        return torch.sum((x - xbar) * (y - ybar), dim=0) / (len(x) - 1)

    cov = estimate_cov(poses_3d_spatial[:, :, 0, :], poses_3d_spatial[:, :, 1, :])
    var_x = torch.var(poses_3d_spatial[:, :, 0, :], dim=0)
    var_y = torch.var(poses_3d_spatial[:, :, 1, :], dim=0)

    # penalize if variances are different than in condition
    # need to revert preprocess scaling
    scale_factor = c.hm_px_to_mm * (10 * c.gt_sigma)**2

    gt_var_x *= scale_factor
    gt_var_y *= scale_factor
    gt_cov *= scale_factor
    # don't compute loss if detector is certain
    var_th = (c.gt_sigma * 1.05)**2 * c.hm_px_to_mm
    indices = torch.logical_or(gt_var_x > var_th, gt_var_y > var_th).double()

    loss = indices * torch.sqrt((relu_(gt_var_x - var_x))**2 + (relu_(gt_var_y - var_y))**2
                                + (gt_cov - cov)**2 + 1e-30)
    return torch.mean(loss)
