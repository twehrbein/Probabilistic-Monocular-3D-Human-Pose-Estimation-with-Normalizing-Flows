import numpy as np
import torch
import config as c


# 17 Joints H36M skeleton
H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Thorax'
H36M_NAMES[9] = 'Neck/Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

# 16 joints MPII skeleton used for the HRNet detections
MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Spine'
MPII_NAMES[8]  = 'Thorax'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'

bones = {'h36m': np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11],
                           [11, 12], [7, 13], [13, 14], [14, 15]])}

# select and permute h36m joints to be consistent with MPII skeleton
H36M17j_TO_MPII = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]


def reinsert_root_joint_torch(poses):
    # reinserts the root to be consistent with other implementations using 17 joints
    # poses.shape == (-1, 48)
    assert poses.shape[1] == 48
    poses = poses.view((-1, 3, 16))
    # root is the midpoint of right and left hip:
    poses = torch.cat((torch.mean(poses[:, :, (0, 3)], dim=2, keepdim=True), poses), dim=2)
    return poses.view((-1, 3 * 17))


def align_gauss_fit_and_p3d(poses_3d, sigma_x, sigma_y, theta):
    # aligns h36m 3d joints with joint ordering of gaussian fits
    # remove hip joints if not already done
    if sigma_y.shape[1] == 16:
        cond_fits_used = [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        cond_fits_used = torch.arange(c.COND_JOINTS)

    theta = theta[:, cond_fits_used]
    sigma_y = sigma_y[:, cond_fits_used]
    sigma_x = sigma_x[:, cond_fits_used]

    h36m_to_mpi_fits_used = [2, 1, 4, 5, 6, 7, 9, 15, 14, 13, 10, 11, 12]

    # remove nose from 3d poses and match joint ordering
    poses_3d = poses_3d[:, :, :, h36m_to_mpi_fits_used]
    return poses_3d, sigma_x, sigma_y, theta


def normalize_poses(poses):
    # mean center 2d/3d poses and additionally divide 2d poses by std
    # poses are numpy array of shape (-1, 32) or (-1, 48)
    assert poses.ndim == 2 and (poses.shape[1] == 32 or poses.shape[1] == 48)
    if poses.shape[1] == 32:
        poses_2d = poses.reshape(-1, 2, 16)

        poses_2d_x = poses_2d[:, 0]
        poses_2d_x = poses_2d_x - poses_2d_x.mean(axis=1, keepdims=True)

        poses_2d_y = poses_2d[:, 1]
        poses_2d_y = poses_2d_y - poses_2d_y.mean(axis=1, keepdims=True)

        poses = np.concatenate((poses_2d_x, -poses_2d_y), axis=1)
        poses = poses / poses.std(axis=1, keepdims=True)
    else:
        poses_3d = poses.reshape(-1, 3, 16)
        poses_3d *= (1 / 1000)

        poses_3d_x = poses_3d[:, 0]
        poses_3d_x = poses_3d_x - poses_3d_x.mean(axis=1, keepdims=True)

        poses_3d_y = poses_3d[:, 1]
        poses_3d_y = poses_3d_y - poses_3d_y.mean(axis=1, keepdims=True)

        poses_3d_z = poses_3d[:, 2]
        poses_3d_z = poses_3d_z - poses_3d_z.mean(axis=1, keepdims=True)

        poses = np.concatenate((poses_3d_x, -poses_3d_y, -poses_3d_z), axis=1)
    return poses


def root_center_poses(poses_3d):
    # poses_3d is torch array of shape (-1, 51)
    assert poses_3d.shape[1] == 51
    poses_3d = poses_3d.view(-1, 3, 17)
    poses_3d -= poses_3d[:, :, 0, None]
    poses_3d = poses_3d.view(-1, 3 * 17)
    return poses_3d


def preprocess_gaussian_fits(gauss_fits):
    gauss_fits = gauss_fits.reshape(-1, 16, 7)
    # preprocess data:
    # clip amplitude:
    gauss_fits[:, :, 0] = np.clip(gauss_fits[:, :, 0], 0.0, 1.2)
    # clip mean x y
    gauss_fits[:, :, 1:3] = np.clip(gauss_fits[:, :, 1:3], 0, c.hm_h - 1)
    # normalize
    mean_x_y = gauss_fits[:, :, [1, 2]]
    mean_x_y = mean_x_y.transpose((0, 2, 1))
    mean_x_y = mean_x_y.reshape(-1, 16 * 2)
    mean_x_y = normalize_poses(mean_x_y)
    gauss_fits[:, :, [1, 2]] = mean_x_y.reshape(-1, 2, 16).transpose((0, 2, 1))

    # clip std devs to reasonable range
    gauss_fits[:, :, 3] = np.clip(gauss_fits[:, :, 3], c.gt_sigma / 10.0, c.gt_sigma * 5.0)
    gauss_fits[:, :, 5] = np.clip(gauss_fits[:, :, 5], c.gt_sigma / 10.0, c.gt_sigma * 5.0)

    # want to scale the standard devs differently, since std=2 equals an exact prediction,
    # while std=3 already shows great uncertainties
    sigma_x = gauss_fits[:, :, 3]
    sigma_y = gauss_fits[:, :, 5]

    # scale sigma uncertain detections
    sigma_x[sigma_x > c.gt_sigma * 1.05] *= 2.0
    sigma_y[sigma_y > c.gt_sigma * 1.05] *= 2.0

    # clip theta angle of first eigenvector
    gauss_fits[:, :, 4] = gauss_fits[:, :, 4] % np.pi

    # normalize roughly between 0 and 1:
    scale_factor = 10 * c.gt_sigma

    # is only for normalization! otherwise training is unstable
    gauss_fits[:, :, 3] = ((gauss_fits[:, :, 3]) / scale_factor)
    gauss_fits[:, :, 5] = ((gauss_fits[:, :, 5]) / scale_factor)

    # convert sigma_x, sigma_y, theta into cov matrix:
    # C_matrix = V @ L @ V^-1, V is eigenvector matrix and L eigenvalues
    # sigma_x**2 and sigma_y**2 are the variances and therefore the eigenvalues
    cos = np.cos(gauss_fits[:, :, 4])
    sin = np.sin(gauss_fits[:, :, 4])
    eigen_vectors = np.array([[cos, -sin], [sin, cos]])

    zeros = np.zeros_like((gauss_fits[:, :, 3]))
    eigen_values = np.array([[gauss_fits[:, :, 3] ** 2, zeros], [zeros, gauss_fits[:, :, 5] ** 2]])

    eigen_values = eigen_values.reshape(2, 2, -1).transpose((2, 0, 1))
    eigen_vectors = eigen_vectors.reshape(2, 2, -1).transpose((2, 0, 1))

    C = np.matmul(np.matmul(eigen_vectors, eigen_values), eigen_vectors.transpose((0, 2, 1)))
    C = C.reshape(gauss_fits.shape[0], gauss_fits.shape[1], 2, 2)
    gauss_fits[:, :, 3] = C[:, :, 0, 0]
    gauss_fits[:, :, 4] = C[:, :, 0, 1]
    gauss_fits[:, :, 5] = C[:, :, 1, 1]

    # remove hip joints
    gauss_fits = gauss_fits[:, [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], :]
    # also remove fit error
    gauss_fits = gauss_fits[:, :, [0, 1, 2, 3, 4, 5]]
    assert gauss_fits.shape[2] == c.COND_LENGTH and gauss_fits.shape[1] == c.COND_JOINTS
    return gauss_fits
