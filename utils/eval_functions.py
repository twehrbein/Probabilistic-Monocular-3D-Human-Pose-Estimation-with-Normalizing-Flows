import torch


def procrustes_torch_parallel(p_gt, p_pred):
    # p_gt and p_pred need to be of shape (-1, 3, #joints)
    # care: run on cpu! way faster than on gpu

    mu_gt = p_gt.mean(dim=2)
    mu_pred = p_pred.mean(dim=2)

    X0 = p_gt - mu_gt[:, :, None]
    Y0 = p_pred - mu_pred[:, :, None]

    ssX = (X0**2.).sum(dim=(1, 2))
    ssY = (Y0**2.).sum(dim=(1, 2))

    # centred Frobenius norm
    normX = torch.sqrt(ssX)
    normY = torch.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX[:, None, None]
    Y0 /= normY[:, None, None]

    # optimum rotation matrix of Y
    A = torch.bmm(X0, Y0.transpose(1, 2))

    try:
        U, s, V = torch.svd(A, some=True)
    except:
        print("ERROR IN SVD, could not converge")
        print("SVD INPUT IS:")
        print(A)
        print(A.shape)
        exit()

    T = torch.bmm(V, U.transpose(1, 2))

    # Make sure we have a rotation
    detT = torch.det(T)
    sign = torch.sign(detT)
    V[:, :, -1] *= sign[:, None]
    s[:, -1] *= sign
    T = torch.bmm(V, U.transpose(1, 2))

    traceTA = s.sum(dim=1)

    # optimum scaling of Y
    b = traceTA * normX / normY

    # standardised distance between X and b*Y*T + c
    d = 1 - traceTA**2

    # transformed coords
    scale = normX*traceTA
    Z = (scale[:, None, None] * torch.bmm(Y0.transpose(1, 2), T) + mu_gt[:, None, :]).transpose(1, 2)

    # transformation matrix
    c = mu_gt - b[:, None]*(torch.bmm(mu_pred[:, None, :], T)).squeeze()

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def err_3dpe_parallel(p_ref, p, return_sum=True, return_poses=False):
    p_ref, p = p_ref.view((-1, 3, 17)), p.view((-1, 3, 17))
    d, Z, tform = procrustes_torch_parallel(p_ref.clone(), p)
    if return_sum:
        err = torch.sum(torch.mean(torch.sqrt(torch.sum((p_ref - Z)**2, dim=1)), dim=1)).item()
    else:
        err = torch.mean(torch.sqrt(torch.sum((p_ref - Z)**2, dim=1)), dim=1)
    if not return_poses:
        return err
    else:
        return err, Z


def pa_hypo_batch(p_ref, p):
    p_ref, p = p_ref.view((-1, 3, 17)), p.view((-1, 3, 17))
    d, Z, tform = procrustes_torch_parallel(p_ref.clone(), p)
    return Z


def compute_3DPCK(poses_gt, poses_pred, threshold=150):
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    # see https://arxiv.org/pdf/1611.09813.pdf
    joints_to_use = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
    if len(poses_pred.shape) == 3:
        # compute distances to gt:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, joints_to_use])**2, dim=1))
        pck = torch.count_nonzero(distances < threshold, dim=1) / len(joints_to_use)
    else:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, :, joints_to_use]) ** 2, dim=2))
        pck = torch.count_nonzero(distances < threshold, dim=2) / len(joints_to_use)
    return pck


def compute_CP(poses_gt, poses_pred, threshold=180):
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    joints_to_use = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if len(poses_pred.shape) == 3:
        # compute distances to gt:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, joints_to_use])**2, dim=1))
        correct_poses = torch.count_nonzero(distances < threshold, dim=1) == len(joints_to_use)
    else:
        distances = torch.sqrt(torch.sum((poses_gt[:, :, joints_to_use]
                                          - poses_pred[:, :, :, joints_to_use]) ** 2, dim=2))
        # distances.shape (n_hypo, bs, 16)
        # pose is correct if all joints of a pose have distance < threshold
        correct_poses = torch.count_nonzero(distances < threshold, dim=2) == len(joints_to_use)
    return correct_poses


def compute_CP_list(poses_gt, poses_pred, min_th=1, max_th=300, step=1):
    # computes Correct Poses Score (CPS) (https://arxiv.org/abs/2011.14679)
    # for different thresholds
    # poses_pred.shape (bs, 3, 17) or (n_hypo, bs, 3, 17)
    assert len(poses_pred.shape) == 3 or len(poses_pred.shape) == 4
    thresholds = torch.arange(min_th, max_th+1, step).tolist()

    if len(poses_pred.shape) == 3:
        cp_values = torch.empty((poses_pred.shape[0], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    else:
        cp_values = torch.empty((poses_pred.shape[0], poses_pred.shape[1], len(thresholds)), dtype=torch.double)
        for i, threshold in enumerate(thresholds):
            cp_values[:, :, i] = compute_CP(poses_gt, poses_pred, threshold=threshold)
    return cp_values
