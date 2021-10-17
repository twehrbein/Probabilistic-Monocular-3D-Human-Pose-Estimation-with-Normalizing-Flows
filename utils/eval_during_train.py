from torch.utils.data import DataLoader
import torch
import config as c
from utils.data_utils import reinsert_root_joint_torch, root_center_poses
from utils.eval_functions import err_3dpe_parallel
import data.data_h36m as data

actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']

quick_eval_stride = 16
batch_size = 512
n_hypo = 200
std_dev = 1.0


def eval_val_set(inn):
    inn.eval()

    # Protocol-I
    final_errors_z0_p1 = []
    final_errors_mean_p1 = []
    final_errors_best_p1 = []
    final_errors_worst_p1 = []
    final_errors_median_p1 = []

    # Protocol-II
    final_errors_z0_p2 = []
    final_errors_mean_p2 = []
    final_errors_best_p2 = []
    final_errors_worst_p2 = []
    final_errors_median_p2 = []
    for action in actions:
        test_dataset = data.H36MDataset(c.test_file, quick_eval=True,
                                        quick_eval_stride=quick_eval_stride,
                                        actions=[action], train_set=False)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        n_poses = len(test_dataset)

        total_err_z0_p1 = 0
        total_err_mean_p1 = 0
        total_err_worst_p1 = 0
        total_err_best_p1 = 0
        total_err_median_p1 = 0

        total_err_z0_p2 = 0
        total_err_mean_p2 = 0
        total_err_worst_p2 = 0
        total_err_best_p2 = 0
        total_err_median_p2 = 0

        for sample in loader:
            x = sample['poses_3d']
            y_gt = sample['p2d_hrnet']
            cond = sample['gauss_fits']

            bs = x.shape[0]
            # 2D to 3D mapping (reverse path)
            z0 = torch.zeros((bs, c.ndim_z), device=c.device)
            y_z0 = torch.cat((z0, y_gt), dim=1)
            with torch.no_grad():
                poses_3d_pred_z0 = inn.reverse(y_z0, cond)

            x_gt = sample['p3d_gt']
            poses_3d_pred_z0 = reinsert_root_joint_torch(poses_3d_pred_z0)
            poses_3d_pred_z0 = root_center_poses(poses_3d_pred_z0) * 1000

            total_err_z0_p1 += torch.sum(torch.mean(torch.sqrt(torch.sum((x_gt.view(bs, 3, 17)
                                                                          - poses_3d_pred_z0.view(bs, 3, 17)) ** 2,
                                                                         dim=1)), dim=1)).item()
            x_cpu = x_gt.cpu()
            poses_3d_pred_z0 = poses_3d_pred_z0.cpu()

            # protocol II
            total_err_z0_p2 += err_3dpe_parallel(x_cpu, poses_3d_pred_z0)

            # sample multiple hypos
            z_all = std_dev * torch.randn(n_hypo, bs, c.ndim_z, device=c.device)
            y_gt = y_gt[None, :].repeat(n_hypo, 1, 1)
            y_rand = torch.cat((z_all, y_gt), dim=2)
            y_rand = y_rand.view(-1, c.ndim_y + c.ndim_z)
            cond = cond[None].repeat(n_hypo, 1, 1).view(-1, inn.cond_in_size)

            with torch.no_grad():
                poses_3d_pred = inn.reverse(y_rand, cond)

            poses_3d_pred = reinsert_root_joint_torch(poses_3d_pred)
            poses_3d_pred = root_center_poses(poses_3d_pred) * 1000
            poses_3d_pred = poses_3d_pred.view(n_hypo, bs, 3, 17)
            errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.view(bs, 3, 17)
                                                             - poses_3d_pred) ** 2, dim=2)), dim=2)

            # procrustes is faster on cpu
            poses_3d_pred = poses_3d_pred.cpu()
            x_gt = x_gt.cpu()
            x_gt = x_gt.repeat(n_hypo, 1)

            errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False).view(-1, bs)

            # finished evaluating a single batch, need to compute hypo statistics per gt pose!
            total_err_mean_p1 += torch.sum(torch.mean(errors_proto1, dim=0)).item()
            total_err_mean_p2 += torch.sum(torch.mean(errors_proto2, dim=0)).item()

            # best hypos
            values, _ = torch.min(errors_proto1, dim=0)
            total_err_best_p1 += torch.sum(values).item()

            # worst hypos
            values, _ = torch.max(errors_proto1, dim=0)
            total_err_worst_p1 += torch.sum(values).item()

            # median hypos
            values, _ = torch.median(errors_proto1, dim=0)
            total_err_median_p1 += torch.sum(values).item()
            # Protocol-II:
            # best hypos
            values, _ = torch.min(errors_proto2, dim=0)
            total_err_best_p2 += torch.sum(values).item()

            # worst hypos
            values, _ = torch.max(errors_proto2, dim=0)
            total_err_worst_p2 += torch.sum(values).item()

            # median hypos
            values, _ = torch.median(errors_proto2, dim=0)
            total_err_median_p2 += torch.sum(values).item()

        final_errors_z0_p1.append(total_err_z0_p1 / n_poses)
        final_errors_mean_p1.append(total_err_mean_p1 / n_poses)
        final_errors_best_p1.append(total_err_best_p1 / n_poses)
        final_errors_worst_p1.append(total_err_worst_p1 / n_poses)
        final_errors_median_p1.append(total_err_median_p1 / n_poses)

        final_errors_z0_p2.append(total_err_z0_p2 / n_poses)
        final_errors_mean_p2.append(total_err_mean_p2 / n_poses)
        final_errors_best_p2.append(total_err_best_p2 / n_poses)
        final_errors_worst_p2.append(total_err_worst_p2 / n_poses)
        final_errors_median_p2.append(total_err_median_p2 / n_poses)

    test_results = dict()

    test_results['z0_p1'] = sum(final_errors_z0_p1) / len(final_errors_z0_p1)
    test_results['zr_mean_p1'] = sum(final_errors_mean_p1) / len(final_errors_mean_p1)
    test_results['zr_best_p1'] = sum(final_errors_best_p1) / len(final_errors_best_p1)
    test_results['zr_worst_p1'] = sum(final_errors_worst_p1) / len(final_errors_worst_p1)
    test_results['zr_median_p1'] = sum(final_errors_median_p1) / len(final_errors_median_p1)

    test_results['z0_p2'] = sum(final_errors_z0_p2) / len(final_errors_z0_p2)
    test_results['zr_mean_p2'] = sum(final_errors_mean_p2) / len(final_errors_mean_p2)
    test_results['zr_best_p2'] = sum(final_errors_best_p2) / len(final_errors_best_p2)
    test_results['zr_worst_p2'] = sum(final_errors_worst_p2) / len(final_errors_worst_p2)
    test_results['zr_median_p2'] = sum(final_errors_median_p2) / len(final_errors_median_p2)

    return test_results
