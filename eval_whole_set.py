import torch
import numpy as np

import models.model as model
import config as c
from utils.eval_functions import (compute_CP_list, pa_hypo_batch,
                                  err_3dpe_parallel, compute_3DPCK)
from utils.data_utils import (reinsert_root_joint_torch, root_center_poses)
import data.data_h36m
from sklearn.metrics import auc


print("Program is running on: ", c.device)
print("EVALUATING EXPERIMENT: ", c.experiment_name, "\n")

inn = model.poseINN()
inn.to(c.device)
inn.load(c.load_model_name, c.device)
inn.eval()

quick_eval_stride = 1
c.batch_size = 512

n_hypo = 200
std_dev = 1.0

cps_min_th = 1
cps_max_th = 300
cps_step = 1
cps_length = int((cps_max_th + 1 - cps_min_th) / cps_step)

f = open(c.result_dir + "HardSubset_eval_" + c.m_name + ".txt", 'w')
f.write("Evaluated on every %d -th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
        (quick_eval_stride, n_hypo, std_dev))

test_dataset = data.data_h36m.H36MDataset(c.test_file, quick_eval=False, train_set=False, hardsubset=True)

loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=False)

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

total_best_pck_oracle_p1 = 0
total_auc_cps_p2_best = torch.zeros((cps_length,))

hypo_stddev = torch.zeros((3, 17))


for batch_idx, sample in enumerate(loader):
    x = sample['poses_3d']
    y_gt = sample['p2d_hrnet']
    cond = sample['gauss_fits']
    bs = x.shape[0]

    # 2D to 3D mapping (reverse path)
    z0 = torch.zeros((bs, c.ndim_z), device=c.device)
    y_z0 = torch.cat((z0, y_gt), dim=1)
    with torch.no_grad():
        poses_3d_z0 = inn.reverse(y_z0, cond)

    poses_3d_z0 = reinsert_root_joint_torch(poses_3d_z0)
    poses_3d_z0 = root_center_poses(poses_3d_z0) * 1000
    x_gt = sample['p3d_gt']

    total_err_z0_p1 += torch.sum(torch.mean(torch.sqrt(torch.sum((x_gt.view(bs, 3, 17)
                                                                  - poses_3d_z0.view(bs, 3, 17)) ** 2,
                                                                 dim=1)), dim=1)).item()

    x_cpu = x_gt.cpu()
    poses_3d_z0 = poses_3d_z0.cpu()

    # protocol II
    total_err_z0_p2 += err_3dpe_parallel(x_cpu, poses_3d_z0)

    # sample multiple z
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

    hypo_stddev += torch.sum(torch.std(poses_3d_pred, dim=0), dim=0).cpu()

    errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.view(bs, 3, 17)
                                                     - poses_3d_pred) ** 2, dim=2)), dim=2)

    errors_pck_p1 = compute_3DPCK(x_gt.view(bs, 3, 17), poses_3d_pred)

    # procrustes is faster on cpu
    poses_3d_pred = poses_3d_pred.cpu()
    x_gt = x_gt.cpu()
    x_gt = x_gt.repeat(n_hypo, 1)

    errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False).view(-1, bs)
    poses_3d_pa = pa_hypo_batch(x_gt, poses_3d_pred.clone())
    poses_3d_pa = poses_3d_pa.view(n_hypo, bs, 3, 17)
    x_gt = x_gt.view(n_hypo, bs, 3, 17)[0, :]

    errors_auc_cps_p2 = compute_CP_list(x_gt.view(bs, 3, 17).cuda(), poses_3d_pa.cuda(), min_th=cps_min_th,
                                        max_th=cps_max_th, step=cps_step)

    print("Evaluated on batch %d" % (batch_idx + 1))
    # finished evaluating a single batch, need to compute hypo statistics per gt pose!
    # best hypos
    values, _ = torch.min(errors_proto1, dim=0)
    total_err_best_p1 += torch.sum(values).item()

    total_err_mean_p1 += torch.sum(torch.mean(errors_proto1, dim=0))
    total_err_mean_p2 += torch.sum(torch.mean(errors_proto2, dim=0))

    # best pck hypos
    values, _ = torch.max(errors_pck_p1, dim=0)
    total_best_pck_oracle_p1 += torch.sum(values).item()

    # best auc cps hypos
    values, _ = torch.max(errors_auc_cps_p2, dim=0)
    total_auc_cps_p2_best += torch.sum(values, dim=0)

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

# from list of cp values (one element per threshold), compute AUC CPS:
k_list = np.arange(cps_min_th, cps_max_th + 1, cps_step)
total_auc_cps_p2_best /= n_poses
cps_auc_p2_best = auc(k_list, total_auc_cps_p2_best.cpu().numpy())

# write result for single action to file:
f.write("Average: \n")
f.write("3D Protocol-I z_0: %.2f\n" % (total_err_z0_p1 / n_poses))
f.write("3D Protocol-I best hypo: %.2f\n" % (total_err_best_p1 / n_poses))
f.write("3D Protocol-I median hypo: %.2f\n" % (total_err_median_p1 / n_poses))
f.write("3D Protocol-I mean hypo: %.2f\n" % (total_err_mean_p1 / n_poses))
f.write("3D Protocol-I worst hypo: %.2f\n" % (total_err_worst_p1 / n_poses))

f.write("3D Protocol-II z_0: %.2f\n" % (total_err_z0_p2 / n_poses))
f.write("3D Protocol-II best hypo: %.2f\n" % (total_err_best_p2 / n_poses))
f.write("3D Protocol-II median hypo: %.2f\n" % (total_err_median_p2 / n_poses))
f.write("3D Protocol-II mean hypo: %.2f\n" % (total_err_mean_p2 / n_poses))
f.write("3D Protocol-II worst hypo: %.2f\n\n" % (total_err_worst_p2 / n_poses))

f.write("oracle best pck: %.4f\n" % (total_best_pck_oracle_p1 / n_poses))
f.write("oracle best PA auc cps: %.4f\n" % cps_auc_p2_best)

std_dev_in_mm = hypo_stddev / n_poses
# standard deviation in mm per dimension and per joint:
f.write("\n\n")
f.write("std dev per joint and dim in mm:\n")
for i in range(std_dev_in_mm.shape[1]):
    f.write("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                std_dev_in_mm[2, i]))

std_dev_means = torch.mean(std_dev_in_mm, dim=1)
f.write("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))
