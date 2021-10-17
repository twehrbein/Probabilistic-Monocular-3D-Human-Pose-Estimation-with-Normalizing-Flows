import torch
import numpy as np

import models.model as model
import config as c
from utils.eval_functions import err_3dpe_parallel
from utils.data_utils import reinsert_root_joint_torch, root_center_poses
import data.data_h36m as data
from scipy.cluster.vq import kmeans


print("Program is running on: ", c.device)
print("EVALUATING EXPERIMENT: ", c.experiment_name, "\n")

actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
           'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']

# load model
inn = model.poseINN()
inn.to(c.device)
inn.load(c.load_model_name, c.device)
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

final_hypo_stddev = torch.zeros((3, 17))

quick_eval_stride = 16  # at dataset creation, every 4th frame is used => evaluate on every 64th frame
c.batch_size = 512

n_hypo = 200
std_dev = 1.0
# can select "mode poses" via k-means (see appendix of paper)
K_MEANS = False
n_clusters = 10

if K_MEANS:
    f = open(c.result_dir + "eval_" + c.m_name + "_withKmeans.txt", 'w')
else:
    f = open(c.result_dir + "eval_" + c.m_name + ".txt", 'w')
f.write("Evaluated on every %d -th frame with %d different hypotheses\nand standard dev of %.2f.\n\n\n" %
        (quick_eval_stride, n_hypo, std_dev))


def eval_hypo_stddev(poses_3d):
    # poses_3d.shape == (n_hypo, bs, 3, 17)
    # compute var over hypos and sum over poses for correct mean estimation over all poses in dataset
    return torch.sum(torch.std(poses_3d, dim=0), dim=0).cpu()


for action_idx, action in enumerate(actions):
    test_dataset = data.H36MDataset(c.test_file, quick_eval=True,
                                    quick_eval_stride=quick_eval_stride,
                                    actions=[action], train_set=False)

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False)

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

        if K_MEANS:
            # reduce n_hypos to n_clusters
            poses_3d_pred = poses_3d_pred.view(n_hypo, bs, 3*17).cpu().numpy()
            poses_3d_km_centers = np.zeros((n_clusters, bs, 3*17), dtype=np.float32)
            for idx in range(bs):
                codebook, distortion = kmeans(poses_3d_pred[:, idx], k_or_guess=n_clusters)
                poses_3d_km_centers[:, idx] = codebook
            poses_3d_pred = torch.from_numpy(poses_3d_km_centers).view(n_clusters, bs, 3, 17).cuda()

        # compute variance in x, y and z direction
        hypo_stddev += eval_hypo_stddev(poses_3d_pred)

        errors_proto1 = torch.mean(torch.sqrt(torch.sum((x_gt.view(bs, 3, 17)
                                                         - poses_3d_pred) ** 2, dim=2)), dim=2)

        # procrustes is faster on cpu
        poses_3d_pred = poses_3d_pred.cpu()
        x_gt = x_gt.cpu()
        if K_MEANS:
            x_gt = x_gt.repeat(n_clusters, 1)
        else:
            x_gt = x_gt.repeat(n_hypo, 1)

        errors_proto2 = err_3dpe_parallel(x_gt, poses_3d_pred.clone(), return_sum=False).view(-1, bs)

        print("Evaluated on batch %d of action %s" % (batch_idx + 1, action))
        # finished evaluating a single batch, need to compute hypo statistics per gt pose!
        # best hypos
        values, _ = torch.min(errors_proto1, dim=0)
        total_err_best_p1 += torch.sum(values).item()

        total_err_mean_p1 += torch.sum(torch.mean(errors_proto1, dim=0))
        total_err_mean_p2 += torch.sum(torch.mean(errors_proto2, dim=0))

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

    # write result for single action to file:
    f.write("Action: %s\n" % action)
    f.write("3D Protocol-I z_0: %.2f\n" % (total_err_z0_p1 / n_poses))
    f.write("3D Protocol-I best hypo: %.2f\n" % (total_err_best_p1 / n_poses))
    f.write("3D Protocol-I median hypo: %.2f\n" % (total_err_median_p1 / n_poses))
    f.write("3D Protocol-I mean hypo: %.2f\n" % (total_err_mean_p1 / n_poses))
    f.write("3D Protocol-I worst hypo: %.2f\n" % (total_err_worst_p1 / n_poses))

    f.write("3D Protocol-II z_0: %.2f\n" % (total_err_z0_p2 / n_poses))
    f.write("3D Protocol-II best hypo: %.2f\n" % (total_err_best_p2 / n_poses))
    f.write("3D Protocol-II median hypo: %.2f\n" % (total_err_median_p2 / n_poses))
    f.write("3D Protocol-II mean hypo: %.2f\n" % (total_err_mean_p2 / n_poses))
    f.write("3D Protocol-II worst hypo: %.2f\n" % (total_err_worst_p2 / n_poses))
    f.write("\n\n")
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

    final_hypo_stddev += (hypo_stddev / n_poses)

avg_z0_p1 = sum(final_errors_z0_p1) / len(final_errors_z0_p1)
avg_mean_p1 = sum(final_errors_mean_p1) / len(final_errors_mean_p1)
avg_best_p1 = sum(final_errors_best_p1) / len(final_errors_best_p1)
avg_worst_p1 = sum(final_errors_worst_p1) / len(final_errors_worst_p1)
avg_median_p1 = sum(final_errors_median_p1) / len(final_errors_median_p1)

avg_z0_p2 = sum(final_errors_z0_p2) / len(final_errors_z0_p2)
avg_mean_p2 = sum(final_errors_mean_p2) / len(final_errors_mean_p2)
avg_best_p2 = sum(final_errors_best_p2) / len(final_errors_best_p2)
avg_worst_p2 = sum(final_errors_worst_p2) / len(final_errors_worst_p2)
avg_median_p2 = sum(final_errors_median_p2) / len(final_errors_median_p2)

# results averaged over all actions
f.write("Average: \n")
f.write("3D Protocol-I z_0: %.2f\n" % avg_z0_p1)
f.write("3D Protocol-I best hypo: %.2f\n" % avg_best_p1)
f.write("3D Protocol-I median hypo: %.2f\n" % avg_median_p1)
f.write("3D Protocol-I mean hypo: %.2f\n" % avg_mean_p1)
f.write("3D Protocol-I worst hypo: %.2f\n" % avg_worst_p1)

f.write("3D Protocol-II z_0: %.2f\n" % avg_z0_p2)
f.write("3D Protocol-II best hypo: %.2f\n" % avg_best_p2)
f.write("3D Protocol-II median hypo: %.2f\n" % avg_median_p2)
f.write("3D Protocol-II mean hypo: %.2f\n" % avg_mean_p2)
f.write("3D Protocol-II worst hypo: %.2f\n" % avg_worst_p2)

std_dev_in_mm = final_hypo_stddev/len(actions)
# standard deviation in mm per dimension and per joint:
print("\nstd dev per joint and dim in mm:")
f.write("\n\n")
f.write("std dev per joint and dim in mm:\n")
for i in range(std_dev_in_mm.shape[1]):
    print("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                            std_dev_in_mm[2, i]))
    f.write("joint %d: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (i, std_dev_in_mm[0, i], std_dev_in_mm[1, i],
                                                                std_dev_in_mm[2, i]))

std_dev_means = torch.mean(std_dev_in_mm, dim=1)
print("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))
f.write("mean: std_x=%.2f, std_y=%.2f, std_z=%.2f\n" % (std_dev_means[0], std_dev_means[1], std_dev_means[2]))
