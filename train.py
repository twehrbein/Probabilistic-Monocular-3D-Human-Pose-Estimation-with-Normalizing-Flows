from time import time
import torch
import os
from shutil import copyfile

import models.model as model
import config as c
from utils.loss_functions import covariance_loss, mmd_multiscale, calc_gradient_penalty
from utils.train_log import draw_loss, write_log, draw_mpjpe_all
import data.data_h36m as data
from utils.eval_during_train import eval_val_set

from utils.data_utils import reinsert_root_joint_torch, root_center_poses
from models.critic_network import CriticNetwork


print("Program is running on: ", c.device)
print("RUNNING EXPERIMENT: ", c.experiment_name, "\n")


if not os.path.isdir(c.result_dir):
    os.mkdir(c.result_dir)
# save files for bookkeeping of experiments
copyfile('train.py', c.result_dir + 'train.py')
copyfile('models/model.py', c.result_dir + 'model.py')
copyfile('models/critic_network.py', c.result_dir + 'critic_network.py')
copyfile('config.py', c.result_dir + 'config.py')
copyfile('data/data_h36m.py', c.result_dir + 'data_h36m.py')
copyfile('utils/data_utils.py', c.result_dir + 'data_utils.py')
copyfile('utils/loss_functions.py', c.result_dir + 'loss_functions.py')
copyfile('utils/eval_functions.py', c.result_dir + 'eval_functions.py')
copyfile('utils/eval_during_train.py', c.result_dir + 'eval_during_training.py')

test_dataset = data.H36MDataset(c.train_file, train_set=True)
train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True)

# create the model:
inn = model.poseINN()
inn.to(c.device)
inn.optimizer.zero_grad()

critic = CriticNetwork().to(c.device)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=c.lr, betas=(0.5, 0.9))
lr_scheduler_critic = torch.optim.lr_scheduler.StepLR(optimizer_critic, step_size=150, gamma=0.5)

n_hypo = 200

loss_latent = mmd_multiscale
l1_loss = torch.nn.L1Loss()


def train():
    inn.train()

    l2d_tot = 0
    lmmd_tot = 0
    ldet_tot = 0
    lmb_tot = 0
    lhm_tot = 0
    lgen_tot = 0
    lcritic_tot = 0
    dataloader_iterator = iter(train_loader)

    for train_iteration in range(c.n_its_per_epoch):
        ############################
        # (1) Update Critic network
        ###########################
        for p in critic.parameters():
            p.requires_grad = True

        for i_critic in range(c.CRITIC_ITERS):
            optimizer_critic.zero_grad()

            # train with real data
            sample = next(dataloader_iterator)
            poses3d = sample['poses_3d'].cuda()
            d_real = -1 * critic(poses3d).mean()
            d_real.backward()

            # train with fake
            sample = next(dataloader_iterator)
            cond = sample['gauss_fits']
            poses_2d = sample['p2d_hrnet']
            z = torch.randn(c.batch_size, c.ndim_z, device=c.device)
            y_rand = torch.cat((z, poses_2d), dim=1)
            with torch.no_grad():
                poses3d_fake = inn.reverse(y_rand, cond)
            d_fake = critic(poses3d_fake).mean()
            d_fake.backward()

            gradient_penalty = calc_gradient_penalty(critic, poses3d, poses3d_fake)
            gradient_penalty.backward()
            optimizer_critic.step()

            d_real = -1 * d_real
            lcritic_tot += (d_fake - d_real + gradient_penalty).item()

        ############################
        # (2) Update G network
        ###########################
        for p in critic.parameters():
            p.requires_grad = False  # freeze the critic

        for i_inn in range(c.GEN_ITERS):
            inn.optimizer.zero_grad()

            sample = next(dataloader_iterator)
            cond = sample['gauss_fits']
            x = sample['poses_3d']
            y = sample['p2d_hrnet']
            poses_2d = y.clone()

            y = torch.cat((torch.randn(c.batch_size, c.ndim_z, device=c.device), y), dim=1)
            # ----- INN Forward step, estimates 2d + latent vector -----
            output = inn(x, cond)

            # supervised l1 loss on y
            loss_forward = l1_loss(output[:, -c.ndim_y:], y[:, -c.ndim_y:])
            l2d_tot += loss_forward.item()

            # block gradient w.r.t. y
            output_block_grad = torch.cat((output[:, :c.ndim_z],
                                           output[:, -c.ndim_y:].detach()), dim=1)
            # loss that enforces z to follow normal distribution,
            # and y and z to be independent on convergence
            mmd_for = c.lambd_mmd * loss_latent(output_block_grad, y)
            lmmd_tot += mmd_for.item()
            loss_forward += mmd_for
            loss_forward.backward()

            # ----- reverse step (y (2D) to x (3D)) -----
            z_det = (output.detach()[:, :c.ndim_z])
            y_rev = torch.cat((z_det, poses_2d), dim=1)
            z_rand = torch.randn(c.batch_size, c.ndim_z, device=c.device)
            y_rev_rand = torch.cat((z_rand, poses_2d), dim=1)
            # reverse path is computed twice: once with z computed by forward path, once with random z
            x_rand = inn.reverse(y_rev_rand, cond)
            x_det = inn.reverse(y_rev, cond)

            # l1 loss on 'deterministic' 3d reconstruction
            loss_inverse = c.lambd_det * l1_loss(x_det, x)
            ldet_tot += loss_inverse.item()

            # create n_hypo hypos
            z_all = torch.randn(n_hypo, c.batch_size, c.ndim_z, device=c.device)
            y_gt = poses_2d[None, :].repeat(n_hypo, 1, 1)
            y_rand = torch.cat((z_all, y_gt), dim=2)
            y_rand = y_rand.view(-1, c.ndim_z+c.ndim_y)

            gt_cov = cond.view(-1, c.COND_JOINTS, c.COND_LENGTH)[:, :, 4]
            gt_var_x = cond.view(-1, c.COND_JOINTS, c.COND_LENGTH)[:, :, 3]
            gt_var_y = cond.view(-1, c.COND_JOINTS, c.COND_LENGTH)[:, :, 5]

            cond = cond[None].repeat(n_hypo, 1, 1).view(-1, c.COND_JOINTS*c.COND_LENGTH)
            x_pred_hypos = inn.reverse(y_rand, cond)
            # calculate loss that forces the network to best reflect the uncertainties
            loss_hm = c.lambd_hm * covariance_loss(x_pred_hypos.view(n_hypo, c.batch_size, 3, 16).clone(),
                                                   gt_cov=gt_cov, gt_var_x=gt_var_x, gt_var_y=gt_var_y)
            loss_inverse += loss_hm
            lhm_tot += loss_hm.item()
            if c.BEST_OF_M_LOSS:
                x_pred_hypos = reinsert_root_joint_torch(x_pred_hypos)
                x_pred_hypos = root_center_poses(x_pred_hypos).view(n_hypo, c.batch_size, 3, 17)

                # rank hypotheses according to MPJPE:
                x_gt = reinsert_root_joint_torch(x)
                x_gt = root_center_poses(x_gt)
                errors_mpjpe = 1000 * torch.mean(torch.sqrt(torch.sum((x_gt.view(c.batch_size, 3, 17)
                                                                       - x_pred_hypos) ** 2, dim=2)), dim=2)

                # compute mean of TOP_K best poses:
                x_pred_hypos = x_pred_hypos.view(n_hypo, c.batch_size, 3*17).transpose(0, 1)
                indices = torch.argsort(errors_mpjpe, dim=0, descending=False).transpose(0, 1)
                sorted_3d_preds = x_pred_hypos[torch.arange(c.batch_size)[:, None], indices]
                best_k_hypos = torch.mean(sorted_3d_preds[:, 0:c.TOP_K], dim=1)

                loss_mb = c.lambd_mb * l1_loss(best_k_hypos, x_gt)

                lmb_tot += loss_mb.item()
                loss_inverse += loss_mb

            loss_inverse.backward(retain_graph=True)

            # train with critic feedback
            gen_cost = -1 * critic(x_rand).mean()
            lgen_tot += gen_cost.item()
            gen_cost.backward()

            for p in inn.trainable_parameters:
                p.grad.data.clamp_(-15.00, 15.00)

            inn.optimizer.step()

        if c.VERBOSE:
            print("### iteration %d of %d" % (train_iteration + 1, c.n_its_per_epoch), end="\r")
    return l2d_tot / c.n_its_per_epoch, lmmd_tot / c.n_its_per_epoch, ldet_tot / c.n_its_per_epoch,\
        lmb_tot / c.n_its_per_epoch, lhm_tot / c.n_its_per_epoch, lgen_tot / c.n_its_per_epoch,\
        lcritic_tot / (c.n_its_per_epoch * c.CRITIC_ITERS)


loss_sum_hist = []
l2d_hist, ldet_hist, lmmd_hist, lhm_hist, lmb_hist = [], [], [], [], []
lgen_hist, lcritic_hist = [], []

z0_p1_hist, z0_p2_hist = [], []
rand_mean_p1_hist, rand_best_p1_hist, rand_worst_p1_hist, rand_median_p1_hist = [], [], [], []
rand_mean_p2_hist, rand_best_p2_hist, rand_worst_p2_hist, rand_median_p2_hist = [], [], [], []

try:
    t_start = time()
    for i_epoch in range(c.n_epochs):
        t = time()
        l2d, lmmd, ldet, lmb, lhm, lgen, lcritic = train()
        loss = l2d + lmmd + ldet + lmb + lhm + lgen
        print("epoch %d, loss: %f, l2d: %f, lmmd: %f, ldet: %f, lmb: %f, lhm: %f, lgen: %f, lcritic: %f" % (
            i_epoch, loss, l2d, lmmd, ldet, lmb, lhm, lgen, lcritic))

        loss_sum_hist.append(loss)
        l2d_hist.append(l2d)
        lmmd_hist.append(lmmd)
        ldet_hist.append(ldet)
        lmb_hist.append(lmb)
        lhm_hist.append(lhm)
        lgen_hist.append(lgen)
        lcritic_hist.append(lcritic)

        # evaluate on test set action wise
        t_r = eval_val_set(inn)

        print("Testset: ProtoI: z0: %.2f, best: %.2f, worst: %.2f; "
              "ProtoII: z0: %.2f, best: %.2f, worst: %.2f"
              % (t_r['z0_p1'], t_r['zr_best_p1'], t_r['zr_worst_p1'],
                 t_r['z0_p2'], t_r['zr_best_p2'], t_r['zr_worst_p2']))

        z0_p1_hist.append(t_r['z0_p1'])
        z0_p2_hist.append(t_r['z0_p2'])

        rand_best_p1_hist.append(t_r['zr_best_p1'])
        rand_median_p1_hist.append(t_r['zr_median_p1'])
        rand_mean_p1_hist.append(t_r['zr_mean_p1'])
        rand_worst_p1_hist.append(t_r['zr_worst_p1'])

        rand_best_p2_hist.append(t_r['zr_best_p2'])
        rand_median_p2_hist.append(t_r['zr_median_p2'])
        rand_mean_p2_hist.append(t_r['zr_mean_p2'])
        rand_worst_p2_hist.append(t_r['zr_worst_p2'])

        write_log([loss_sum_hist, l2d_hist, lmmd_hist, ldet_hist, lmb_hist, lhm_hist,
                   lgen_hist, lcritic_hist],
                  ['loss_sum, l2d', 'lmmd', 'ldet', 'lmb', 'lhm', 'lgen', 'lcritic'],
                  path=c.result_dir, filename='train_losses')

        write_log([z0_p1_hist, rand_best_p1_hist, rand_median_p1_hist,
                   rand_mean_p1_hist, rand_worst_p1_hist, z0_p2_hist,
                   rand_best_p2_hist, rand_median_p2_hist, rand_mean_p2_hist, rand_worst_p2_hist],
                  ['protoI_z0, protoI_best, protoI_median', 'protoI_mean, protoI_worst, protoII_z0',
                   'protoII_best, protoII_median, protoII_mean, protoII_worst',
                   ], path=c.result_dir, filename='test_metrics')

        if i_epoch > 0:
            draw_loss(loss_sum_hist, 'loss_sum', path=c.result_dir)
            draw_loss(l2d_hist, 'l2d', path=c.result_dir)
            draw_loss(lmmd_hist, 'lmmd', path=c.result_dir)
            draw_loss(ldet_hist, 'ldet', path=c.result_dir)
            draw_loss(lmb_hist, 'lmb', path=c.result_dir)
            draw_loss(lhm_hist, 'lhm', path=c.result_dir)
            draw_loss(lgen_hist, 'lgen', path=c.result_dir)
            draw_loss(lcritic_hist, 'lcritic', path=c.result_dir)

            draw_mpjpe_all(z0_p1_hist, rand_best_p1_hist, rand_median_p1_hist,
                           rand_worst_p1_hist, 'Protocol-I', th=120, path=c.result_dir)
            draw_mpjpe_all(z0_p2_hist, rand_best_p2_hist, rand_median_p2_hist,
                           rand_worst_p2_hist, 'Protocol-II', th=80, path=c.result_dir)

        if i_epoch == c.n_epochs - 1:
            inn.save(c.result_dir + 'model_last_epoch.pt')

        # save every 30 epochs:
        if (i_epoch != 0 and (i_epoch % 30) == 0):
            inn.save(c.result_dir + 'model_epoch_' + str(i_epoch) + '.pt')

        lr_scheduler_critic.step()
        inn.lr_scheduler.step()
        print("time for epoch: %.2f sec, total time: %0.2f min\n" % (time() - t, (time() - t_start) / 60))
except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time() - t_start) / 60:.2f} minutes\n")
