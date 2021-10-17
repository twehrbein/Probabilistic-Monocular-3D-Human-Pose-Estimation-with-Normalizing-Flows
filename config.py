import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='original_model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VERBOSE = True
data_base_dir = 'data/'
result_base_dir = 'results/'

experiment_name = args.exp  # name of the experiment (log files saved in folder with this name)
result_dir = result_base_dir + experiment_name + '/'

m_name = 'model'
load_model_name = result_dir + m_name + '.pt'

train_file = data_base_dir + 'trainset_h36m.pickle'
test_file = data_base_dir + 'testset_h36m.pickle'
h36m_img_base_loc = ''

# Training parameters
batch_size = 64
n_epochs = 155
n_its_per_epoch = 482
clamp = 2.0  # clamping parameter of RealNVP coupling block

GEN_ITERS = 1
CRITIC_ITERS = 5
BEST_OF_M_LOSS = True
TOP_K = 5

lr = 1e-4

# width and height of heatmaps and images
hm_h = hm_w = 64
img_h = img_w = 256

# relative weighting of losses:
lambd_det = 4
lambd_mb = 4
lambd_mmd = 10
lambd_hm = 750

gt_sigma = 2.0  # sigma in px of the ground truth heatmaps for training the 2d detector
p3d_std = 0.010  # stddev in m corresponding to gt_sigma px stddev
# conversion factor to relate between covariance matrices from 3d pose hypotheses and from heatmaps:
hm_px_to_mm = (p3d_std / gt_sigma)**2

COND_JOINTS = 13  # number of heatmap fits per img (16 - (root + left&right hip joint))
COND_LENGTH = 6  # ampl, mu_x, mu_y, sigma_x, cov, sigma_y

N_JOINTS = 16

ndim_x = N_JOINTS * 3
ndim_y = N_JOINTS * 2
ndim_z = ndim_x - ndim_y
