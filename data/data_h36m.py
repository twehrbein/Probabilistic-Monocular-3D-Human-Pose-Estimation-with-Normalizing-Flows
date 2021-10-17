import torch
from torch.utils.data import Dataset
import config as c
import numpy as np
from utils.data_utils import normalize_poses, H36M17j_TO_MPII, preprocess_gaussian_fits
import pickle

subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
subjects_test = ['S9', 'S11']
actions_all = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
               'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
subactions_all = ['0', '1']
cameras_all = ['54138969', '55011271', '58860488', '60457274']


class H36MDataset(Dataset):

    def __init__(self, pickle_file, quick_eval=False, quick_eval_stride=16,
                 train_set=True, actions=actions_all, hardsubset=False):

        if train_set:
            subjects_to_use = subjects_train
        else:
            subjects_to_use = subjects_test

        with open(pickle_file, 'rb') as handle:
            dataset = pickle.load(handle)

        poses_3d = None
        p2d_hrnet = None
        gauss_fits = None
        img_paths = []
        for subject in subjects_to_use:
            for action in actions:
                for subaction in subactions_all:
                    for cam in cameras_all:
                        if subject == 'S5' and subaction == '1' and action == 'Waiting' and cam == '55011271':
                            continue
                        if subject == 'S11' and subaction == '0' and action == 'Directions' and cam == '54138969':
                            continue  # S11 Directions.54138969 does not exist
                        if len(dataset[subject][action][subaction][cam]['imgpath']) < 1:
                            continue
                        img_paths.extend(dataset[subject][action][subaction][cam]['imgpath'])
                        if poses_3d is not None:
                            poses_3d = np.vstack((poses_3d, dataset[subject][action][subaction][cam]['3d_gt']))
                            p2d_hrnet = np.vstack(
                                (p2d_hrnet, dataset[subject][action][subaction][cam]['2d_hrnet']))
                            gauss_fits = np.vstack((gauss_fits, dataset[subject][action][subaction][cam]['gaussfit']))
                        else:
                            poses_3d = dataset[subject][action][subaction][cam]['3d_gt']
                            p2d_hrnet = dataset[subject][action][subaction][cam]['2d_hrnet']
                            gauss_fits = dataset[subject][action][subaction][cam]['gaussfit']

        if hardsubset:
            # use only examples where detector is uncertain
            # filter according to sqrt(eigenvalues) of Cov Matrix:, default is value of 2 (2px std deviation in train)
            fits = gauss_fits.reshape(-1, 16, 7)
            indices = np.logical_or(fits[:, :, 3] > 5, fits[:, :, 5] > 5)
            indices = np.any(indices, axis=1)
            print("number of poses with low confidence: ", np.count_nonzero(indices))
            print("percentage of poses with low confidence: ", np.count_nonzero(indices) / (len(indices)))

            poses_3d = poses_3d[indices]
            p2d_hrnet = p2d_hrnet[indices]
            gauss_fits = gauss_fits[indices]
            img_paths = [i for (i, v) in zip(img_paths, indices) if v]

        if quick_eval:
            poses_3d = poses_3d[::quick_eval_stride]
            p2d_hrnet = p2d_hrnet[::quick_eval_stride]
            gauss_fits = gauss_fits[::quick_eval_stride]
            img_paths = img_paths[::quick_eval_stride]

        img_paths = [c.h36m_img_base_loc + i for i in img_paths]

        # preprocess 3d gt poses
        p3d_gt = poses_3d.copy().reshape(-1, 3, 17)
        # root center gt poses
        p3d_gt -= p3d_gt[:, :, 0, None]
        # invert y and z axes:
        p3d_gt[:, 1:, :] *= -1
        p3d_gt = p3d_gt.reshape(-1, 3*17)
        self.p3d_gt = torch.from_numpy(p3d_gt).float().to(c.device)

        dims_to_use_3d = np.arange(1, 17)
        dims_to_use_3d = np.append(dims_to_use_3d, (dims_to_use_3d + 17, dims_to_use_3d + 34))

        # remove root joint
        poses_3d = poses_3d[:, dims_to_use_3d]
        poses_3d = normalize_poses(poses_3d)

        p2d_hrnet_unnorm = p2d_hrnet.copy()
        p2d_hrnet = normalize_poses(p2d_hrnet)

        gauss_fits = preprocess_gaussian_fits(gauss_fits)
        self.gauss_fits = torch.from_numpy(gauss_fits.reshape(-1, c.COND_JOINTS * c.COND_LENGTH)).float().to(c.device)

        self.p2d_hrnet = torch.from_numpy(p2d_hrnet).float().to(c.device)
        self.poses_3d = torch.from_numpy(poses_3d).float().to(c.device)
        self.p2d_hrnet_unnorm = torch.from_numpy(p2d_hrnet_unnorm).float().to(c.device)
        self.img_paths = img_paths

    def __len__(self):
        return self.poses_3d.size(0)

    def __getitem__(self, idx):
        return {'poses_3d': self.poses_3d[idx], 'p2d_hrnet': self.p2d_hrnet[idx],
                'gauss_fits': self.gauss_fits[idx], 'img_paths': self.img_paths[idx],
                'p3d_gt': self.p3d_gt[idx], 'p2d_hrnet_unnorm': self.p2d_hrnet_unnorm[idx]}
