import config as c
import numpy as np
import pickle
from pathlib import Path
import cdflib


# correct keys:
subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
subjects_test = ['S9', 'S11']
actions_all = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
               'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking']
subactions_all = ['0', '1']
cameras_all = ['54138969', '55011271', '58860488', '60457274']


def fix_dot_2(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "1.")
    return infos


def fix_dot_3(infos):
    if '3.' in infos:
        infos = infos.replace("3.", "1.")
    return infos


def fix_1_2(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "")
    return infos


def fix_2_3(infos):
    if '2.' in infos:
        infos = infos.replace("2.", "1.")
    if '3.' in infos:
        infos = infos.replace("3.", "")
    return infos


def fill_3d_gt_poses(dataset_file, train_set):
    """
    insert the 3d ground truth poses in the right format into the dataset pickle file
    """
    with open(dataset_file, 'rb') as handle:
        dataset = pickle.load(handle)

    if train_set:
        subjects = subjects_train
    else:
        subjects = subjects_test

    for subject in subjects:
        data_path = Path('data/') / subject / 'MyPoseFeatures' / 'D3_Positions_mono'
        files = list(sorted(data_path.glob('*.cdf')))
        assert len(files) > 0  # something is wrong with data paths...
        for file in files:
            cdf_file = cdflib.CDF(file)
            poses_3d = cdf_file[0].squeeze()
            assert poses_3d.shape[1] == 96
            # select 17 joints:
            joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
            poses_3d = poses_3d.reshape(-1, 32, 3)[:, joints]
            poses_3d = poses_3d.swapaxes(1, 2).reshape(-1, 3*17)
            # select every 4th frame
            indices = np.arange(3, len(poses_3d), 4)
            poses_3d = poses_3d[indices, :]

            # extract action, subaction and cam from filename
            filename = str(file.stem)
            if ' ' in filename:
                action, rest_info = filename.split(" ")
            else:
                action, rest_info = filename.split(".")

            # rename for consistency:
            # TakingPhoto -> Photo, WalkingDog -> WalkDog
            if action == 'TakingPhoto':
                action = 'Photo'
            if action == 'WalkingDog':
                action = 'WalkDog'

            # take care of inconsistent naming...
            if subject == 'S1':
                if action == 'Eating':
                    # S1 Eating (., 2)
                    rest_info = fix_dot_2(rest_info)
                if action == 'Sitting':
                    # S1 Sitting (1, 2)
                    rest_info = fix_1_2(rest_info)
                if action == 'SittingDown':
                    # S1 SittingDown (., 2)
                    rest_info = fix_dot_2(rest_info)

            if subject == 'S5':
                if action == 'Directions':
                    # S5 Directions (1, 2)
                    rest_info = fix_1_2(rest_info)
                if action == 'Discussion':
                    # S5 Discussion (2, 3)
                    rest_info = fix_2_3(rest_info)
                if action == 'Greeting':
                    # S5 Greeting (1, 2)
                    rest_info = fix_1_2(rest_info)
                if action == 'Photo':
                    # S5 Photo (., 2)
                    rest_info = fix_dot_2(rest_info)
                if action == 'Waiting':
                    # S5 Waiting (1, 2)
                    rest_info = fix_1_2(rest_info)

            if subject == 'S6':
                if action == 'Eating':
                    # S6 Eating (1, 2)
                    rest_info = fix_1_2(rest_info)
                if action == 'Posing':
                    # S6 Posing (., 2)
                    rest_info = fix_dot_2(rest_info)
                if action == 'Sitting':
                    # S6 Sitting (1,2)
                    rest_info = fix_1_2(rest_info)
                if action == 'Waiting':
                    # S6 Waiting (., 3)
                    rest_info = fix_dot_3(rest_info)

            if subject == 'S7':
                if action == 'Phoning':
                    # S7 Phoning (., 2)
                    rest_info = fix_dot_2(rest_info)
                if action == 'Waiting':
                    # S7 Waiting (1, 2)
                    rest_info = fix_1_2(rest_info)
                if action == 'Walking':
                    # S7 Walking (1, 2)
                    rest_info = fix_1_2(rest_info)

            if subject == 'S8':
                if action == 'WalkTogether':
                    # S8 WalkTogether (1, 2)
                    rest_info = fix_1_2(rest_info)

            if subject == 'S9':
                if action == 'Discussion':
                    # S9 discussion (1, 2)
                    rest_info = fix_1_2(rest_info)

            if subject == 'S11':
                if action == 'Discussion':
                    rest_info = fix_1_2(rest_info)
                if action == 'Greeting':
                    # S11 greeting (., 2)
                    rest_info = fix_dot_2(rest_info)
                if action == 'Phoning':
                    # S11 phoning (2,3)
                    rest_info = fix_2_3(rest_info)
                if action == 'Smoking':
                    # S11 smoking (., 2)
                    if '2.' in rest_info:
                        # replace 2. with .
                        rest_info = fix_dot_2(rest_info)

            assert rest_info[:2] == '1.' or '.' not in rest_info
            if '.' not in rest_info:
                subact = '0'
                cam = rest_info
            else:
                subact = '1'
                cam = rest_info.split('.')[-1]

            if subject == 'S5' and subact == '1' and action == 'Waiting' and cam == '55011271':
                continue
            if subject == 'S11' and subact == '0' and action == 'Directions' and cam == '54138969':
                continue

            used_frames = len(dataset[subject][action][subact][cam]['imgpath'])
            assert used_frames <= len(poses_3d)
            poses_3d = poses_3d[:used_frames]
            dataset[subject][action][subact][cam]['3d_gt'] = poses_3d

    if train_set:
        out_file = c.train_file
    else:
        out_file = c.test_file
    with open(out_file, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


fill_3d_gt_poses('data/testset_h36m_without_3d.pickle', train_set=False)
fill_3d_gt_poses('data/trainset_h36m_without_3d.pickle', train_set=True)
