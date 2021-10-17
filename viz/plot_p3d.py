# import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D


BONES17j = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def get_limits(x, y, z):
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
    return Xb, Yb, Zb


def plot17j(pose, swap_axes=True):
    # plots a single 3d pose of shape (51,)
    assert pose.size == 3*17
    if swap_axes:
        pose = swap_pose_axes17j(pose.reshape(1, 3*17)).flatten()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    x = pose[0:17]
    y = pose[17:34]
    z = pose[34:51]
    ax.scatter(x, y, z)

    B = BONES17j
    for l in range(len(B)):
        ax.plot(x[B[l]], y[B[l]], z[B[l]], lw=3)

    # Create cubic bounding box to simulate equal aspect ratio
    Xb, Yb, Zb = get_limits(x, y, z)

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()
    plt.close()


def plot17j_multi(poses, swap_axes=True):
    # plots multiple 3d poses, shape: (-1, 51)
    assert poses.shape[1] == 3 * 17
    if swap_axes:
        poses = swap_pose_axes17j(poses)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=6, azim=-78)
    # first pose is colored red
    colors = ['r']
    while poses.shape[0] > len(colors):
        colors.append('g')

    for i in range(poses.shape[0]):
        pose = poses[i]
        x = pose[0:17]
        y = pose[17:34]
        z = pose[34:51]

        B = BONES17j
        for l in range(len(B)):
            ax.plot(x[B[l]], y[B[l]], z[B[l]], lw=2, c=colors[i])

        # Create cubic bounding box to simulate equal aspect ratio
        Xb, Yb, Zb = get_limits(x, y, z)

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    plt.show()
    plt.close()


def swap_pose_axes17j(poses_3d):
    # poses are in order [x: width, y:height, z:depth]
    # but matplotlib has axes order [x: width, y:depth, z:height]
    # poses: (-1, 3*17)
    assert len(poses_3d.shape) == 2
    y_save = poses_3d[:, 17:34].copy()
    poses_3d[:, 17:34] = poses_3d[:, 34:51] * -1  # otherwise its mirrored/flipped compared to 2d plot...
    poses_3d[:, 34:51] = y_save
    return poses_3d


if __name__ == '__main__':
    p3d = np.array([   0.0000, -131.9649,    9.4699,  126.8843,  131.9648,  140.2145,
         241.5268,  -92.8957, -177.4044, -175.2286, -157.8241,  -48.0008,
          22.5457,  111.6097, -290.4013, -399.6419, -565.8513,   -0.0000,
         -27.6830, -107.2068, -292.1895,   27.6830,  194.9382, -208.1003,
         199.1910,  435.7689,  430.3586,  544.0071,  430.6024,  250.5603,
          27.9209,  355.3474,  159.5432,   81.3813,   -0.0000,   16.7925,
         435.0356,  814.3091,  -16.7925,  399.3955,  537.7695,  -53.6768,
          -7.6089,   99.3799,   96.7646,  -60.0405, -256.3647, -316.8291,
         -24.4648, -184.6743,  -19.0801])
    plot17j(p3d)
