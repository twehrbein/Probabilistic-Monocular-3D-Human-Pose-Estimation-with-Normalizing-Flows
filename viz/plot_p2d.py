# import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np


BONES17j = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

BONES16j_MPII = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8],
                 [8, 9], [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15]]

BONES16j_H36M = [[0, 1], [1, 2], [0, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9],
                 [7, 13], [13, 14], [14, 15], [7, 10], [10, 11], [11, 12]]


def plot2d_16j(pose, mpii_skeleton=True):
    # plots a single 2D pose (MPII skeleton)
    assert pose.size == 2 * 16
    fig = plt.figure()
    ax = fig.gca()

    pose = pose.flatten()
    x = pose[0:16]
    y = -pose[16:32]
    ax.scatter(x, y)

    if mpii_skeleton:
        B = BONES16j_MPII
    else:
        B = BONES16j_H36M
    for l in range(len(B)):
        ax.plot(x[B[l]], y[B[l]], lw=3)

    plt.axis('equal')
    plt.show()
    plt.close()


def plot2d_h36m_17j(pose):
    assert pose.size == 2 * 17
    fig = plt.figure()
    ax = fig.gca()

    pose = pose.flatten()
    x = pose[0:17]
    y = -pose[17:34]
    ax.scatter(x, y)

    B = BONES17j
    for l in range(len(B)):
        ax.plot(x[B[l]], y[B[l]], lw=3)

    plt.axis('equal')
    plt.show()
    plt.close()


if __name__ == '__main__':
    p2d = np.array([198.33682 , 172.83542 , 144.05945 , 198.4612  , 196.96811 ,
       228.16269 , 171.80464 , 151.5066  , 133.81554 , 134.0021  ,
        48.03241 ,  85.71535 , 110.1205  , 165.2981  , 175.95096 ,
       169.4131  , 226.3386  , 174.4512  , 151.28787 , 138.06049 ,
       106.07468 , 199.58871 , 145.42224 , 102.41721 ,  53.911667,
        25.472271, 129.07518 , 112.283165,  67.51121 ,  51.64825 ,
        93.26934 , 104.82332 ])
    print(p2d.shape)
    plot2d_16j(p2d)
