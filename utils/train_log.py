import numpy as np
import matplotlib.pyplot as plt
import config as c


def draw_loss(loss_history, name, path=c.result_dir):
    plt.clf()
    plt.plot(loss_history)
    plt.title(name)
    plt.xlim(xmin=0)
    plt.savefig(path + "loss_" + name + '.png')
    plt.close()


def draw_mpjpe_all(z0_hist, best_hist, median_hist, worst_hist, name, th=-1, path=c.result_dir):
    plt.clf()
    plt.plot(worst_hist, 'k', label='worst')
    plt.plot(median_hist, 'm', label='median')
    plt.plot(best_hist, 'b', label='best')
    plt.plot(z0_hist, 'r', label='z_0')
    plt.legend(loc="upper right")
    plt.title(name)

    plt.xlim(xmin=0)
    if th != -1:
        plt.gca().set_ylim([None, th])
    plt.xlabel('epoch')
    plt.ylabel('mm')
    plt.savefig(path + "loss_" + name + '.png', dpi=200)
    plt.close()


def write_log(losses, names, path=c.result_dir, filename='log'):
    n_losses = len(losses)
    loss_len = len(losses[0])
    losses = np.stack([np.arange(loss_len)] + losses, axis=1)
    header = '\t\t'.join(['epoch'] + names)
    np.savetxt(path + filename + '.txt', losses, fmt=['%d'] + ['%.5e'] * n_losses,
               header=header, delimiter='\t')
