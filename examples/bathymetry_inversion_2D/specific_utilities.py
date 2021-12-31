"""
Some specific utility functions, such as plotting.

"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable

import json

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_training_validation_losses(training_history_filename):
    """"

    ":param training_history_filename: str
        The JSON file name for the training history
    """

    # load the JSON file
    with open(training_history_filename) as json_file:
        history = json.load(json_file)

    loss = history['loss']
    val_loss = history['val_loss']
    lr = history['lr']

    #plot losses
    if (len(loss) !=0):
        plt.plot(np.arange(len(loss)),loss, 'k', label='training loss')

    if (len(val_loss) !=0):
        plt.plot(np.arange(len(val_loss)), val_loss, 'r', label='validation loss')

    plt.yscale('log')

    plt.tick_params(axis='both', which='major', labelsize=12)

    #plt.title('training loss and validation loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right', fontsize=14, frameon=False)
    plt.savefig("training_validtion_losses.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #plot learning rate
    plt.plot(np.arange(len(lr)), lr, 'k')

    #plt.yscale('log')

    plt.tick_params(axis='both', which='major', labelsize=12)

    #plt.title('training loss and validation loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Learning rate', fontsize=16)
    #plt.legend(loc='upper right', fontsize=14, frameon=False)
    plt.savefig("learning_rate_history.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_one_prediction(ID, b_uv_only, vel_WSE_pred, vel_WSE_test, zb_test):
    """
    Make plot for one single prediction

    :param ID: int
        case ID
    :param b_uv_only: bool
        whether the data only has (u,v) or (u,v,WSE)
    :param vel_WSE_pred: ndarray
    :param vel_WSE_test:
    :param zb_test:
    :return:
    """

    fig, axs = plt.subplots(4, 3, figsize=(3 * 10, 4 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.25, wspace=.01)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_test.numpy().shape[0]
    n_cols = zb_test.numpy().shape[1]

    # if WSE data is not in the dataset, create a WSE with zero values
    if b_uv_only:
        WSE_zero = np.zeros((n_rows, n_cols))

    # plot vel_x (simulated)
    cf_vel_x_sim = axs[0, 0].contourf(np.squeeze(vel_WSE_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 0].set_xlim([0, n_cols-1])
    axs[0, 0].set_ylim([0, n_rows-1])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title("Simulated $u$ from SRH-2D", fontsize=14)
    clb_vel_x_sim = fig.colorbar(cf_vel_x_sim, ticks=np.linspace(min, max, 7), ax=axs[0, 0])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)

    # plot vel_x (predicted from NN)
    cf_vel_x_pred = axs[0, 1].contourf(np.squeeze(vel_WSE_pred[:, :, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 1].set_xlim([0, n_cols-1])
    axs[0, 1].set_ylim([0, n_rows-1])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("Predicted $u$ by NN", fontsize=14)
    clb_vel_x_pred = fig.colorbar(cf_vel_x_pred, ticks=np.linspace(min, max, 7), ax=axs[0, 1])
    clb_vel_x_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_pred.ax.tick_params(labelsize=12)

    # plot diff(vel_x_test - vel_x_pred)
    vel_x_diff = np.squeeze(vel_WSE_pred[:, :, :, 0]) - np.squeeze(vel_WSE_test[:, :, 0])
    v_min = vel_x_diff.min()
    v_max = vel_x_diff.max()
    print("vel_x_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_vel_x_diff = axs[0, 2].contourf(vel_x_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  #cm: PRGn
    axs[0, 2].set_xlim([0, n_cols-1])
    axs[0, 2].set_ylim([0, n_rows-1])
    axs[0, 2].set_aspect('equal')
    axs[0, 2].set_title("$u$ differences", fontsize=14)
    clb_vel_x_diff = fig.colorbar(cf_vel_x_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[0, 2])
    clb_vel_x_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_x_diff.ax.tick_params(labelsize=12)

    # plot vel_y (simulated)
    cf_vel_y_sim = axs[1, 0].contourf(np.squeeze(vel_WSE_test[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 0].set_xlim([0, n_cols-1])
    axs[1, 0].set_ylim([0, n_rows-1])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("Simulated $v$ by SRH-2D", fontsize=14)
    clb_vel_y_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(min, max, 7), ax=axs[1, 0])
    clb_vel_y_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_sim.ax.tick_params(labelsize=12)

    # plot vel_y (predicted from NN)
    cf_vel_y_pred = axs[1, 1].contourf(np.squeeze(vel_WSE_pred[:, :, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 1].set_xlim([0, n_cols-1])
    axs[1, 1].set_ylim([0, n_rows-1])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("Predicted $v$ by NN", fontsize=14)
    clb_vel_y_pred = fig.colorbar(cf_vel_y_pred, ticks=np.linspace(min, max, 7), ax=axs[1, 1])
    clb_vel_y_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_pred.ax.tick_params(labelsize=12)

    # plot diff(vel_y_test - vel_y_pred)
    vel_y_diff = np.squeeze(vel_WSE_pred[:, :, :, 1]) - np.squeeze(vel_WSE_test[:, :, 1])
    v_min = vel_y_diff.min()
    v_max = vel_y_diff.max()
    print("vel_y_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_vel_y_diff = axs[1, 2].contourf(vel_y_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  # cm: PRGn
    axs[1, 2].set_xlim([0, n_cols-1])
    axs[1, 2].set_ylim([0, n_rows-1])
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_title("$v$ differences", fontsize=14)
    clb_vel_y_diff = fig.colorbar(cf_vel_y_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[1, 2])
    clb_vel_y_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_y_diff.ax.tick_params(labelsize=12)

    # plot WSE (simulated)
    if b_uv_only:
        cf_WSE_sim = axs[2, 0].contourf(WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_sim = axs[2, 0].contourf(np.squeeze(vel_WSE_test[:, :, 2]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[2, 0].set_xlim([0, n_cols-1])
    axs[2, 0].set_ylim([0, n_rows-1])
    axs[2, 0].set_aspect('equal')
    if b_uv_only:
        axs[2, 0].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 0].set_title("Simulated WSE from SRH-2D", fontsize=14)
    clb_WSE_sim = fig.colorbar(cf_WSE_sim, ticks=np.linspace(min, max, 7), ax=axs[2, 0])
    clb_WSE_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_sim.ax.tick_params(labelsize=12)

    # plot WSE (predicted from NN)
    if b_uv_only:
        cf_WSE_pred = axs[2, 1].contourf(WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_pred = axs[2, 1].contourf(np.squeeze(vel_WSE_pred[:, :, :, 2]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[2, 1].set_xlim([0, n_cols-1])
    axs[2, 1].set_ylim([0, n_rows-1])
    axs[2, 1].set_aspect('equal')
    if b_uv_only:
        axs[2, 1].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 1].set_title("Predicted WSE from NN", fontsize=14)
    clb_WSE_pred = fig.colorbar(cf_WSE_pred, ticks=np.linspace(min, max, 7), ax=axs[2, 1])
    clb_WSE_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_pred.ax.tick_params(labelsize=12)

    # plot diff(WSE_test - WSE_pred)
    if b_uv_only:
        WSE_diff = WSE_zero
    else:
        WSE_diff = np.squeeze(vel_WSE_pred[:, :, :, 2]) - np.squeeze(vel_WSE_test[:, :, 2])

    v_min = WSE_diff.min()
    v_max = WSE_diff.max()
    print("WSE_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)

    if b_uv_only:
        cf_WSE_diff = axs[2, 2].contourf(WSE_diff, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_diff = axs[2, 2].contourf(WSE_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)
    axs[2, 2].set_xlim([0, n_cols-1])
    axs[2, 2].set_ylim([0, n_rows-1])
    axs[2, 2].set_aspect('equal')

    if b_uv_only:
        axs[2, 2].set_title("WSE differences (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 2].set_title("WSE differences", fontsize=14)

    if b_uv_only:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(min, max, 7), ax=axs[2, 2])
    else:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[2, 2])
    clb_WSE_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_WSE_diff.ax.tick_params(labelsize=12)

    # plot zb
    cf_zb = axs[3, 0].contourf(np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[3, 0].set_xlim([0, n_cols-1])
    axs[3, 0].set_ylim([0, n_rows-1])
    axs[3, 0].set_aspect('equal')
    axs[3, 0].set_title("zb", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    cf_zb = axs[3, 1].contourf(np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[3, 1].set_xlim([0, n_cols-1])
    axs[3, 1].set_ylim([0, n_rows-1])
    axs[3, 1].set_aspect('equal')
    axs[3, 1].set_title("zb", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 1])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    cf_zb = axs[3, 2].contourf(np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[3, 2].set_xlim([0, n_cols-1])
    axs[3, 2].set_ylim([0, n_rows-1])
    axs[3, 2].set_aspect('equal')
    axs[3, 2].set_title("zb", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 2])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    # set labels
    plt.setp(axs[-1, :], xlabel='x')
    plt.setp(axs[:, 0], ylabel='y')

    plt.savefig("surrogate_prediction_" + str(ID.numpy()).zfill(4) + ".png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    #plt.show()


def plot_zb_inversion_result(zb_inverted_result_filename):
    """
    Plot the zb inversion result and compare with zb_truth
    1. plot zb comparison
    2. plot inversion loss history

    :param zb_inverted_result_filename:

    :return:
    """

    zbs = np.load(zb_inverted_result_filename)
    zb_truth = zbs['zb_truth']
    zb_inverted = zbs['zb_inverted']

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.01)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_inverted.shape[0]
    n_cols = zb_inverted.shape[1]

    # plot zb_truth
    cf_zb_truth = axs[0].contourf(zb_truth, levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[0].set_xlim([0, n_cols-1])
    axs[0].set_ylim([0, n_rows-1])
    axs[0].set_aspect('equal')
    axs[0].set_title("zb (truth)", fontsize=14)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(min, max, 7), cax=cax)
    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)

    # plot zb_inverted
    cf_zb_inverted = axs[1].contourf(zb_inverted, levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[1].set_xlim([0, n_cols-1])
    axs[1].set_ylim([0, n_rows-1])
    axs[1].set_aspect('equal')
    axs[1].set_title("zb (inverted)", fontsize=14)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(min, max, 7), cax=cax)
    clb_zb_inverted.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_inverted.ax.tick_params(labelsize=12)

    # plot diff(zb_truth - zb_inverted)
    zb_diff = zb_truth - zb_inverted
    v_min = zb_diff.min()
    v_max = zb_diff.max()
    print("zb_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_zb_diff = axs[2].contourf(zb_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.terrain)  # cm: PRGn
    axs[2].set_xlim([0, n_cols-1])
    axs[2].set_ylim([0, n_rows-1])
    axs[2].set_aspect('equal')
    axs[2].set_title("zb differences", fontsize=14)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(v_min, v_max, 7), cax=cax)
    clb_zb_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_zb_diff.ax.tick_params(labelsize=12)

    # set labels
    plt.setp(axs[2], xlabel='x')
    plt.setp(axs, ylabel='y')

    plt.savefig("zb_inversion.png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    plt.show()

def plot_zb_inversion_WSE_comparison(zb_inverted_result_filename):
    """
    Make plot for comparing zb, uvWSE, etc from inversion


    :return:
    """

    zb_inverted_results = np.load(zb_inverted_result_filename)
    zb_truth = zb_inverted_results['zb_truth']
    zb_inverted = zb_inverted_results['zb_inverted']
    vel_WSE_target = zb_inverted_results['uvWSE_target']
    vel_WSE_pred = zb_inverted_results['uvWSE_pred']

    fig, axs = plt.subplots(4, 3, figsize=(3 * 10, 4 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.25, wspace=.01)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_truth.shape[0]
    n_cols = zb_truth.shape[1]

    b_uv_only = True
    if vel_WSE_target.shape[-1] == 3:
        b_uv_only = False

    # if WSE data is not in the dataset, create a WSE with zero values
    if b_uv_only:
        WSE_zero = np.zeros((n_rows, n_cols))

    # plot vel_x (simulated from SRH-2D; target)
    cf_vel_x_sim = axs[0, 0].contourf(np.squeeze(vel_WSE_target[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 0].set_xlim([0, n_cols-1])
    axs[0, 0].set_ylim([0, n_rows-1])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title("Simulated $u$ from SRH-2D", fontsize=14)
    clb_vel_x_sim = fig.colorbar(cf_vel_x_sim, ticks=np.linspace(min, max, 7), ax=axs[0, 0])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)

    # plot vel_x (predicted from NN)
    cf_vel_x_pred = axs[0, 1].contourf(np.squeeze(vel_WSE_pred[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 1].set_xlim([0, n_cols-1])
    axs[0, 1].set_ylim([0, n_rows-1])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("Predicted $u$ by NN with inverted $z_b$", fontsize=14)
    clb_vel_x_pred = fig.colorbar(cf_vel_x_pred, ticks=np.linspace(min, max, 7), ax=axs[0, 1])
    clb_vel_x_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_pred.ax.tick_params(labelsize=12)

    # plot diff(vel_x_test - vel_x_pred)
    vel_x_diff = np.squeeze(vel_WSE_pred[:, :, 0]) - np.squeeze(vel_WSE_target[:, :, 0])
    v_min = vel_x_diff.min()
    v_max = vel_x_diff.max()
    print("vel_x_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_vel_x_diff = axs[0, 2].contourf(vel_x_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  #cm: PRGn
    axs[0, 2].set_xlim([0, n_cols-1])
    axs[0, 2].set_ylim([0, n_rows-1])
    axs[0, 2].set_aspect('equal')
    axs[0, 2].set_title("$u$ differences", fontsize=14)
    clb_vel_x_diff = fig.colorbar(cf_vel_x_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[0, 2])
    clb_vel_x_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_x_diff.ax.tick_params(labelsize=12)

    # plot vel_y (simulated)
    cf_vel_y_sim = axs[1, 0].contourf(np.squeeze(vel_WSE_target[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 0].set_xlim([0, n_cols-1])
    axs[1, 0].set_ylim([0, n_rows-1])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("Simulated $v$ by SRH-2D", fontsize=14)
    clb_vel_y_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(min, max, 7), ax=axs[1, 0])
    clb_vel_y_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_sim.ax.tick_params(labelsize=12)

    # plot vel_y (predicted from NN)
    cf_vel_y_pred = axs[1, 1].contourf(np.squeeze(vel_WSE_pred[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 1].set_xlim([0, n_cols-1])
    axs[1, 1].set_ylim([0, n_rows-1])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("Predicted $v$ by NN with inverted $z_b$", fontsize=14)
    clb_vel_y_pred = fig.colorbar(cf_vel_y_pred, ticks=np.linspace(min, max, 7), ax=axs[1, 1])
    clb_vel_y_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_pred.ax.tick_params(labelsize=12)

    # plot diff(vel_y_test - vel_y_pred)
    vel_y_diff = np.squeeze(vel_WSE_pred[:, :, 1]) - np.squeeze(vel_WSE_target[:, :, 1])
    v_min = vel_y_diff.min()
    v_max = vel_y_diff.max()
    print("vel_y_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_vel_y_diff = axs[1, 2].contourf(vel_y_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  # cm: PRGn
    axs[1, 2].set_xlim([0, n_cols-1])
    axs[1, 2].set_ylim([0, n_rows-1])
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_title("$v$ differences", fontsize=14)
    clb_vel_y_diff = fig.colorbar(cf_vel_y_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[1, 2])
    clb_vel_y_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_y_diff.ax.tick_params(labelsize=12)

    # plot WSE (simulated)
    if b_uv_only:
        cf_WSE_sim = axs[2, 0].contourf(WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_sim = axs[2, 0].contourf(np.squeeze(vel_WSE_target[:, :, 2]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[2, 0].set_xlim([0, n_cols-1])
    axs[2, 0].set_ylim([0, n_rows-1])
    axs[2, 0].set_aspect('equal')
    if b_uv_only:
        axs[2, 0].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 0].set_title("Simulated WSE from SRH-2D", fontsize=14)
    clb_WSE_sim = fig.colorbar(cf_WSE_sim, ticks=np.linspace(min, max, 7), ax=axs[2, 0])
    clb_WSE_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_sim.ax.tick_params(labelsize=12)

    # plot WSE (predicted from NN)
    if b_uv_only:
        cf_WSE_pred = axs[2, 1].contourf(WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_pred = axs[2, 1].contourf(np.squeeze(vel_WSE_pred[:, :, 2]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[2, 1].set_xlim([0, n_cols-1])
    axs[2, 1].set_ylim([0, n_rows-1])
    axs[2, 1].set_aspect('equal')
    if b_uv_only:
        axs[2, 1].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 1].set_title("Predicted WSE from NN with inverted $z_b$", fontsize=14)
    clb_WSE_pred = fig.colorbar(cf_WSE_pred, ticks=np.linspace(min, max, 7), ax=axs[2, 1])
    clb_WSE_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_pred.ax.tick_params(labelsize=12)

    # plot diff(WSE_test - WSE_pred)
    if b_uv_only:
        WSE_diff = WSE_zero
    else:
        WSE_diff = np.squeeze(vel_WSE_pred[:, :, 2]) - np.squeeze(vel_WSE_target[:, :, 2])

    v_min = WSE_diff.min()
    v_max = WSE_diff.max()
    print("WSE_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)

    if b_uv_only:
        cf_WSE_diff = axs[2, 2].contourf(WSE_diff, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_diff = axs[2, 2].contourf(WSE_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)
    axs[2, 2].set_xlim([0, n_cols-1])
    axs[2, 2].set_ylim([0, n_rows-1])
    axs[2, 2].set_aspect('equal')

    if b_uv_only:
        axs[2, 2].set_title("WSE differences (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 2].set_title("WSE differences", fontsize=14)

    if b_uv_only:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(min, max, 7), ax=axs[2, 2])
    else:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[2, 2])
    clb_WSE_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_WSE_diff.ax.tick_params(labelsize=12)

    # plot zb (truth)
    cf_zb = axs[3, 0].contourf(np.squeeze(zb_truth[:, :]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
    axs[3, 0].set_xlim([0, n_cols-1])
    axs[3, 0].set_ylim([0, n_rows-1])
    axs[3, 0].set_aspect('equal')
    axs[3, 0].set_title("$z_b$ (truth)", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    # plot zb (inverted)
    v_min = zb_inverted.min()
    v_max = zb_inverted.max()
    print("zb_inverted min, max =", v_min, v_max)
    local_levels = np.linspace(v_min, v_max, 51)
    cf_zb = axs[3, 1].contourf(np.squeeze(zb_inverted[:, :]), local_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.terrain)
    axs[3, 1].set_xlim([0, n_cols-1])
    axs[3, 1].set_ylim([0, n_rows-1])
    axs[3, 1].set_aspect('equal')
    axs[3, 1].set_title("$z_b$ (inverted)", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(v_min, v_max, 7), ax=axs[3, 1])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    # plot diff(zb_truth - zb_inverted)
    zb_diff = zb_truth - zb_inverted
    v_min = zb_diff.min()
    v_max = zb_diff.max()
    print("zb_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)
    cf_zb_diff = axs[3, 2].contourf(zb_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.terrain)  # cm: PRGn
    axs[3, 2].set_xlim([0, n_cols - 1])
    axs[3, 2].set_ylim([0, n_rows - 1])
    axs[3, 2].set_aspect('equal')
    axs[3, 2].set_title("$z_b$ differences", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(v_min, v_max, 7), ax=axs[3, 2])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)

    # set labels
    plt.setp(axs[-1, :], xlabel='x')
    plt.setp(axs[:, 0], ylabel='y')

    plt.savefig("inversion_comparison_all.png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    #plt.show()

    plt.close('all')


def plot_inversion_losses(inversion_history_filename):
    """"

    """

    loss = np.load(inversion_history_filename)['losses']

    plt.plot(np.arange(len(loss)), loss, 'k')

    plt.yscale('log')

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Mean squared error for inversion', fontsize=16)
    #plt.legend(loc='upper right', fontsize=14, frameon=False)
    plt.savefig("inversion_loss_history.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.close('all')



