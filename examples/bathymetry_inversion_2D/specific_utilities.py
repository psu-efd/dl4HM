"""
Some specific utility functions, such as plotting.

"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dl4HM.data_loader.swe_2D_data_loader import SWEs2DDataLoader

from dl4HM.utils.config import process_config

import tensorflow as tf

import cv2

import json

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def scale_back(scaled_var, min, max):
    """
    Scale back the scaled_var with given min and max.

    Here we assume the scaled_var is within [-0.5, 0.5].

    :param scaled_var:
    :param min:
    :param max:
    :return:
    """

    return (scaled_var + 0.5)*(max-min) + min

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
        plt.plot(np.arange(len(val_loss)), val_loss, 'r--', label='validation loss')

    plt.yscale('log')

    plt.tick_params(axis='both', which='major', labelsize=12)

    #plt.title('training loss and validation loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xlim([0,100])
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


def plot_one_prediction(ID, b_uv_only, vel_WSE_pred, vel_WSE_test, zb_test, var_min_max, bPlot_zb=True, bPlot_dimless=False):
    """
    Make plot for one single prediction

    :param ID: int
        case ID
    :param b_uv_only: bool
        whether the data only has (u,v) or (u,v,WSE)
    :param vel_WSE_pred: ndarray
    :param vel_WSE_test:
    :param zb_test:
    :param var_min_max: dictionary
        Variable min and max
    :param bPlot_zb: bool
        whether to plot zb. Default is True.
    :param bPlot_dimless: bool
        whether to plot the variables in dimensionless form. Default is false.
    :return:
    """

    if bPlot_zb:
        fig, axs = plt.subplots(4, 3, figsize=(3 * 10, 4 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    else:
        fig, axs = plt.subplots(3, 3, figsize=(3 * 8, 3 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')

    fig.subplots_adjust(hspace=.25, wspace=.01)

    n_rows = zb_test.numpy().shape[0]
    n_cols = zb_test.numpy().shape[1]

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    vel_x_min = var_min_max['vel_x_min']
    vel_x_max = var_min_max['vel_x_max']
    vel_y_min = var_min_max['vel_y_min']
    vel_y_max = var_min_max['vel_y_max']
    WSE_min = var_min_max['WSE_min']
    WSE_max = var_min_max['WSE_max']

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    # if WSE data is not in the dataset, create a WSE with zero values
    if b_uv_only:
        WSE_zero = np.zeros((n_rows, n_cols))

    # plot vel_x (simulated)
    local_levels = scale_back(levels, vel_x_min, vel_x_max)
    if bPlot_dimless:
        cf_vel_x_sim = axs[0, 0].contourf(X, Y, np.squeeze(vel_WSE_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_vel_x_sim = axs[0, 0].contourf(X, Y, scale_back(np.squeeze(vel_WSE_test[:, :, 0]), vel_x_min, vel_x_max), local_levels,
                                      vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)
    axs[0, 0].set_xlim([xl, xh])
    axs[0, 0].set_ylim([yl, yh])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)
    axs[0, 0].set_title("Simulated $u$ from SRH-2D", fontsize=14)
    if bPlot_dimless:
        clb_vel_x_sim = fig.colorbar(cf_vel_x_sim, ticks=np.linspace(min, max, 7), ax=axs[0, 0])
    else:
        clb_vel_x_sim = fig.colorbar(cf_vel_x_sim, ticks=np.linspace(local_levels.min(), local_levels.max(), 7), ax=axs[0, 0])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_x_sim.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot vel_x (predicted from NN)
    if bPlot_dimless:
        cf_vel_x_pred = axs[0, 1].contourf(X, Y, np.squeeze(vel_WSE_pred[:, :, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_vel_x_pred = axs[0, 1].contourf(X, Y, scale_back(np.squeeze(vel_WSE_pred[:, :, :, 0]), vel_x_min, vel_x_max), local_levels,
                                           vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)

    axs[0, 1].set_xlim([xl, xh])
    axs[0, 1].set_ylim([yl, yh])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("Predicted $u$ by CNN surrogate", fontsize=14)
    if bPlot_dimless:
        clb_vel_x_pred = fig.colorbar(cf_vel_x_pred, ticks=np.linspace(min, max, 7), ax=axs[0, 1])
    else:
        clb_vel_x_pred = fig.colorbar(cf_vel_x_pred, ticks=np.linspace(local_levels.min(), local_levels.max(), 7), ax=axs[0, 1])

    clb_vel_x_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_pred.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_x_pred.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot diff(vel_x_test - vel_x_pred)
    if bPlot_dimless:
        vel_x_diff = np.squeeze(vel_WSE_pred[:, :, :, 0]) - np.squeeze(vel_WSE_test[:, :, 0])
    else:
        vel_x_diff = scale_back(np.squeeze(vel_WSE_pred[:, :, :, 0]), vel_x_min, vel_x_max)\
                     - scale_back(np.squeeze(vel_WSE_test[:, :, 0]), vel_x_min, vel_x_max)

    v_min = vel_x_diff.min()
    v_max = vel_x_diff.max()
    print("vel_x_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)

    #calculate RMSE
    rmse_vel_x = np.sqrt((vel_x_diff ** 2).mean())

    cf_vel_x_diff = axs[0, 2].contourf(vel_x_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  #cm: PRGn
    axs[0, 2].set_xlim([xl, xh])
    axs[0, 2].set_ylim([yl, yh])
    axs[0, 2].set_aspect('equal')
    if bPlot_dimless:
        axs[0, 2].set_title("$u$ differences, $e_m$ = {0:.4f}".format(rmse_vel_x), fontsize=16)
    else:
        axs[0, 2].set_title("$u$ differences, $e_m$ = {0:.4f} m/s".format(rmse_vel_x), fontsize=16)

    clb_vel_x_diff = fig.colorbar(cf_vel_x_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[0, 2])
    clb_vel_x_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_x_diff.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_x_diff.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot vel_y (simulated)
    local_levels = scale_back(levels, vel_y_min, vel_y_max)

    if bPlot_dimless:
        cf_vel_y_sim = axs[1, 0].contourf(X, Y, np.squeeze(vel_WSE_test[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_vel_y_sim = axs[1, 0].contourf(X, Y, scale_back(np.squeeze(vel_WSE_test[:, :, 1]), vel_y_min, vel_y_max),
                                          local_levels,
                                          vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)

    axs[1, 0].set_xlim([xl, xh])
    axs[1, 0].set_ylim([yl, yh])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='y', labelsize=14)
    axs[1, 0].set_title("Simulated $v$ by SRH-2D", fontsize=14)

    if bPlot_dimless:
        clb_vel_y_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(min, max, 7), ax=axs[1, 0])
    else:
        clb_vel_y_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(local_levels.min(), local_levels.max(), 7),
                                     ax=axs[1, 0])

    clb_vel_y_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_sim.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_y_sim.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot vel_y (predicted from NN)
    if bPlot_dimless:
        cf_vel_y_pred = axs[1, 1].contourf(X, Y, np.squeeze(vel_WSE_pred[:, :, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_vel_y_pred = axs[1, 1].contourf(X, Y, scale_back(np.squeeze(vel_WSE_pred[:, :, :, 1]), vel_y_min, vel_y_max),
                                          local_levels,
                                          vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)

    axs[1, 1].set_xlim([xl, xh])
    axs[1, 1].set_ylim([yl, yh])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("Predicted $v$ by CNN surrogate", fontsize=14)
    if bPlot_dimless:
        clb_vel_y_pred = fig.colorbar(cf_vel_y_pred, ticks=np.linspace(min, max, 7), ax=axs[1, 1])
    else:
        clb_vel_y_pred = fig.colorbar(cf_vel_y_pred, ticks=np.linspace(local_levels.min(), local_levels.max(), 7),
                                     ax=axs[1, 1])

    clb_vel_y_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y_pred.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_y_pred.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot diff(vel_y_test - vel_y_pred)
    vel_y_diff = np.squeeze(vel_WSE_pred[:, :, :, 1]) - np.squeeze(vel_WSE_test[:, :, 1])
    if bPlot_dimless:
        vel_y_diff = np.squeeze(vel_WSE_pred[:, :, :, 1]) - np.squeeze(vel_WSE_test[:, :, 1])
    else:
        vel_y_diff = scale_back(np.squeeze(vel_WSE_pred[:, :, :, 1]), vel_y_min, vel_y_max) \
                     - scale_back(np.squeeze(vel_WSE_test[:, :, 1]), vel_y_min, vel_y_max)

    v_min = vel_y_diff.min()
    v_max = vel_y_diff.max()
    print("vel_y_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)

    # calculate RMSE
    rmse_vel_y = np.sqrt((vel_y_diff ** 2).mean())

    cf_vel_y_diff = axs[1, 2].contourf(vel_y_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)  # cm: PRGn
    axs[1, 2].set_xlim([xl, xh])
    axs[1, 2].set_ylim([yl, yh])
    axs[1, 2].set_aspect('equal')

    if bPlot_dimless:
        axs[1, 2].set_title("$v$ differences, $e_m$ = {0:.4f}".format(rmse_vel_y), fontsize=16)
    else:
        axs[1, 2].set_title("$v$ differences, $e_m$ = {0:.4f} m/s".format(rmse_vel_y), fontsize=16)

    clb_vel_y_diff = fig.colorbar(cf_vel_y_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[1, 2])
    clb_vel_y_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_vel_y_diff.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_vel_y_diff.ax.set_title("(m/s)", loc='center', fontsize=12)

    # plot WSE (simulated)
    if b_uv_only:
        cf_WSE_sim = axs[2, 0].contourf(X, Y, WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        local_levels = scale_back(levels, WSE_min, WSE_max)
        if bPlot_dimless:
            cf_WSE_sim = axs[2, 0].contourf(X, Y, np.squeeze(vel_WSE_test[:, :, 2]), levels, vmin=min, vmax=max,
                                              cmap=plt.cm.jet)
        else:
            cf_WSE_sim = axs[2, 0].contourf(X, Y, scale_back(np.squeeze(vel_WSE_test[:, :, 2]), WSE_min, WSE_max),
                                              local_levels,
                                              vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)

    axs[2, 0].set_xlim([xl, xh])
    axs[2, 0].set_ylim([yl, yh])
    axs[2, 0].set_aspect('equal')
    axs[2, 0].set_xlabel('$x$ (m)', fontsize = 16)
    axs[2, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[2, 0].tick_params(axis='x', labelsize=14)
    axs[2, 0].tick_params(axis='y', labelsize=14)
    if b_uv_only:
        axs[2, 0].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 0].set_title("Simulated WSE from SRH-2D", fontsize=14)

    if bPlot_dimless:
        clb_WSE_sim = fig.colorbar(cf_WSE_sim, ticks=np.linspace(min, max, 7), ax=axs[2, 0])
    else:
        clb_WSE_sim = fig.colorbar(cf_WSE_sim, ticks=np.linspace(local_levels.min(), local_levels.max(), 7),
                                     ax=axs[2, 0])

    clb_WSE_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_sim.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_WSE_sim.ax.set_title("(m)", loc='center', fontsize=12)

    # plot WSE (predicted from NN)
    if b_uv_only:
        cf_WSE_pred = axs[2, 1].contourf(X, Y, WSE_zero, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:

        if bPlot_dimless:
            cf_WSE_pred = axs[2, 1].contourf(X, Y, np.squeeze(vel_WSE_pred[:, :, :, 2]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
        else:
            cf_WSE_pred = axs[2, 1].contourf(X, Y, scale_back(np.squeeze(vel_WSE_pred[:, :, :, 2]), WSE_min, WSE_max),
                                              local_levels,
                                              vmin=local_levels.min(), vmax=local_levels.max(), cmap=plt.cm.jet)

    axs[2, 1].set_xlim([xl, xh])
    axs[2, 1].set_ylim([yl, yh])
    axs[2, 1].set_aspect('equal')
    axs[2, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[2, 1].tick_params(axis='x', labelsize=14)
    if b_uv_only:
        axs[2, 1].set_title("WSE (zeros, not in dataset)", fontsize=14)
    else:
        axs[2, 1].set_title("Predicted WSE from CNN surrogate", fontsize=14)

    if bPlot_dimless:
        clb_WSE_pred = fig.colorbar(cf_WSE_pred, ticks=np.linspace(min, max, 7), ax=axs[2, 1])
    else:
        clb_WSE_pred = fig.colorbar(cf_WSE_pred, ticks=np.linspace(local_levels.min(), local_levels.max(), 7),
                                     ax=axs[2, 1])

    clb_WSE_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_WSE_pred.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_WSE_pred.ax.set_title("(m)", loc='center', fontsize=12)

    # plot diff(WSE_test - WSE_pred)
    if b_uv_only:
        WSE_diff = WSE_zero
    else:
        if bPlot_dimless:
            WSE_diff = np.squeeze(vel_WSE_pred[:, :, :, 2]) - np.squeeze(vel_WSE_test[:, :, 2])
        else:
            WSE_diff = scale_back(np.squeeze(vel_WSE_pred[:, :, :, 2]), WSE_min, WSE_max) \
                         - scale_back(np.squeeze(vel_WSE_test[:, :, 2]), WSE_min, WSE_max)

    v_min = WSE_diff.min()
    v_max = WSE_diff.max()
    print("WSE_diff min, max =", v_min, v_max)
    diff_levels = np.linspace(v_min, v_max, 51)

    # calculate RMSE
    rmse_WSE = np.sqrt((WSE_diff ** 2).mean())

    if b_uv_only:
        cf_WSE_diff = axs[2, 2].contourf(X, Y, WSE_diff, levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    else:
        cf_WSE_diff = axs[2, 2].contourf(X, Y, WSE_diff, diff_levels, vmin=v_min, vmax=v_max, cmap=plt.cm.jet)
    axs[2, 2].set_xlim([xl, xh])
    axs[2, 2].set_ylim([yl, yh])
    axs[2, 2].set_aspect('equal')
    axs[2, 2].set_xlabel('$x$ (m)', fontsize=16)
    axs[2, 2].tick_params(axis='x', labelsize=14)

    if b_uv_only:
        axs[2, 2].set_title("WSE differences (zeros, not in dataset)", fontsize=14)
    else:

        if bPlot_dimless:
            axs[2, 2].set_title("WSE differences, $e_m$ = {0:.4f}".format(rmse_WSE), fontsize=16)
        else:
            axs[2, 2].set_title("WSE differences, $e_m$ = {0:.4f} m".format(rmse_WSE), fontsize=16)

    if b_uv_only:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(min, max, 7), ax=axs[2, 2])
    else:
        clb_WSE_diff = fig.colorbar(cf_WSE_diff, ticks=np.linspace(v_min, v_max, 7), ax=axs[2, 2])
    clb_WSE_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.4f'))
    clb_WSE_diff.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_WSE_diff.ax.set_title("(m)", loc='center', fontsize=12)

    if bPlot_zb:
        # plot zb
        cf_zb = axs[3, 0].contourf(X, Y, np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
        axs[3, 0].set_xlim([xl, xh])
        axs[3, 0].set_ylim([yl, yh])
        axs[3, 0].set_aspect('equal')
        axs[3, 0].set_title("zb", fontsize=14)
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 0])
        clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb.ax.tick_params(labelsize=12)

        cf_zb = axs[3, 1].contourf(X, Y, np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
        axs[3, 1].set_xlim([xl, xh])
        axs[3, 1].set_ylim([yl, yh])
        axs[3, 1].set_aspect('equal')
        axs[3, 1].set_title("zb", fontsize=14)
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 1])
        clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb.ax.tick_params(labelsize=12)

        cf_zb = axs[3, 2].contourf(X, Y, np.squeeze(zb_test[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
        axs[3, 2].set_xlim([xl, xh])
        axs[3, 2].set_ylim([yl, yh])
        axs[3, 2].set_aspect('equal')
        axs[3, 2].set_title("zb", fontsize=14)
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), ax=axs[3, 2])
        clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb.ax.tick_params(labelsize=12)

    plt.savefig("surrogate_prediction_" + str(ID.numpy()).zfill(4) + ".png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    #plt.show()


def plot_zb_inversion_result(zb_inverted_result_filename, config_filename, bPlot_dimless=False):
    """
    Plot the zb inversion result and compare with zb_truth
    1. plot zb comparison
    2. plot inversion loss history

    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    zbs = np.load(zb_inverted_result_filename)
    zb_init_all = zbs['zb_init']
    zb_truth = zbs['zb_truth']
    zb_inverted_all = zbs['zb_inverted']

    #mean zb from all samples
    zb_inverted_mean = np.mean(zb_inverted_all, axis=-1)

    #number of samples
    nSamples = zb_inverted_all.shape[-1]

    #hack: to only plot some
    nSamples = 3

    #three columns for subplots: init/truth, zb_inverted/mean_zb, diff
    fig, axs = plt.subplots(nSamples+1, 3, figsize=(20, (nSamples+1)*2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.15)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_inverted_all.shape[0]
    n_cols = zb_inverted_all.shape[1]

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']
    local_levels = scale_back(levels, zb_min, zb_max)

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    # plot zb_truth
    if bPlot_dimless:
        cf_zb_truth = axs[0, 0].contourf(X, Y, zb_truth, levels, vmin=min, vmax=max, cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb_truth = axs[0, 0].contourf(X, Y, scale_back(zb_truth, zb_min, zb_max), local_levels,
                                         vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain, extend='both')

    axs[0,0].set_xlim([xl, xh])
    axs[0,0].set_ylim([yl, yh])
    axs[0,0].set_aspect('equal')
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)
    axs[0,0].set_title("$z_b$ (truth)", fontsize=16)
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="3%", pad=0.1)

    if bPlot_dimless:
        clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(min, max, 7), cax=cax)
    else:
        clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)

    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb_truth.ax.set_title("(m)", loc='center', fontsize=12)

    # plot zb_inverted_mean
    if bPlot_dimless:
        cf_zb_inverted = axs[0,1].contourf(X, Y, zb_inverted_mean, levels, vmin=min, vmax=max, cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb_inverted = axs[0, 1].contourf(X, Y, scale_back(zb_inverted_mean, zb_min, zb_max), local_levels,
                                            vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain,
                                            extend='both')

    axs[0,1].set_xlim([xl, xh])
    axs[0,1].set_ylim([yl, yh])
    axs[0,1].set_aspect('equal')
    axs[0,1].set_title("Inverted $z_b$ (mean)", fontsize=16)
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="3%", pad=0.1)

    if bPlot_dimless:
        clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(min, max, 7), cax=cax)
    else:
        clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)

    clb_zb_inverted.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_inverted.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb_inverted.ax.set_title("(m)", loc='center', fontsize=12)

    # plot diff(zb_truth - zb_inverted)
    if bPlot_dimless:
        zb_diff = zb_truth - zb_inverted_mean
    else:
        zb_diff = scale_back(zb_truth, zb_min, zb_max) - scale_back(zb_inverted_mean, zb_min, zb_max)

    v_min = zb_diff.min()
    v_max = zb_diff.max()
    print("zb_diff min, max =", v_min, v_max)
    if bPlot_dimless:
        diff_levels = np.linspace(min, max, 51)
    else:
        #diff_levels = np.linspace(v_min, v_max, 51)
        diff_levels = np.linspace(zb_min, zb_max, 51)

    # calculate RMSE
    rmse_zb = np.sqrt((zb_diff ** 2).mean())

    if bPlot_dimless:
        cf_zb_diff = axs[0,2].contourf(X, Y, zb_diff, diff_levels, vmin=min, vmax=max, cmap=plt.cm.terrain, extend='both')  # cm: PRGn
    else:
        cf_zb_diff = axs[0, 2].contourf(X, Y, zb_diff, diff_levels, vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain,
                                        extend='both')  # cm: PRGn

    axs[0,2].set_xlim([xl, xh])
    axs[0,2].set_ylim([yl, yh])
    axs[0,2].set_aspect('equal')

    if bPlot_dimless:
        axs[0,2].set_title("$z_b$ differences, $e_m$ = {0:.4f}".format(rmse_zb), fontsize=16)
    else:
        axs[0, 2].set_title("$z_b$ differences, $e_m$ = {0:.4f} m".format(rmse_zb), fontsize=16)

    divider = make_axes_locatable(axs[0,2])
    cax = divider.append_axes("right", size="3%", pad=0.1)

    if bPlot_dimless:
        clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(min, max, 5), cax=cax)
    else:
        clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)

    clb_zb_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_diff.ax.tick_params(labelsize=12)
    #clb_zb_diff.ax.set_yticklabels(["{:4.3f}".format(i) for i in (np.linspace(min, max, 5))])  # add the labels
    if not bPlot_dimless:
        clb_zb_diff.ax.set_title("(m)", loc='center', fontsize=12)

    #plot each of the samples
    for i in range(nSamples):
        # plot zb_init
        if bPlot_dimless:
            cf_zb_truth = axs[i+1, 0].contourf(X, Y, zb_init_all[:,:,i], levels, vmin=min, vmax=max, cmap=plt.cm.terrain, extend='both')
        else:
            cf_zb_truth = axs[i + 1, 0].contourf(X, Y, scale_back(zb_init_all[:, :, i], zb_min, zb_max), local_levels, vmin=zb_min, vmax=zb_max,
                                                 cmap=plt.cm.terrain, extend='both')

        axs[i+1, 0].set_xlim([xl, xh])
        axs[i+1, 0].set_ylim([yl, yh])
        axs[i+1, 0].set_aspect('equal')
        axs[i+1, 0].set_ylabel('$y$ (m)', fontsize=16)
        axs[i+1, 0].tick_params(axis='y', labelsize=14)

        if i == (nSamples-1): #last row; add x-label
            axs[i + 1, 0].set_xlabel('$x$ (m)', fontsize=16)
            axs[i + 1, 0].tick_params(axis='x', labelsize=14)

        axs[i+1, 0].set_title("Initial $z_b$: "+str(i), fontsize=16)
        divider = make_axes_locatable(axs[i+1, 0])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        if bPlot_dimless:
            clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(min, max, 7), cax=cax)
        else:
            clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)
        clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb_truth.ax.tick_params(labelsize=12)
        if not bPlot_dimless:
            clb_zb_truth.ax.set_title("(m)", loc='center', fontsize=12)

        # plot zb_inverted
        v_min = zb_inverted_all[:,:,i].min()
        v_max = zb_inverted_all[:,:,i].max()
        print("zb_inverted min, max =", v_min, v_max)

        if bPlot_dimless:
            local_levels = np.linspace(v_min, v_max, 51)
            cf_zb_inverted = axs[i+1, 1].contourf(X, Y, zb_inverted_all[:,:,i], local_levels, vmin=min, vmax=max,
                                              cmap=plt.cm.terrain, extend='both')
        else:
            cf_zb_inverted = axs[i + 1, 1].contourf(X, Y, scale_back(zb_inverted_all[:, :, i], zb_min, zb_max),
                                                    local_levels, vmin=zb_min, vmax=zb_max,
                                                    cmap=plt.cm.terrain, extend='both')

        axs[i+1, 1].set_xlim([xl, xh])
        axs[i+1, 1].set_ylim([yl, yh])
        axs[i+1, 1].set_aspect('equal')

        if i == (nSamples-1): #last row; add x-label
            axs[i + 1, 1].set_xlabel('$x$ (m)', fontsize=16)
            axs[i + 1, 1].tick_params(axis='x', labelsize=14)

        axs[i+1, 1].set_title("Inverted $z_b$: "+str(i), fontsize=16)
        divider = make_axes_locatable(axs[i+1, 1])
        cax = divider.append_axes("right", size="3%", pad=0.1)

        if bPlot_dimless:
            clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(min, max, 7), cax=cax)
        else:
            clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)


        clb_zb_inverted.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb_inverted.ax.tick_params(labelsize=12)
        if not bPlot_dimless:
            clb_zb_inverted.ax.set_title("(m)", loc='center', fontsize=12)

        # plot diff(zb_truth - zb_inverted)
        if bPlot_dimless:
            zb_diff = zb_truth - zb_inverted_all[:,:,i]
        else:
            zb_diff = scale_back(zb_truth, zb_min, zb_max) - scale_back(zb_inverted_all[:,:,i], zb_min, zb_max)

        v_min = zb_diff.min()
        v_max = zb_diff.max()
        print("zb_diff min, max =", v_min, v_max)
        diff_levels = np.linspace(min, max, 51)

        # calculate RMSE
        rmse_zb = np.sqrt((zb_diff ** 2).mean())

        if bPlot_dimless:
            cf_zb_diff = axs[i+1, 2].contourf(X, Y, zb_diff, diff_levels, vmin=min, vmax=max, cmap=plt.cm.terrain, extend='both')  # cm: PRGn
        else:
            cf_zb_diff = axs[i + 1, 2].contourf(X, Y, zb_diff, local_levels, vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain,
                                                extend='both')  # cm: PRGn

        axs[i+1, 2].set_xlim([xl, xh])
        axs[i+1, 2].set_ylim([yl, yh])
        axs[i+1, 2].set_aspect('equal')

        if i == (nSamples-1): #last row; add x-label
            axs[i + 1, 2].set_xlabel('$x$ (m)', fontsize=16)
            axs[i + 1, 2].tick_params(axis='x', labelsize=14)

        if bPlot_dimless:
            axs[i+1, 2].set_title("$z_b$ differences, $e_m$ = {0:.4f}".format(rmse_zb), fontsize=16)
        else:
            axs[i + 1, 2].set_title("$z_b$ differences, $e_m$ = {0:.4f} m".format(rmse_zb), fontsize=16)

        divider = make_axes_locatable(axs[i+1, 2])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        if bPlot_dimless:
            clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(min, max, 5), cax=cax)
        else:
            clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)

        clb_zb_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb_diff.ax.tick_params(labelsize=12)
        if not bPlot_dimless:
            clb_zb_diff.ax.set_title("(m)", loc='center', fontsize=12)

    plt.savefig("zb_inversion.png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    plt.show()


def plot_zb_inversion_result_profiles(zb_inverted_result_filename, config_filename):
    """
    Plot the zb inversion result profiles and compare with zb_truth


    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    zbs = np.load(zb_inverted_result_filename)
    zb_init_all = zbs['zb_init']
    zb_truth = zbs['zb_truth']
    zb_inverted_all = zbs['zb_inverted']

    #mean zb from all samples
    zb_inverted_mean = np.mean(zb_inverted_all, axis=-1)

    #number of samples
    nSamples = zb_inverted_all.shape[-1]

    #hack: to only plot some
    #nSamples = 4

    #
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=False, sharey=False, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.15)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_inverted_all.shape[0]
    n_cols = zb_inverted_all.shape[1]

    #middle row and column
    n_row_middle = int(n_rows/2)
    n_col_middle = int(n_cols/2)

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    # plot longitudinal profiles

    axs[0].set_xlim([xl, xh])
    axs[0].set_ylim([-0.5, 0.5])
    #axs[].set_aspect('equal')
    axs[0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].set_ylabel('$z_b$ (m)', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].set_yticks(np.linspace(-0.5,0.5,5))

    # plot cross-sectional profiles
    axs[1].set_xlim([yl, yh])
    axs[1].set_ylim([-0.5, 0.5])
    # axs[1].set_aspect('equal')
    axs[1].set_xlabel('$y$ (m)', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].set_ylabel('$z_b$ (m)', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].set_yticks(np.linspace(-0.5, 0.5, 5))

    #add caption
    #axs[0].text(0.5, -0.35, "(a)", size=16, ha="center", transform=axs[0].transAxes)  #bottom center
    #axs[1].text(0.5, -0.35, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    axs[0].text(-0.1, 1.05, "(a)", size=16, ha="center", transform=axs[0].transAxes)   #upper left
    axs[1].text(-0.1, 1.05, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    #plot each of the samples + 1 truth + 1 mean
    for i in range(nSamples+2):
        if i == 0: #plot truth
            axs[0].plot(x, scale_back(zb_truth[n_row_middle, :], zb_min, zb_max), 'k', linewidth=2, label='$z_b$ (truth)')
            axs[1].plot(y, scale_back(zb_truth[:, n_col_middle], zb_min, zb_max), 'k', linewidth=2, label='$z_b$ (truth)')
        elif i == 1: #plot mean
            axs[0].plot(x, scale_back(zb_inverted_mean[n_row_middle, :], zb_min, zb_max), 'r', linewidth=2, label='$\hat{z}_b$ (inverted mean)')
            axs[1].plot(y, scale_back(zb_inverted_mean[:, n_col_middle], zb_min, zb_max), 'r', linewidth=2, label='$\hat{z}_b$ (inverted mean)')
        else:   #each sample
            axs[0].plot(x, scale_back(zb_inverted_all[n_row_middle, :, i-2], zb_min, zb_max), 'r', linewidth=1, alpha=0.1)
            axs[1].plot(y, scale_back(zb_inverted_all[:, n_col_middle, i-2], zb_min, zb_max), 'r', linewidth=1, alpha=0.1)

    axs[0].legend(loc='upper right', fontsize=14, frameon=False)
    axs[1].legend(loc='upper right', fontsize=14, frameon=False)

    plt.savefig("zb_inversion_profiles.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_zb_inversion_result_profiles_regularization_effects(config_filename):
    """
    Plot the zb inversion result profiles and compare with zb_truth.

    This is to check the regularization effects.
    1. both regularization
    2. slope only, no value
    3. value only, no slope
    4. no regularization


    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    # results with both regularizations
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_1/zb_inverted_result.npz')
    zb_truth = zbs['zb_truth']
    zb_inverted_all_all_reg = zbs['zb_inverted']

    # results with no value regularization
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_2/zb_inverted_result.npz')
    zb_inverted_all_no_value_reg = zbs['zb_inverted']

    # results with no slope regularization
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_3/zb_inverted_result.npz')
    zb_inverted_all_no_slope_reg = zbs['zb_inverted']

    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_4/zb_inverted_result.npz')
    zb_inverted_all_no_reg = zbs['zb_inverted']

    #mean zb from all samples
    zb_inverted_all_all_reg_mean = np.mean(zb_inverted_all_all_reg, axis=-1)
    zb_inverted_all_no_value_reg_mean = np.mean(zb_inverted_all_no_value_reg, axis=-1)
    zb_inverted_all_no_slope_reg_mean = np.mean(zb_inverted_all_no_slope_reg, axis=-1)
    zb_inverted_all_no_reg_mean = np.mean(zb_inverted_all_no_reg, axis=-1)

    #number of samples
    nSamples = zb_inverted_all_all_reg.shape[-1]

    #plot
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), sharex='col', sharey='row', facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.2)

    min = -0.5
    max = 0.5

    n_rows = zb_inverted_all_all_reg.shape[0]
    n_cols = zb_inverted_all_all_reg.shape[1]

    #middle row and column
    n_row_middle = int(n_rows/2)
    n_col_middle = int(n_cols/2)

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)

    # plot mean
    #   Longitudinal
    zb_truth_long, = axs[0,0].plot(x, (zb_truth[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2, label='$z_b$ (truth)')
    zb_both_long, = axs[0,0].plot(x, (zb_inverted_all_all_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'r', linewidth=1,
                label='$z_b$ (inverted mean, both regularizations)')
    zb_slope_long, = axs[0, 0].plot(x, (zb_inverted_all_no_value_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'b--', linewidth=1,
                   label='$z_b$ (inverted mean, slope regularization only)')
    zb_value_long, = axs[0, 0].plot(x, (zb_inverted_all_no_slope_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'm-.', linewidth=1,
                   label='$z_b$ (inverted mean, value regularization only)')
    zb_no_long, = axs[0, 0].plot(x, (zb_inverted_all_no_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'g:', linewidth=1, alpha=0.4,
                   label='$z_b$ (inverted mean, no regularization)')

    zb_long = [zb_both_long, zb_slope_long, zb_value_long, zb_no_long]

    axs[0,0].set_xlim([xl, xh])
    axs[0,0].set_ylim([-0.5, 0.5])
    #axs[].set_aspect('equal')
    #axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0,0].tick_params(axis='x', labelsize=12)
    axs[0,0].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[0,0].tick_params(axis='y', labelsize=12)
    axs[0,0].set_yticks(np.linspace(-0.5,0.5,5))

    legend1 = axs[0,0].legend(zb_long, ['$\hat{z}_b$ (both)', '$\hat{z}_b$ (slope only)', '$\hat{z}_b$ (value only)', '$\hat{z}_b$ (none)'], loc='upper right', ncol=2,
                    bbox_to_anchor=(1.0, 1.45), bbox_transform=axs[0,0].transAxes, fontsize=10, labelspacing=0.2, columnspacing=1, frameon=False)
    legend2 = axs[0,0].legend([zb_truth_long], ['$z_b$ (truth)'], loc='upper right', fontsize=10, frameon=False)
    axs[0,0].add_artist(legend1)
    axs[0,0].add_artist(legend2)

    #   cross-sectional
    zb_truth_cs, = axs[0, 1].plot(y, (zb_truth[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2, label='$z_b$ (truth)')
    zb_both_cs, = axs[0, 1].plot(y, (zb_inverted_all_all_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'r', linewidth=1,
                   label='$\hat{z}_b$ (both)')
    zb_slope_cs, = axs[0, 1].plot(y, (zb_inverted_all_no_value_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'b--', linewidth=1,
                   label='$\hat{z}_b$ (slope only)')
    zb_value_cs, = axs[0, 1].plot(y, (zb_inverted_all_no_slope_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'm-.', linewidth=1,
                   label='$\hat{z}_b$ (value only)')
    zb_no_cs, = axs[0, 1].plot(y, (zb_inverted_all_no_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'g:', linewidth=1, alpha=0.4,
                   label='$\hat{z}_b$ (none)')

    zb_cs = [zb_both_cs, zb_slope_cs, zb_value_cs, zb_no_cs]

    axs[0,1].set_xlim([yl, yh])
    axs[0,1].set_ylim([-0.5, 0.5])
    # axs[1].set_aspect('equal')
    #axs[0,1].set_xlabel('$y$ (m)', fontsize=16)
    axs[0,1].tick_params(axis='x', labelsize=12)
    #axs[0,1].set_ylabel('$z_b$ (m)', fontsize=16)
    axs[0,1].tick_params(axis='y', labelsize=12)
    axs[0, 1].set_yticks(np.linspace(-0.5, 0.5, 5))

    legend1 = axs[0,1].legend(zb_cs, ['$\hat{z}_b$ (both)', '$\hat{z}_b$ (slope only)', '$\hat{z}_b$ (value only)', '$\hat{z}_b$ (none)'], loc='upper right', ncol=2,
                    bbox_to_anchor=(1.1, 1.45), bbox_transform=axs[0,1].transAxes, fontsize=10, labelspacing=0.2, columnspacing=1, frameon=False)
    legend2 = axs[0,1].legend([zb_truth_cs], ['$z_b$ (truth)'], loc='upper right', fontsize=10, frameon=False)
    axs[0,1].add_artist(legend1)
    axs[0,1].add_artist(legend2)

    # slope only
    #   Longitudinal
    axs[1, 0].set_xlim([xl, xh])
    axs[1, 0].set_ylim([-0.5, 0.5])
    # axs[].set_aspect('equal')
    # axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=12)
    axs[1, 0].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[1, 0].tick_params(axis='y', labelsize=12)
    axs[1, 0].set_title('slope regularization only', fontsize=14)
    axs[1, 0].set_yticks(np.linspace(-0.5, 0.5, 5))

    #    Longitudinal truth and mean
    zb_truth_long, = axs[1,0].plot(x, (zb_truth[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2, label='$z_b$ (truth)')
    zb_slope_long, = axs[1, 0].plot(x, (zb_inverted_all_no_value_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'r', linewidth=2,
                   label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[1, 0].plot(x, (zb_inverted_all_no_value_reg[n_row_middle, :, i]+0.5) * (zb_max - zb_min) + zb_min, 'r', linewidth=1, alpha=0.1)

    axs[1, 0].legend(loc='best', fontsize=10, frameon=False)

    #   Cross-section
    axs[1, 1].set_xlim([xl, xh])
    axs[1, 1].set_ylim([-0.5, 0.5])
    # axs[].set_aspect('equal')
    # axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=12)
    #axs[1, 1].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[1, 1].tick_params(axis='y', labelsize=12)
    axs[1, 1].set_title('slope regularization only', fontsize=14)
    axs[1, 1].set_yticks(np.linspace(-0.5, 0.5, 5))

    #    cross section truth and mean
    zb_truth_long, = axs[1, 1].plot(y, (zb_truth[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2,
                                    label='$z_b$ (truth)')
    zb_slope_long, = axs[1, 1].plot(y, (zb_inverted_all_no_value_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min,
                                    'r', linewidth=2,
                                    label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[1, 1].plot(y, (zb_inverted_all_no_value_reg[:, n_col_middle, i]+0.5) * (zb_max - zb_min) + zb_min, 'r',
                       linewidth=1, alpha=0.1)

    axs[1, 1].legend(loc='upper right', fontsize=10, frameon=False)

    # value only
    #   Longitudinal
    axs[2, 0].set_xlim([xl, xh])
    axs[2, 0].set_ylim([-0.5, 0.5])
    # axs[].set_aspect('equal')
    # axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
    axs[2, 0].tick_params(axis='x', labelsize=12)
    axs[2, 0].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[2, 0].tick_params(axis='y', labelsize=12)
    axs[2, 0].set_title('value regularization only', fontsize=14)
    axs[2, 0].set_yticks(np.linspace(-0.5, 0.5, 5))

    #    Longitudinal truth and mean
    zb_truth_long, = axs[2, 0].plot(x, (zb_truth[n_row_middle, :] * (zb_max - zb_min)+0.5) + zb_min, 'k', linewidth=2,
                                    label='$z_b$ (truth)')
    zb_slope_long, = axs[2, 0].plot(x, (zb_inverted_all_no_slope_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min,
                                    'r', linewidth=2,
                                    label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[2, 0].plot(x, (zb_inverted_all_no_slope_reg[n_row_middle, :, i]+0.5) * (zb_max - zb_min) + zb_min, 'r',
                       linewidth=1, alpha=0.1)

    axs[2, 0].legend(loc='upper center', ncol=2, columnspacing=1, fontsize=10, frameon=False)

    #   Cross-section
    axs[2, 1].set_xlim([xl, xh])
    axs[2, 1].set_ylim([-0.5, 0.5])
    # axs[].set_aspect('equal')
    # axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
    axs[2, 1].tick_params(axis='x', labelsize=12)
    # axs[1, 1].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[2, 1].tick_params(axis='y', labelsize=12)
    axs[2, 1].set_title('value regularization only', fontsize=14)
    axs[2, 1].set_yticks(np.linspace(-0.5, 0.5, 5))

    #    cross section truth and mean
    zb_truth_long, = axs[2, 1].plot(y, (zb_truth[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2,
                                    label='$z_b$ (truth)')
    zb_slope_long, = axs[2, 1].plot(y, (zb_inverted_all_no_slope_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min,
                                    'r', linewidth=2,
                                    label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[2, 1].plot(y, (zb_inverted_all_no_slope_reg[:, n_col_middle, i]+0.5) * (zb_max - zb_min) + zb_min, 'r',
                       linewidth=1, alpha=0.05)

    axs[2, 1].legend(loc='upper right', fontsize=10, frameon=False)

    # none
    #   Longitudinal
    axs[3, 0].set_xlim([xl, xh])
    axs[3, 0].set_ylim([-5, 5])
    # axs[].set_aspect('equal')
    axs[3, 0].set_xlabel('$x$ (m)', fontsize=14)
    axs[3, 0].tick_params(axis='x', labelsize=12)
    axs[3, 0].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[3, 0].tick_params(axis='y', labelsize=12)
    axs[3, 0].set_title('no regularization', fontsize=14)
    axs[3, 0].set_yticks(np.linspace(-5, 5, 5))

    #    Longitudinal truth and mean
    zb_truth_long, = axs[3, 0].plot(x, (zb_truth[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min, 'k', linewidth=2,
                                    label='$z_b$ (truth)')
    zb_slope_long, = axs[3, 0].plot(x, (zb_inverted_all_no_reg_mean[n_row_middle, :]+0.5) * (zb_max - zb_min) + zb_min,
                                    'r', linewidth=2,
                                    label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[3, 0].plot(x, (zb_inverted_all_no_reg[n_row_middle, :, i]+0.5) * (zb_max - zb_min) + zb_min, 'r',
                       linewidth=1, alpha=0.1)

    axs[3, 0].legend(loc='upper center', ncol=2, columnspacing=1, fontsize=10, frameon=False)

    #   Cross-section
    axs[3, 1].set_xlim([yl, yh])
    axs[3, 1].set_ylim([-5, 5])
    # axs[].set_aspect('equal')
    axs[3, 1].set_xlabel('$x$ (m)', fontsize=14)
    axs[3, 1].tick_params(axis='x', labelsize=12)
    # axs[1, 1].set_ylabel('$z_b$ (m)', fontsize=14)
    axs[3, 1].tick_params(axis='y', labelsize=12)
    axs[3, 1].set_title('no regularization', fontsize=14)
    axs[3, 1].set_yticks(np.linspace(-5, 5, 5))

    #    cross section truth and mean
    zb_truth_long, = axs[3, 1].plot(y, (zb_truth[:, n_col_middle] * (zb_max - zb_min)+0.5) + zb_min, 'k', linewidth=2,
                                    label='$z_b$ (truth)')
    zb_slope_long, = axs[3, 1].plot(y, (zb_inverted_all_no_reg_mean[:, n_col_middle]+0.5) * (zb_max - zb_min) + zb_min,
                                    'r', linewidth=2,
                                    label='$\hat{z}_b$ (mean)')

    #   Longitudinal samples
    for i in range(nSamples):
        axs[3, 1].plot(y, (zb_inverted_all_no_reg[:, n_col_middle, i]+0.5) * (zb_max - zb_min) + zb_min, 'r',
                       linewidth=1, alpha=0.1)

    axs[3, 1].legend(loc='upper right', ncol=2, columnspacing=1, fontsize=10, frameon=False)

    #add caption
    #axs[0].text(0.5, -0.35, "(a)", size=16, ha="center", transform=axs[0].transAxes)  #bottom center
    #axs[1].text(0.5, -0.35, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    axs[0,0].text(-0.25, 1.1, "(a)", size=16, ha="center", transform=axs[0,0].transAxes)   #upper left
    axs[0,1].text(-0.1, 1.1, "(b)", size=16, ha="center", transform=axs[0,1].transAxes)
    axs[1,0].text(-0.25, 1.0, "(c)", size=16, ha="center", transform=axs[1,0].transAxes)
    axs[1,1].text(-0.1, 1.0, "(d)", size=16, ha="center", transform=axs[1,1].transAxes)
    axs[2,0].text(-0.25, 1.0, "(e)", size=16, ha="center", transform=axs[2,0].transAxes)
    axs[2,1].text(-0.1, 1.0, "(f)", size=16, ha="center", transform=axs[2,1].transAxes)
    axs[3,0].text(-0.25, 1.0, "(g)", size=16, ha="center", transform=axs[3,0].transAxes)
    axs[3,1].text(-0.1, 1.0, "(h)", size=16, ha="center", transform=axs[3,1].transAxes)

    #plot each of the samples + 1 truth + 1 mean





    #plt.savefig("zb_inversion_profiles.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig("zb_inversion_profiles_regularization_effects.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_zb_inversion_regularization_effects(config_filename, bPlot_dimless=False):
    """
    Plot the zb inversion results with different regularizations
    1. with both value and slope regularizations
    2. with slope, but not value regularization
    3. with value, but not slope regularization
    4. with no regularization


    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    #results with both regularizations
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_1/zb_inverted_result.npz')
    zb_truth = zbs['zb_truth']
    zb_inverted_all_all_reg = zbs['zb_inverted']

    #results with no value regularization
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_2/zb_inverted_result.npz')
    zb_inverted_all_no_value_reg = zbs['zb_inverted']

    #results with no slope regularization
    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_3/zb_inverted_result.npz')
    zb_inverted_all_no_slope_reg = zbs['zb_inverted']

    zbs = np.load('inversion_hyperparameter_cases/inversion_with_uv_only/case_4/zb_inverted_result.npz')
    zb_inverted_all_no_reg = zbs['zb_inverted']

    #calculate, report, and save the RMSE of each result
    rmse_all_reg = np.sqrt(((zb_inverted_all_all_reg[:,:,0] - zb_truth) ** 2).mean())
    rmse_no_value_reg = np.sqrt(((zb_inverted_all_no_value_reg[:, :, 0] - zb_truth) ** 2).mean())
    rmse_no_slope_reg = np.sqrt(((zb_inverted_all_no_slope_reg[:, :, 0] - zb_truth) ** 2).mean())
    rmse_no_reg = np.sqrt(((zb_inverted_all_no_reg[:, :, 0] - zb_truth) ** 2).mean())

    #plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 5.5), sharex=False, sharey=False, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.05, wspace=.3)

    n_rows = zb_inverted_all_all_reg.shape[0]
    n_cols = zb_inverted_all_all_reg.shape[1]

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)
    local_levels = scale_back(levels, zb_min, zb_max)

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    # plot zb_inverted with both regularization
    if bPlot_dimless:
        cf_zb = axs[0, 0].contourf(X, Y, np.squeeze(zb_inverted_all_all_reg[:, :, 0]), levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb = axs[0, 0].contourf(X, Y, scale_back(np.squeeze(zb_inverted_all_all_reg[:, :, 0]),zb_min, zb_max),
                                   local_levels, vmin=zb_min, vmax=zb_max,
                                   cmap=plt.cm.terrain, extend='both')

    axs[0, 0].set_xlim([xl, xh])
    axs[0, 0].set_ylim([yl, yh])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='x', labelsize=14)
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)
    if bPlot_dimless:
        axs[0, 0].set_title("both regularizations, $e_m$ = {0:.2f}".format(rmse_all_reg), fontsize=16)
    else:
        axs[0, 0].set_title("both regularizations, $e_m$ = {0:.2f} m".format(rmse_all_reg), fontsize=16)

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if bPlot_dimless:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), cax=cax)
    else:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb.ax.set_title("(m)", loc='center', fontsize=12)

    # plot zb_inverted with slope, but not value regularization
    v_min = zb_inverted_all_no_value_reg[:, :, 0].min()
    v_max = zb_inverted_all_no_value_reg[:, :, 0].max()
    print("zb_inverted min, max =", v_min, v_max)
    #local_levels = np.linspace(v_min, v_max, 51)

    if bPlot_dimless:
        cf_zb = axs[0, 1].contourf(X, Y, np.squeeze(zb_inverted_all_no_value_reg[:, :, 0]), local_levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb = axs[0, 1].contourf(X, Y, scale_back(np.squeeze(zb_inverted_all_no_value_reg[:, :, 0]), zb_min, zb_max), local_levels, vmin=zb_min,  vmax=zb_max,
                               cmap=plt.cm.terrain, extend='both')

    axs[0, 1].set_xlim([xl, xh])
    axs[0, 1].set_ylim([yl, yh])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[0, 1].tick_params(axis='x', labelsize=14)
    axs[0, 1].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 1].tick_params(axis='y', labelsize=14)

    if bPlot_dimless:
        axs[0, 1].set_title("slope regularization only, $e_m$ = {0:.2f}".format(rmse_no_value_reg), fontsize=16)
    else:
        axs[0, 1].set_title("slope regularization only, $e_m$ = {0:.2f} m".format(rmse_no_value_reg), fontsize=16)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if bPlot_dimless:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(v_min, v_max, 7), cax=cax)
    else:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb.ax.set_title("(m)", loc='center', fontsize=12)

    # plot zb_inverted with value, but not slope regularization
    v_min = zb_inverted_all_no_slope_reg[:, :, 0].min()
    v_max = zb_inverted_all_no_slope_reg[:, :, 0].max()
    print("zb_inverted min, max =", v_min, v_max)
    #local_levels = np.linspace(v_min, v_max, 51)

    if bPlot_dimless:
        cf_zb = axs[1, 0].contourf(X, Y, np.squeeze(zb_inverted_all_no_slope_reg[:, :, 0]), local_levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb = axs[1, 0].contourf(X, Y, np.squeeze(scale_back(zb_inverted_all_no_slope_reg[:, :, 0],zb_min, zb_max)),
                                   local_levels, vmin=zb_min, vmax=zb_max,
                                   cmap=plt.cm.terrain, extend='both')
    axs[1, 0].set_xlim([xl, xh])
    axs[1, 0].set_ylim([yl, yh])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='y', labelsize=14)
    if bPlot_dimless:
        axs[1, 0].set_title("value regularization only, $e_m$ = {0:.2f}".format(rmse_no_slope_reg), fontsize=16)
    else:
        axs[1, 0].set_title("value regularization only, $e_m$ = {0:.2f} m".format(rmse_no_slope_reg), fontsize=16)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if bPlot_dimless:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(v_min, v_max, 7), cax=cax)
    else:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb.ax.set_title("(m)", loc='center', fontsize=12)

    # plot zb_inverted with no regularization
    v_min = zb_inverted_all_no_reg[:, :, 0].min()
    v_max = zb_inverted_all_no_reg[:, :, 0].max()
    print("zb_inverted min, max =", v_min, v_max)
    #local_levels = np.linspace(v_min, v_max, 51)

    if bPlot_dimless:
        cf_zb = axs[1, 1].contourf(X, Y, np.squeeze(zb_inverted_all_no_reg[:, :, 0]), local_levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    else:
        cf_zb = axs[1, 1].contourf(X, Y, scale_back(np.squeeze(zb_inverted_all_no_reg[:, :, 0]), zb_min, zb_max),
                                   local_levels, vmin=zb_min, vmax=zb_max,
                                   cmap=plt.cm.terrain, extend='both')

    axs[1, 1].set_xlim([xl, xh])
    axs[1, 1].set_ylim([yl, yh])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=14)
    axs[1, 1].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='y', labelsize=14)
    if bPlot_dimless:
        axs[1, 1].set_title("no regularization, $e_m$ = {0:.2f}".format(rmse_no_reg), fontsize=16)
    else:
        axs[1, 1].set_title("no regularization, $e_m$ = {0:.2f} m".format(rmse_no_reg), fontsize=16)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if bPlot_dimless:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(v_min, v_max, 7), cax=cax)
    else:
        clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    if not bPlot_dimless:
        clb_zb.ax.set_title("(m)", loc='center', fontsize=12)

    #add caption
    #axs[0].text(0.5, -0.35, "(a)", size=16, ha="center", transform=axs[0].transAxes)  #bottom center
    #axs[1].text(0.5, -0.35, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    axs[0,0].text(-0.1, 1.05, "(a)", size=16, ha="center", transform=axs[0,0].transAxes)   #upper left
    axs[0,1].text(-0.1, 1.05, "(b)", size=16, ha="center", transform=axs[0,1].transAxes)
    axs[1,0].text(-0.1, 1.05, "(c)", size=16, ha="center", transform=axs[1,0].transAxes)
    axs[1,1].text(-0.1, 1.05, "(d)", size=16, ha="center", transform=axs[1,1].transAxes)

    plt.savefig("zb_inversion_contours_regularization_effects.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_zb_inversion_loss_components_cnn_structure():
    """
    Plot the zb inversion results with different options on what to include in the loss calculation
    1. (u,v) from NN_(u,v,WSE)
    2. (u,v,WSE) from NN_(u,v,WSE)
    3. (u,v) from NN_(u,v)

    :return:
    """

    with open('varialbes_min_max.json') as json_file:
        var_min_max = json.load(json_file)

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']

    #results: (u,v) out of NN_(u,v,WSE)
    zbs = np.load('uvWSE_cases/sampled_64_256/inversion_hyperparameter_cases/inversion_with_uv_only/case_1/zb_inverted_result.npz')
    zb_truth = (zbs['zb_truth']+0.5)*(zb_max-zb_min) + zb_min
    zb_inverted_all_uv_uvWSE = (zbs['zb_inverted']+0.5)*(zb_max-zb_min) + zb_min

    #results: (u,v,WSE) out of NN_(u,v,WSE)
    zbs = np.load('uvWSE_cases/sampled_64_256/inversion_hyperparameter_cases/inversion_with_uvWSE/case_1/zb_inverted_result.npz')
    zb_inverted_all_uvWSE_uvWSE = (zbs['zb_inverted']+0.5)*(zb_max-zb_min) + zb_min

    #results with no slope regularization
    zbs = np.load('uv_only_cases/sampled_64_256/inversion_hyperparameter_cases/case_1/zb_inverted_result.npz')
    zb_inverted_all_uv_uv = (zbs['zb_inverted']+0.5)*(zb_max-zb_min) + zb_min

    #calculate the mean
    zb_inverted_all_uv_uvWSE_mean = np.mean(zb_inverted_all_uv_uvWSE, axis=-1)
    zb_inverted_all_uvWSE_uvWSE_mean = np.mean(zb_inverted_all_uvWSE_uvWSE, axis=-1)
    zb_inverted_all_uv_uv_mean = np.mean(zb_inverted_all_uv_uv, axis=-1)

    #calculate, report, and save the RMSE of each result
    rmse_all_uv_uvWSE = np.sqrt(((zb_inverted_all_uv_uvWSE_mean[:,:] - zb_truth) ** 2).mean())
    rmse_all_uvWSE_uvWSE = np.sqrt(((zb_inverted_all_uvWSE_uvWSE_mean[:, :] - zb_truth) ** 2).mean())
    rmse_all_uv_uv = np.sqrt(((zb_inverted_all_uv_uv_mean[:, :] - zb_truth) ** 2).mean())


    #plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=False, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.1, wspace=.3)

    #min = -0.5
    #max = 0.5
    min = zb_min
    max = zb_max
    levels = np.linspace(min, max, 51)

    n_rows = zb_inverted_all_uv_uvWSE.shape[0]
    n_cols = zb_inverted_all_uv_uvWSE.shape[1]

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    # plot zb_inverted: uv_uvWSE
    cf_zb = axs[0].contourf(X, Y, np.squeeze(zb_inverted_all_uv_uvWSE_mean[:, :]), levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    axs[0].set_xlim([xl, xh])
    axs[0].set_ylim([yl, yh])
    axs[0].set_aspect('equal')
    #axs[0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].set_title("Inversion using $(u,v)$ from NN$_{{(u,v,WSE)}}$, $e_m$ = {0:.3f} m".format(rmse_all_uv_uvWSE), fontsize=16)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    clb_zb.ax.set_title('(m)', fontsize=12)

    # plot zb_inverted: uvWSE_uvWSE
    cf_zb = axs[1].contourf(X, Y, np.squeeze(zb_inverted_all_uvWSE_uvWSE_mean[:, :]), levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    axs[1].set_xlim([xl, xh])
    axs[1].set_ylim([yl, yh])
    axs[1].set_aspect('equal')
    #axs[1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].set_ylabel('$y$ (m)', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].set_title("Inversion using $(u,v,WSE)$ from NN$_{{(u,v,WSE)}}$, $e_m$ = {0:.3f} m".format(rmse_all_uvWSE_uvWSE), fontsize=16)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    clb_zb.ax.set_title('(m)', fontsize=12)

    # plot zb_inverted: uv_uv
    cf_zb = axs[2].contourf(X, Y, np.squeeze(zb_inverted_all_uv_uv_mean[:, :]), levels, vmin=min, vmax=max,
                               cmap=plt.cm.terrain, extend='both')
    axs[2].set_xlim([xl, xh])
    axs[2].set_ylim([yl, yh])
    axs[2].set_aspect('equal')
    axs[2].set_xlabel('$x$ (m)', fontsize=16)
    axs[2].tick_params(axis='x', labelsize=14)
    axs[2].set_ylabel('$y$ (m)', fontsize=16)
    axs[2].tick_params(axis='y', labelsize=14)
    axs[2].set_title("Inversion using $(u,v)$ from NN$_{{(u,v)}}$, $e_m$ = {0:.3f} m".format(rmse_all_uv_uv), fontsize=16)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(min, max, 7), cax=cax)
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    clb_zb.ax.set_title('(m)', fontsize=12)

    #add caption
    #axs[0].text(0.5, -0.35, "(a)", size=16, ha="center", transform=axs[0].transAxes)  #bottom center
    #axs[1].text(0.5, -0.35, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    axs[0].text(-0.05, 1.05, "(a)", size=18, ha="center", transform=axs[0].transAxes)   #upper left
    axs[1].text(-0.05, 1.05, "(b)", size=18, ha="center", transform=axs[1].transAxes)
    axs[2].text(-0.05, 1.05, "(c)", size=18, ha="center", transform=axs[2].transAxes)

    plt.savefig("zb_inversion_contours_loss_comp_cnn_structure.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_zb_inversion_result_profiles_uv_uncertainty(zb_inverted_result_filename, config_filename):
    """
    Plot the zb inversion result profiles and compare with zb_truth

    This is for the uv uncertainty.


    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    zb_min = var_min_max['zb_min']
    zb_max = var_min_max['zb_max']

    zbs = np.load(zb_inverted_result_filename)
    zb_init_all = zbs['zb_init']
    zb_truth = zbs['zb_truth']
    zb_inverted_all = zbs['zb_inverted']

    zb_truth = scale_back(zb_truth, zb_min, zb_max)
    zb_inverted_all = scale_back(zb_inverted_all, zb_min, zb_max)

    #mean zb from all samples
    zb_inverted_mean = np.mean(zb_inverted_all, axis=-1)

    #plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=False, sharey=False, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.15)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = zb_inverted_all.shape[0]
    n_cols = zb_inverted_all.shape[1]

    #middle row and column
    n_row_middle = int(n_rows/2)
    n_col_middle = int(n_cols/2)

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    # plot longitudinal profiles

    axs[0].set_xlim([xl, xh])
    axs[0].set_ylim([-0.5, 0.5])
    #axs[].set_aspect('equal')
    axs[0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].set_ylabel('$z_b$ (m)', fontsize=16)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].set_yticks(np.linspace(-0.5,0.5,5))

    # plot cross-sectional profiles
    axs[1].set_xlim([yl, yh])
    axs[1].set_ylim([-0.5, 0.5])
    # axs[1].set_aspect('equal')
    axs[1].set_xlabel('$y$ (m)', fontsize=16)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].set_ylabel('$z_b$ (m)', fontsize=16)
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].set_yticks(np.linspace(-0.5, 0.5, 5))

    #add caption
    axs[0].text(-0.1, 1.05, "(a)", size=16, ha="center", transform=axs[0].transAxes)   #upper left
    axs[1].text(-0.1, 1.05, "(b)", size=16, ha="center", transform=axs[1].transAxes)

    #plot the zb_truth
    axs[0].plot(x, zb_truth[n_row_middle, :] , 'k', linewidth=2, label='$z_b$ (truth)')
    axs[1].plot(y, zb_truth[:, n_col_middle] , 'k', linewidth=2, label='$z_b$ (truth)')

    #plot mean
    axs[0].plot(x, zb_inverted_mean[n_row_middle, :], 'r', linewidth=2,
                label='$\hat{z}_b$ (inverted mean)')
    axs[1].plot(y, zb_inverted_mean[:, n_col_middle], 'r', linewidth=2,
                label='$\hat{z}_b$ (inverted mean)')

    #plot 95% uncertainty
    #calculate 95 percentile uncertainty band (between 2.5% and 97.5%)
    centerline_lower_bound = np.percentile(zb_inverted_all[n_row_middle, :, :], 2.5, axis=-1)
    centerline_upper_bound = np.percentile(zb_inverted_all[n_row_middle, :, :], 97.5, axis=-1)

    crossline_lower_bound = np.percentile(zb_inverted_all[:, n_col_middle, :], 2.5, axis=-1)
    crossline_upper_bound = np.percentile(zb_inverted_all[:, n_col_middle, :], 97.5, axis=-1)

    #axs[0].plot(x, zb_inverted_all[n_row_middle, :, i - 2] * (zb_max - zb_min) + zb_min, 'r', linewidth=1, alpha=0.1)
    #axs[1].plot(y, zb_inverted_all[:, n_col_middle, i - 2] * (zb_max - zb_min) + zb_min, 'r', linewidth=1, alpha=0.1)

    axs[0].fill_between(x, centerline_lower_bound, centerline_upper_bound, color="red", alpha=0.3)
    axs[1].fill_between(y, crossline_lower_bound, crossline_upper_bound, color="red", alpha=0.3)


    axs[0].legend(loc='upper right', fontsize=14, frameon=False)
    axs[1].legend(loc='upper right', fontsize=14, frameon=False)

    #plt.savefig("zb_inversion_profiles.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig("zb_inversion_profiles_uv_uncertainty.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def animate_zb_inversion_process(config_filename):
    """
    Animate the zb inversion process
    1. plot zb comparison
    2. plot inversion loss history

    :param zb_inverted_result_filename:

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        # args = get_args()
        # config = process_config(args.config)

        # hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    var_min_max = data_loader.get_variables_min_max()

    zbs = np.load('zb_inverted_result_0.npz')
    zb_init_all = zbs['zb_init']
    zb_truth = zbs['zb_truth']
    zb_inverted_all = zbs['zb_inverted']
    zb_inverted_all = zbs['zb_inverted']
    vel_WSE_target = zbs['uvWSE_target']
    vel_WSE_pred_all = zbs['uvWSE_pred']

    #load intermediate zb results
    zbs = np.load('zb_intermediate_0.npz')
    zb_intermediate = zbs['zb_intermediate']

    #number of intermediate steps
    nSteps = zb_intermediate.shape[-1]

    #load loss data
    zb_inverted_results = np.load('zb_inverted_result_0.npz')
    losses = zb_inverted_results['losses']

    total_loss = np.squeeze(losses[:, 0])
    loss_prediction_error = np.squeeze(losses[:, 1])
    loss_value_regularization = np.squeeze(losses[:, 2])
    loss_slope_regularization = np.squeeze(losses[:, 3])

    # get the max and min of all losses (for plotting axis limits)
    v_min = min(total_loss.min(), loss_prediction_error.min(), loss_value_regularization.min(),
                loss_slope_regularization.min())
    v_max = max(total_loss.max(), loss_prediction_error.max(), loss_value_regularization.max(),
                loss_slope_regularization.max())

    x_min = 0
    x_max = total_loss.shape[0]

    vmin = -0.5
    vmax = 0.5
    levels = np.linspace(vmin, vmax, 51)

    n_rows = zb_inverted_all.shape[0]
    n_cols = zb_inverted_all.shape[1]

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    x = np.linspace(xl, xh, n_cols)
    y = np.linspace(yl, yh, n_rows)
    X, Y = np.meshgrid(x, y)

    nSteps = 1000
    for i in range(851,nSteps):
        print(i)

        # three columns for subplots: init/truth, zb_inverted/mean_zb, diff
        fig, axs = plt.subplots(2, 3, figsize=(20, 5), sharex=False, sharey=False, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.2, wspace=.15)

        # plot zb_truth
        cf_zb_truth = axs[0,0].contourf(X, Y, zb_truth, levels, vmin=vmin, vmax=vmax, cmap=plt.cm.terrain, extend='both')
        axs[0,0].set_xlim([xl, xh])
        axs[0,0].set_ylim([yl, yh])
        axs[0,0].set_aspect('equal')
        axs[0,0].set_ylabel('$y$ (m)', fontsize=16)
        axs[0,0].tick_params(axis='y', labelsize=14)
        axs[0,0].set_xlabel('$x$ (m)', fontsize=16)
        axs[0,0].tick_params(axis='x', labelsize=14)
        axs[0,0].set_title("$z_b$ (truth)", fontsize=16)
        divider = make_axes_locatable(axs[0,0])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(vmin, vmax, 7), cax=cax)
        clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb_truth.ax.tick_params(labelsize=12)

        # plot zb_inverted_mean
        cf_zb_inverted = axs[0,1].contourf(X, Y, np.squeeze(zb_intermediate[:,:,i]), levels, vmin=vmin, vmax=vmax, cmap=plt.cm.terrain, extend='both')
        axs[0,1].set_xlim([xl, xh])
        axs[0,1].set_ylim([yl, yh])
        axs[0,1].set_aspect('equal')
        axs[0,1].set_xlabel('$x$ (m)', fontsize=16)
        axs[0,1].tick_params(axis='x', labelsize=14)
        axs[0, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
        axs[0,1].set_title("Inverted $z_b$", fontsize=16)
        divider = make_axes_locatable(axs[0,1])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        clb_zb_inverted = fig.colorbar(cf_zb_inverted, ticks=np.linspace(vmin, vmax, 7), cax=cax)
        clb_zb_inverted.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
        clb_zb_inverted.ax.tick_params(labelsize=12)

        # plot diff(zb_truth - zb_inverted)
        zb_diff = zb_truth - np.squeeze(zb_intermediate[:,:,i])
        v_min = zb_diff.min()
        v_max = zb_diff.max()
        print("zb_diff min, max =", v_min, v_max)
        diff_levels = np.linspace(vmin, vmax, 51)
        cf_zb_diff = axs[0,2].contourf(X, Y, zb_diff, diff_levels, vmin=vmin, vmax=vmax, cmap=plt.cm.terrain, extend='both')  # cm: PRGn
        axs[0,2].set_xlim([xl, xh])
        axs[0,2].set_ylim([yl, yh])
        axs[0,2].set_aspect('equal')
        axs[0,2].set_xlabel('$x$ (m)', fontsize=16)
        axs[0,2].tick_params(axis='x', labelsize=14)
        axs[0, 2].tick_params(axis='y', which='both', left=False, labelleft=False)
        axs[0,2].set_title("$z_b$ differences", fontsize=16)
        divider = make_axes_locatable(axs[0,2])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        clb_zb_diff = fig.colorbar(cf_zb_diff, ticks=np.linspace(vmin, vmax, 5), cax=cax)
        clb_zb_diff.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
        clb_zb_diff.ax.tick_params(labelsize=12)

        axs[1,0].axis('off')
        axs[1,2].axis('off')

        #plot the loss histories
        axs[1, 1].plot(np.arange(i), total_loss[0:i], 'k', label="Total loss")
        axs[1, 1].plot(np.arange(i), loss_prediction_error[0:i], 'r', linestyle='dotted',
                       label="Prediction error loss")
        axs[1, 1].plot(np.arange(i), loss_value_regularization[0:i], 'g',
                       linestyle='dashed', label="Value regularization loss")
        axs[1, 1].plot(np.arange(i), loss_slope_regularization[0:i], 'b',
                       linestyle='dashdot', label="Slope regularization loss")

        axs[1, 1].set_yscale('log')

        # plt.xlim([x_min, x_max])
        axs[1, 1].set_xlim([x_min, nSteps])  # have more control on the upper x limit
        axs[1, 1].set_ylim([1e-4, 2000])  # have more control on the lower limit
        axs[1,1].set_yticks([1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3])

        axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

        axs[1, 1].set_xlabel('Iterations', fontsize=14)
        axs[1, 1].set_ylabel('Losses for inversion', fontsize=14)
        axs[1, 1].legend(loc='upper right', fontsize=8, frameon=False)

        fig.suptitle("Inversion iteration "+str(i), fontsize=18)

        plt.savefig("zb_intermediate_" + str(i).zfill(4) + ".png", dpi=300, bbox_inches='tight', pad_inches=0)

        #plt.close('all')

def image_sequence_to_animation():

    width = 4992
    height = 1456

    width_resize = int(width/2)
    height_resize = int(height/2)

    # choose codec according to format needed
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter('inversion_process.avi', fourcc, 25, (width_resize, height_resize))

    for i in range(0, 1000):
        print("i = ", i)

        img = cv2.imread("zb_intermediate_" + str(i).zfill(4) + ".png")
        img_resize = cv2.resize(img,(width_resize,height_resize),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        video.write(img_resize)

    cv2.destroyAllWindows()
    video.release()


def plot_zb_inversion_WSE_comparison(zb_inverted_result_filename):
    """
    Make plot for comparing zb, uvWSE, etc from inversion. There could
    be many samples in the result file.


    :return:
    """

    zb_inverted_results = np.load(zb_inverted_result_filename)
    zb_truth = zb_inverted_results['zb_truth']
    zb_init_all = zb_inverted_results['zb_init']
    zb_inverted_all = zb_inverted_results['zb_inverted']
    vel_WSE_target = zb_inverted_results['uvWSE_target']
    vel_WSE_pred_all = zb_inverted_results['uvWSE_pred']

    #number of samples
    nSamples = zb_inverted_all.shape[-1]

    #loop over all samples
    for i in range(nSamples):
        #for the case there is only one vel_WSE_target
        plot_one_zb_inversion_WSE(i, zb_truth, vel_WSE_target, vel_WSE_pred_all[:,:,:,i], zb_inverted_all[:,:,i])

        #for the case where there are many vel_WSE_target, e.g., with perturbations
       # plot_one_zb_inversion_WSE(i, zb_truth, vel_WSE_target[:,:,:,i], vel_WSE_pred_all[:,:,:,i], zb_inverted_all[:,:,i])

def plot_one_zb_inversion_WSE(sampleID, zb_truth, vel_WSE_target, vel_WSE_pred, zb_inverted):
    """
    Plot just one given zb inversion and WSE result.

    :return:
    """

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

    plt.savefig("inversion_comparison_zb_WSE_"+str(sampleID).zfill(4)+".png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    #plt.show()

    plt.close('all')


def plot_uvWSE_masking_vs_original(zb_inverted_result_filename):
    """
    Make plots of uvWSE comparing original and masked


    :return:
    """

    masks = np.load('masks_05.npz')['masks']

    zb_inverted_results = np.load(zb_inverted_result_filename)
    zb_truth = zb_inverted_results['zb_truth']
    zb_init_all = zb_inverted_results['zb_init']
    zb_inverted_all = zb_inverted_results['zb_inverted']
    vel_WSE_target = zb_inverted_results['uvWSE_target']
    vel_WSE_pred_all = zb_inverted_results['uvWSE_pred']

    #number of samples
    if np.ndim(zb_inverted_all) == 2:
        nSamples = 1
    else:
        nSamples = zb_inverted_all.shape[-1]

    #loop over all samples
    for i in range(nSamples):
        print("plot i = ",i,' out of ', nSamples)

        #for the case there is only one vel_WSE_target
        plot_one_uvWSE_masking_vs_original(i, vel_WSE_target, vel_WSE_pred_all[:,:,:,i], masks)


def plot_one_uvWSE_masking_vs_original(sampleID, vel_WSE_target, vel_WSE_pred, masks):
    """
    Plot just one uvWSE comparing original and masked

    :return:
    """

    fig, axs = plt.subplots(2, 3, figsize=(3 * 10, 3 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.25, wspace=.01)

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)

    n_rows = vel_WSE_target.shape[0]
    n_cols = vel_WSE_target.shape[1]

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

    # plot masked vel_x (simulated from SRH-2D; target)
    cf_vel_x_sim = axs[0, 1].contourf(np.multiply(np.squeeze(vel_WSE_target[:, :, 0]),masks), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 1].set_xlim([0, n_cols-1])
    axs[0, 1].set_ylim([0, n_rows-1])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("Simulated $u$ from SRH-2D (masked)", fontsize=14)
    clb_vel_x_sim = fig.colorbar(cf_vel_x_sim, ticks=np.linspace(min, max, 7), ax=axs[0, 1])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)

    # plot vel_x (predicted from NN)
    cf_vel_x_pred = axs[0, 2].contourf(np.squeeze(vel_WSE_pred[:, :, 0]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[0, 2].set_xlim([0, n_cols-1])
    axs[0, 2].set_ylim([0, n_rows-1])
    axs[0, 2].set_aspect('equal')
    axs[0, 2].set_title("Predicted $u$ by NN with inverted $z_b$", fontsize=14)
    clb_vel_x_pred = fig.colorbar(cf_vel_x_pred, ticks=np.linspace(min, max, 7), ax=axs[0, 2])
    clb_vel_x_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_pred.ax.tick_params(labelsize=12)

    # plot vel_y (simulated from SRH-2D; target)
    cf_vel_y_sim = axs[1, 0].contourf(np.squeeze(vel_WSE_target[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 0].set_xlim([0, n_cols - 1])
    axs[1, 0].set_ylim([0, n_rows - 1])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("Simulated $v$ from SRH-2D", fontsize=14)
    clb_vel_x_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(min, max, 7), ax=axs[1, 0])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)

    # plot masked vel_y (simulated from SRH-2D; target)
    cf_vel_y_sim = axs[1, 1].contourf(np.multiply(np.squeeze(vel_WSE_target[:, :, 1]), masks), levels, vmin=min,
                                      vmax=max, cmap=plt.cm.jet)
    axs[1, 1].set_xlim([0, n_cols - 1])
    axs[1, 1].set_ylim([0, n_rows - 1])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("Simulated $v$ from SRH-2D (masked)", fontsize=14)
    clb_vel_x_sim = fig.colorbar(cf_vel_y_sim, ticks=np.linspace(min, max, 7), ax=axs[1, 1])
    clb_vel_x_sim.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_sim.ax.tick_params(labelsize=12)

    # plot vel_y (predicted from NN)
    cf_vel_y_pred = axs[1, 2].contourf(np.squeeze(vel_WSE_pred[:, :, 1]), levels, vmin=min, vmax=max, cmap=plt.cm.jet)
    axs[1, 2].set_xlim([0, n_cols - 1])
    axs[1, 2].set_ylim([0, n_rows - 1])
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_title("Predicted $v$ by NN with inverted $z_b$", fontsize=14)
    clb_vel_x_pred = fig.colorbar(cf_vel_y_pred, ticks=np.linspace(min, max, 7), ax=axs[1, 2])
    clb_vel_x_pred.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x_pred.ax.tick_params(labelsize=12)

    # set labels
    plt.setp(axs[-1, :], xlabel='x')
    plt.setp(axs[:, 0], ylabel='y')

    plt.savefig("comparison_uvWSE_masked_vs_original_"+str(sampleID).zfill(4)+".png", dpi=300, bbox_inches='tight',
                pad_inches=0)
    #plt.show()

    plt.close('all')



def plot_inversion_losses(zb_inverted_result_filename):
    """"

    """

    zb_inverted_results = np.load(zb_inverted_result_filename)
    losses = zb_inverted_results['losses']

    # number of samples
    nSamples = losses.shape[-1]

    total_loss = np.squeeze(losses[:,0,:])
    loss_prediction_error = np.squeeze(losses[:, 1, :])
    loss_value_regularization = np.squeeze(losses[:, 2, :])
    loss_slope_regularization = np.squeeze(losses[:, 3, :])

    #get the max and min of all losses (for plotting axis limits)
    v_min = min(total_loss.min(), loss_prediction_error.min(), loss_value_regularization.min(),
                loss_slope_regularization.min())
    v_max = max(total_loss.max(), loss_prediction_error.max(), loss_value_regularization.max(),
                loss_slope_regularization.max())

    x_min = 0
    x_max = total_loss.shape[0]

    # loop over all samples
    for i in range(nSamples):
        print("Plotting inversion loss history for i = ", i, "out of", nSamples)
        #plot losses
        plt.plot(np.arange(total_loss.shape[0]), total_loss[:,i], 'k', label="Total loss")
        plt.plot(np.arange(loss_prediction_error.shape[0]), loss_prediction_error[:,i], 'r', linestyle='dotted', label="Prediction error loss")
        plt.plot(np.arange(loss_value_regularization.shape[0]), loss_value_regularization[:,i], 'g', linestyle='dashed', label="Value regularization loss")
        plt.plot(np.arange(loss_slope_regularization.shape[0]), loss_slope_regularization[:,i], 'b', linestyle='dashdot', label="Slope regularization loss")

        #plt.xlim([x_min, x_max])
        plt.xlim([x_min, 600])  #have more control on the upper x limit
        #plt.ylim([v_min, v_max])
        plt.ylim([1e-4, v_max])  #have more control on the lower limit

        plt.yscale('log')

        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Losses for inversion', fontsize=14)
        plt.legend(loc='upper right', fontsize=12, frameon=False)
        plt.savefig("inversion_loss_history"+str(i).zfill(4)+".png", dpi=300, bbox_inches='tight', pad_inches=0)
        #plt.show()

        plt.close('all')


def plot_filters_biases(model):
    """
    Plot the filters and biases of a CNN model for visualization purpose

    Reference:
    https://gist.githubusercontent.com/GeorgeSeif/5c26b6a84a08c2d844cd42fa87492981/raw/7cb283b7a0127c3c1afc073180c76630de5d93c3/visualise_filters.py

    :param model:
    :return:
    """

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv1'

    # Grab the filters and biases for that layer
    filters, biases = layer_dict[layer_name].get_weights()

    # Normalize filter values to a range of 0 to 1
    f_min, f_max = np.amin(filters), np.amax(filters)
    filters = (filters - f_min) / (f_max - f_min)

    # Plot first few filters
    n_rows = 3
    n_cols = 3
    n_filters = n_rows*n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 6), facecolor='w', edgecolor='k')
    #fig.subplots_adjust(hspace=.25, wspace=.01)

    for i in range(n_rows):
        for j in range(n_cols):
            filter_index = i*n_cols + j

            f = np.squeeze(filters[:, :, :, filter_index])

            # There is only one channel
            ax = axs[i, j]
            ax.imshow(f, cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


def plot_feature_maps(model, input):
    """
    Plot the filters and biases of a CNN model for visualization purpose

    Reference:
    https://gist.githubusercontent.com/GeorgeSeif/5c26b6a84a08c2d844cd42fa87492981/raw/7cb283b7a0127c3c1afc073180c76630de5d93c3/visualise_filters.py

    :param model:
    :param input:
        input image
    :return:
    """

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv1'

    #construct a model with only the input and the first conv layer
    sub_model = tf.keras.Model(inputs=model.inputs, outputs=layer_dict[layer_name].output)

    sub_model.summary()

    input = np.expand_dims(input, axis=0)

    # apply the sub_model to the input image
    feature_maps = sub_model.predict(input)

    # Plot first few feature maps
    n_rows = 4
    n_cols = 4
    n_maps = n_rows*n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.05, wspace=.05)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            #plot the bathyetry on [0,0] for reference
            if i == 0 and j == 0:
                ax.imshow(np.flip(np.squeeze(input), axis=0), cmap=plt.cm.terrain)

            else:
                map_index = i * n_cols + j

                f = np.squeeze(feature_maps[:, :, :, map_index])

                ax.imshow(f, cmap='viridis')



    plt.savefig("feature_maps_"+str(n_rows)+"_"+str(n_cols)+".png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_feature_maps_for_publication(model, input, data_loader):
    """
    For publication: plot input bathymetry, inversion at iteration 2, and two example feature maps

    To show the effect of feature map on the blocky nature of inversion at the beginning.

    Reference:
    https://gist.githubusercontent.com/GeorgeSeif/5c26b6a84a08c2d844cd42fa87492981/raw/7cb283b7a0127c3c1afc073180c76630de5d93c3/visualise_filters.py

    :param model:
    :param input:
        input image
    :return:
    """

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv1'

    #construct a model with only the input and the first conv layer
    sub_model = tf.keras.Model(inputs=model.inputs, outputs=layer_dict[layer_name].output)

    sub_model.summary()

    input = np.expand_dims(input, axis=0)

    # apply the sub_model to the input image
    feature_maps = sub_model.predict(input)

    # load intermediate zb results
    zbs = np.load('zb_intermediate_0.npz')
    zb_intermediate = zbs['zb_intermediate']

    var_min_max = data_loader.get_variables_min_max()

    xl = var_min_max['bounds'][0]
    xh = var_min_max['bounds'][1]
    yl = var_min_max['bounds'][2]
    yh = var_min_max['bounds'][3]

    x = np.linspace(xl, xh, input.shape[2])
    y = np.linspace(yl, yh, input.shape[1])
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(2, 2, figsize=(12, 4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.3)

    #plot input bathymetry
    vmin = -0.5
    vmax = 0.5
    levels = np.linspace(vmin, vmax, 51)
    cf_zb_truth = axs[0, 0].contourf(X, Y, (np.squeeze(input)), levels, vmin=vmin, vmax=vmax, cmap=plt.cm.terrain)
    axs[0, 0].set_xlim([xl, xh])
    axs[0, 0].set_ylim([yl, yh])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)
    axs[0, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='x', labelsize=14)
    axs[0, 0].set_title("Example $z_b$ (truth)", fontsize=16)
    axs[0, 0].text(-0.05, 1.1, "(a)", size=16, ha="center", transform=axs[0,0].transAxes)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(vmin, vmax, 7), cax=cax)
    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)

    #plot first feature map
    f = np.squeeze(np.flip(feature_maps[:, :, :, 1]))
    vmin = f.min()
    vmax = f.max()

    cf_zb_truth = axs[0, 1].imshow(f, cmap='viridis', origin='lower')
    #axs[0, 1].set_xlim([xl, xh])
    #axs[0, 1].set_ylim([yl, yh])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_ylabel('$y$', fontsize=16)
    axs[0, 1].tick_params(axis='y', labelsize=14)
    axs[0, 1].set_xlabel('$x$', fontsize=16)
    axs[0, 1].tick_params(axis='x', labelsize=14)
    axs[0, 1].set_title("Feature map 0", fontsize=16)
    axs[0, 1].text(-0.05, 1.1, "(b)", size=16, ha="center", transform=axs[0, 1].transAxes)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(vmin, vmax, 7), cax=cax)
    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)

    # plot second feature map
    f = np.squeeze(np.flip(feature_maps[:, :, :, 3]))
    vmin = f.min()
    vmax = f.max()

    cf_zb_truth = axs[1, 0].imshow(f, cmap='viridis', origin='lower')
    #axs[1, 0].set_xlim([xl, xh])
    #axs[1, 0].set_ylim([yl, yh])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_ylabel('$y$', fontsize=16)
    axs[1, 0].tick_params(axis='y', labelsize=14)
    axs[1, 0].set_xlabel('$x$', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 0].set_title("Feature map 1", fontsize=16)
    axs[1, 0].text(-0.05, 1.1, "(c)", size=16, ha="center", transform=axs[1, 0].transAxes)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(vmin, vmax, 7), cax=cax)
    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)

    # plot zb at inversion initial iteration
    vmin = -0.5
    vmax = 0.5
    levels = np.linspace(vmin, vmax, 51)
    cf_zb_truth = axs[1, 1].contourf(X, Y, zb_intermediate[:,:,1], levels, vmin=vmin, vmax=vmax, cmap=plt.cm.terrain
                                                   )
    axs[1, 1].set_xlim([xl, xh])
    axs[1, 1].set_ylim([yl, yh])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='y', labelsize=14)
    axs[1, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=14)
    axs[1, 1].set_title("Inverted $z_b$ at iteration 1", fontsize=16)
    axs[1, 1].text(-0.05, 1.1, "(d)", size=16, ha="center", transform=axs[1, 1].transAxes)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    clb_zb_truth = fig.colorbar(cf_zb_truth, ticks=np.linspace(vmin, vmax, 7), cax=cax)
    clb_zb_truth.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb_truth.ax.tick_params(labelsize=12)

    plt.savefig("feature_maps_inversion.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_modified_relu_scheme():

    # modified Rectified Linear Unit (ReLU)
    def modified_ReLU(x, xc, a):
        data = [max(0, abs(value-xc)-a) for value in x]
        return np.array(data, dtype=float)

    # Generating data for Graph
    xc = 5
    a = 1.5
    x_data = np.linspace(xc-a*2, xc+a*2, 100)
    y_data = modified_ReLU(x_data, xc, a)

    plt.figure(figsize=(5, 5))  # fig size same as before
    ax = plt.gca()  # you first need to get the axis handle
    ax.set_aspect(2)  # sets the height to width ratio
    fig=plt.gcf()

    # Graph
    plt.plot(x_data, y_data, 'b', clip_on=False)

    #plt.xlabel('$x$', fontsize=16)
    #ax.xaxis.set_label_coords(1.0, -0.05)
    #plt.ylabel('$f(x)$', fontsize=16)

    plt.axis([1.2,8.8,0,2])

    #arrowed_spines(fig, ax)

    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)

    plt.text(5, -0.1, '$x_c$', fontsize=16, horizontalalignment='center', verticalalignment='center')
    ax.scatter(5,0,s=20, c='k', clip_on=False)

    plt.text(3.5, -0.1, '$x_c-a$', fontsize=16, horizontalalignment='center', verticalalignment='center')
    ax.scatter(3.5, 0, s=20, c='k', clip_on=False)

    plt.text(6.5, -0.1, '$x_c+a$', fontsize=16, horizontalalignment='center', verticalalignment='center')
    ax.scatter(6.5, 0, s=20, c='k', clip_on=False)

    #anotate a
    ax.annotate("", xy=(3.5, 0.3), xytext=(5.05, 0.3), arrowprops=dict(arrowstyle='<->'), fontsize=14)
    ax.annotate("", xy=(3.5, 0.3), xytext=(5.05, 0.3), arrowprops=dict(arrowstyle='|-|'))
    bbox = dict(fc="white", ec="none")
    ax.text(4.25, 0.31, "$a$", ha="center", va="center", bbox=bbox, fontsize=16)

    ax.annotate("", xy=(4.95, 0.3), xytext=(6.5, 0.3), arrowprops=dict(arrowstyle='<->'), fontsize=14)
    ax.annotate("", xy=(4.95, 0.3), xytext=(6.5, 0.3), arrowprops=dict(arrowstyle='|-|'))
    bbox = dict(fc="white", ec="none")
    ax.text(5.75, 0.31, "$a$", ha="center", va="center", bbox=bbox, fontsize=16)

    plt.text(9, -0.15, '$x$', fontsize=18, horizontalalignment='center', verticalalignment='center')
    plt.text(0.7, 1.7, '$f(x)$', fontsize=17, horizontalalignment='center', verticalalignment='center', rotation=90)

    plt.text(0.9, -0.1, '0', fontsize=18, horizontalalignment='center', verticalalignment='center')

    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #x arrow
    ax.arrow(1.2, 0, 8, 0., fc='k', ec='k', lw=1,
             head_width=0.1, head_length=0.2, overhang=0.1,
             length_includes_head=True, clip_on=False)
    #y arrow
    ax.arrow(1.2, 0, 0, 2., fc='k', ec='k', lw=1,
             head_width=0.2, head_length=0.1, overhang=0.1,
             length_includes_head=True, clip_on=False)

    plt.savefig("modified_relu.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def test_tf_random_generator():
    g1 = tf.random.Generator.from_seed(1)
    print(g1.uniform(shape=[5, 5],minval=-0.025, maxval=0.025))
    print(g1.uniform(shape=[5, 5],minval=-0.025, maxval=0.025))

    #g2 = tf.random.get_global_generator()
    #print(g2.normal(shape=[2, 3]))



if __name__ == '__main__':

    #plot_modified_relu_scheme()

    #test_tf_random_generator()

    plot_zb_inversion_loss_components_cnn_structure()

    print("Done!")