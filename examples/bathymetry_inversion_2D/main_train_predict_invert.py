#the following import has to be done twice (at the very beginning and in trainer).
#see bug report: https://github.com/Ahmkel/Keras-Project-Template/issues/7
from comet_ml import Experiment

import sys
sys.path.append("..")  #for execution in terminal

from dl4HM.data_loader.swe_2D_data_loader import SWEs2DDataLoader
from dl4HM.models.swe_2D_model import SWEs2DModel
from dl4HM.trainers.swe_2D_trainer import SWEs2DModelTrainer
from dl4HM.inverters.swe_2D_inverter import SWEs2DModelInverter

from dl4HM.utils.config import process_config
from dl4HM.utils.dirs import create_dirs
from dl4HM.utils.args import get_args
from dl4HM.utils.misc import print_now_time

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as tick

import tensorflow as tf

import json

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def train():
    """The training step

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config('surrogate_bathymetry_inversion_2D_config.json')
    except:
        raise Exception("missing or invalid arguments")

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Create the trainer')
    trainer = SWEs2DModelTrainer(modelWrapper, data_loader, config)

    print('Start training the model.')
    print_now_time(string_before="Training start:")
    trainer.train()
    print_now_time(string_before="Training end:")

    #plot the loss and acc for each epoch
    loss = trainer.loss
    val_loss = trainer.val_loss

    if (len(loss) !=0):
        plt.plot(np.arange(len(loss)),loss, label='training loss')

    if (len(val_loss) !=0):
        plt.plot(np.arange(len(val_loss)), val_loss, label='validation loss')

    plt.yscale('log')

    plt.title('training loss and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("training_validtion_losses.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #This is for the losses for each epoch
    loss_value = modelWrapper.loss_value_epoch

    plt.plot(np.arange(len(loss_value)),loss_value, label='loss (value)')

    plt.yscale('log')

    plt.title('losses due to value errors')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("loss_components.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #print('Test the trained model.')
    #model.test_model(data_loader.get_test_data())


def predict():
    """The prediction step

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config('surrogate_bathymetry_inversion_2D_config.json')
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    #the following path needs to be adjusted according to where the model is saved
    modelWrapper.load("./experiments/2021-12-26/test/checkpoints/test-200.hdf5")

    print('Make prediction with the trained model.')

    #Get the min and max of all variables for scaling


    plot_ID = 0  # which one to plot

    test_dataset = data_loader.get_test_data()
    iterator = iter(test_dataset)
    zb_test, vel_WSE_test = next(iterator)

    vel_WSE_pred = modelWrapper.model.predict(zb_test)

    fig, axs = plt.subplots(3, 2, figsize=(2 * 10, 3 * 2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.01)

    # plot vel_x (simulated)
    levels = np.linspace(0, 1, 51)
    cf_zb = axs[0, 0].contourf(np.squeeze(vel_WSE_test[plot_ID, :, :, 0]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title("Simulated vel_x", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[0, 0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot vel_x (predicted from NN)
    cf_zb = axs[0, 1].contourf(np.squeeze(vel_WSE_pred[plot_ID, :, :, 0]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("Predicted vel_x", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[0, 1])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot vel_y (simulated)
    levels = np.linspace(0, 1, 51)
    cf_zb = axs[1, 0].contourf(np.squeeze(vel_WSE_test[plot_ID, :, :, 1]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("Simulated vel_y", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[1, 0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot vel_y (predicted from NN)
    cf_zb = axs[1, 1].contourf(np.squeeze(vel_WSE_pred[plot_ID, :, :, 1]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("Predicted vel_y", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[1, 1])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot WSE (simulated)
    levels = np.linspace(0, 1, 51)
    cf_zb = axs[2, 0].contourf(np.squeeze(vel_WSE_test[plot_ID, :, :, 2]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[2, 0].set_aspect('equal')
    axs[2, 0].set_title("Simulated WSE", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[2, 0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot vel_y (predicted from NN)
    cf_zb = axs[2, 1].contourf(np.squeeze(vel_WSE_pred[plot_ID, :, :, 2]), levels, vmin=0, vmax=1, cmap=plt.cm.jet)
    # axs[0, 0].set_xlim([bounds[0], bounds[1]])
    # axs[0, 0].set_ylim([bounds[2], bounds[3]])
    axs[2, 1].set_aspect('equal')
    axs[2, 1].set_title("Predicted WSE", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(0, 1, 7), ax=axs[2, 1])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    # clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # set labels
    plt.setp(axs[-1, :], xlabel='x (m)')
    plt.setp(axs[:, 0], ylabel='y (m)')

    plt.savefig("prediction.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def invert():
    """The inversion step

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config('backwater_curve_surrogate_config.json')
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = BackwaterCurveDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = BackwaterCurveModel(config, data_loader)

    print('Load model weights from checkpoint.')
    #the following path needs to be adjusted according to where the model is saved
    modelWrapper.load("./experiments/2021-12-16/backwater_curve/checkpoints/backwater_curve-459.hdf5")

    print('Create the inverter.')
    inverter = BackwaterCurveModelInverter(modelWrapper, data_loader, config)

    print_now_time(string_before="Inversion start:")
    inverter.invert()
    print_now_time(string_before="Inversion end:")

    print("Inverted scaled zb = ", inverter.get_zb())

    # Get the min and max of all variables for scaling
    zb_beds_min = data_loader.zb_beds_min
    zb_beds_max = data_loader.zb_beds_max
    WSE_min = data_loader.WSE_min
    WSE_max = data_loader.WSE_max

    zb_inverted = inverter.get_zb() * (zb_beds_max - zb_beds_min) + zb_beds_min

    print("Inverted zb = ", zb_inverted)

    zb_truth = data_loader.get_zb_beds_truth()* (zb_beds_max - zb_beds_min) + zb_beds_min

    print("True zb = ", zb_truth)

    WSE_target = inverter.get_WSE_target()
    WSE_pred = inverter.get_WSE_pred()

    #scale
    WSE_target = WSE_target * (WSE_max - WSE_min) + WSE_min
    WSE_pred = WSE_pred * (WSE_max - WSE_min) + WSE_min

    x_bed_train = data_loader.get_x_bed_train()
    x_pred_plot = data_loader.get_x_train()

    # plot x vs y_pred
    plt.plot(x_pred_plot, WSE_pred, 'r--', label='Predicted')

    # plot x vs y_truth
    plt.plot(x_pred_plot, WSE_target, 'r', label='Truth')

    # plot x vs bed elevation
    plt.plot(x_bed_train, zb_inverted, 'k--', label = 'Inverted zb')
    plt.plot(x_bed_train, zb_truth, 'k', label = 'True zb')


    # set the limit for the x and y axes
    # plt.xlim([0,1.0])
    # plt.ylim([-1,3.5])

    plt.title('Backwater curve')
    plt.xlabel('x (m)')
    plt.ylabel('elevation (m)')
    plt.legend()
    plt.savefig("inversion.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()



if __name__ == '__main__':

    train()

    #predict()

    #invert()

    print("All done!")
