#the following import has to be done twice (at the very beginning and in trainer).
#see bug report: https://github.com/Ahmkel/Keras-Project-Template/issues/7
from comet_ml import Experiment

import sys
sys.path.append("../..")  #for execution in terminal

from dl4HM.data_loader.backwater_curve_data_loader import BackwaterCurveDataLoader
from dl4HM.models.backwater_curve_model import BackwaterCurveModel
from dl4HM.trainers.backwater_curve_trainer import BackwaterCurveModelTrainer
from dl4HM.inverters.backwater_inverter import BackwaterCurveModelInverter

from dl4HM.utils.config import process_config
from dl4HM.utils.dirs import create_dirs
from dl4HM.utils.args import get_args
from dl4HM.utils.misc import print_now_time

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np

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
        config = process_config('backwater_curve_surrogate_config.json')
    except:
        raise Exception("missing or invalid arguments")

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data loader/generator.')
    data_loader = BackwaterCurveDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = BackwaterCurveModel(config, data_loader)

    print('Create the trainer')
    trainer = BackwaterCurveModelTrainer(modelWrapper, data_loader, config)

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

    #plot training loss components
    #This is for the losses for each batch
    #loss_value = model.loss_value

    #Thi is for the losses for each epoch
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

    print('Make prediction with the trained model.')

    #Get the min and max of all variables for scaling
    zb_beds_min = data_loader.zb_beds_min
    zb_beds_max = data_loader.zb_beds_max
    WSE_min = data_loader.WSE_min
    WSE_max = data_loader.WSE_max

    profile_ID = 20  # which profile to predict and plot

    #zb_pred, y_train = data_loader.get_train_data()   #[::10] every 10th x value for plotting
    zb_pred, y_train = data_loader.get_test_data()
    WSE_pred = modelWrapper.model.predict(np.reshape(zb_pred[profile_ID,:], (1,-1)))

    #
    x_pred_plot = data_loader.get_x_train()
    WSE_truth= y_train[profile_ID,:]
    WSE_pred_plot = WSE_pred[0]

    zb_plot = zb_pred[profile_ID,:]

    x_bed_train = data_loader.get_x_bed_train()

    # report model error
    print('MSE: %e' % mean_squared_error(WSE_truth, WSE_pred_plot))

    #scale WSE
    WSE_truth = WSE_truth*(WSE_max-WSE_min) + WSE_min
    WSE_pred_plot = WSE_pred_plot * (WSE_max - WSE_min) + WSE_min

    zb_plot = zb_plot * (zb_beds_max - zb_beds_min) + zb_beds_min

    # plot x vs y_truth
    plt.plot(x_pred_plot, WSE_truth, 'r', label='Truth')

    # plot x vs y_pred
    #plt.scatter(x_pred_plot, WSE_pred_plot, s=3, facecolors='none', edgecolors='k', label='Predicted')
    plt.plot(x_pred_plot, WSE_pred_plot, 'b', label='Predicted')

    # plot x vs bed elevation
    plt.plot(x_bed_train, zb_plot, 'k')

    # set the limit for the x and y axes
    # plt.xlim([0,1.0])
    #plt.ylim([-1,3.5])

    plt.title('Backwater curve')
    plt.xlabel('x (m)')
    plt.ylabel('elevation (m)')
    plt.legend()
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

    #train()

    #predict()

    invert()

    print("All done!")
