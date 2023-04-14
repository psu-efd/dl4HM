#the following import has to be done twice (at the very beginning and in trainer).
#see bug report: https://github.com/Ahmkel/Keras-Project-Template/issues/7
from comet_ml import Experiment

import sys
sys.path.append("../../../bathymetry_inversion_2D")  #for execution in terminal
#print(sys.path)

import specific_utilities

from dl4HM.data_loader.swe_2D_data_loader import SWEs2DDataLoader
from dl4HM.models.swe_2D_model import SWEs2DModel
from dl4HM.trainers.swe_2D_trainer import SWEs2DModelTrainer
from dl4HM.inverters.swe_2D_inverter import SWEs2DModelInverter

from dl4HM.utils.config import process_config
from dl4HM.utils.dirs import create_dirs
from dl4HM.utils.args import get_args
from dl4HM.utils.misc import print_now_time

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def train(config_filename):
    """The training step

    :param config_filename: str
        file name for the configuration file

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
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
    #plot_training_validation_losses(trainer.config.trainer.history_save_filename)

    #print('Test the trained model.')
    #model.test_model(data_loader.get_test_data())


def predict(config_filename, trained_model_data_filename, nPlotSamples=1):
    """The prediction step

    :param config_filename: str
        file name for the configuration
    :param trained_model_data_filename: str
        file name for the trained model in checkpoints
    :param nPlotSamples: int
        number of samples to plot. If -1, plot all in the records. Default = 1

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    modelWrapper.load(trained_model_data_filename)

    print('Make prediction on test data with the trained model.')

    #Get test data
    test_dataset = data_loader.get_test_data()

    # IDs in the recrod file
    IDs = []

    # check whether vel_WSE contains WSE or it is (u,v) only
    b_uv_only = True

    for record in test_dataset:
        ID, zb, vel_WSE = record
        IDs.append(ID.numpy())

        if vel_WSE.numpy().shape[2] == 3:  # This in fact only needs to be checked once
            b_uv_only = False

    print("There are total of ", len(IDs), " records in the dataset. They are: ", IDs)

    # randonly draw cases
    nExamples = 0
    if nPlotSamples == -1:
        nExamples = len(IDs)
    else:
        if nPlotSamples <= len(IDs):
            nExamples = nPlotSamples
        else:
            raise Exception("The specified number of plots nPlotSamples, ", nPlotSamples,
                            " is larger than the number of records.")

    choices = np.sort(np.random.choice(IDs, size=nExamples, replace=False))

    print("Chosen case IDs for plotting: ", choices)

    #hack: to plot the ID you want as the first one
    choices[0] = 592

    counter = 0

    zb_test_all = []
    uvWSE_test_all = []
    uvWSE_pred_all = []

    prediction_l2norms = np.zeros(nExamples)
    IDs_l2norms = []

    # loop over all records
    for record in test_dataset:
        ID, zb_test, uvWSE_test = record

        # if the current case is in the chosen list for plotting
        if ID.numpy() in choices:
            counter = counter + 1

            print("Predicting case ID =", ID.numpy(), ",", counter, "out of", len(IDs))

            # predict using the trained model
            #zb_test = zb_test*0
            uvWSE_pred = modelWrapper.model.predict(tf.expand_dims(zb_test, axis=0))

            #calculate the L2 norm of prediction error (for inversion accuracy purpose)
            if config.dataLoader.uv_only:  # The NN output only includes u and v
                error = uvWSE_pred[:,:,:,0:2].squeeze() - uvWSE_test[:,:,0:2]   #if only uv is used for inversion
            else:
                error = uvWSE_pred.squeeze() - uvWSE_test                       #if uvWSE is used for inversion

            prediction_l2norms[counter-1] = np.linalg.norm(error)
            IDs_l2norms.append(ID.numpy())

            #hack:
            #if ID.numpy() == 180:
            #    print("l2_norm of case 180 = ", prediction_l2norms[counter-1])

            #collect all result data
            zb_test_all.append(zb_test)
            uvWSE_test_all.append(tf.expand_dims(uvWSE_test, axis=-1))
            uvWSE_pred_all.append(tf.expand_dims(tf.squeeze(uvWSE_pred), axis=-1))

            # make plot
            specific_utilities.plot_one_prediction(ID, b_uv_only, uvWSE_pred, uvWSE_test, zb_test, var_min_max = data_loader.get_variables_min_max(), bPlot_zb=False)

    # put all collected data together as a 3D numpy array
    zb_test_all = np.concatenate(zb_test_all, axis=-1)
    uvWSE_test_all = np.concatenate(uvWSE_test_all, axis=-1)
    uvWSE_pred_all = np.concatenate(uvWSE_pred_all, axis=-1)

    #print l2 norms and their mean
    print("IDs_l2norms = ", IDs_l2norms)
    print("prediction_l2norms = ", prediction_l2norms)
    print("prediction_l2norms.mean = ", prediction_l2norms.mean())

    #save the collected data
    #np.savez("prediction_results.npz", zb_test_all=zb_test_all, uvWSE_test_all=uvWSE_test_all,
    #         uvWSE_pred_all=uvWSE_pred_all, IDs_l2norms=np.array(IDs_l2norms), prediction_l2norms=prediction_l2norms)


def visualize_filters_feature_maps(config_filename, trained_model_data_filename):
    """

    Given an input bathymetry image, visualize filters and feature maps from a trained CNN model.

    :param config_filename: str
        file name for the configuration
    :param trained_model_data_filename: str
        file name for the trained model in checkpoints
    :param nPlotSamples: int
        number of samples to plot. If -1, plot all in the records. Default = 1

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    modelWrapper.load(trained_model_data_filename)

    print('Visualizing the trained model.')

    #plot the filters
    #specific_utilities.plot_filters_biases(modelWrapper.model)

    #Get test data
    test_dataset = data_loader.get_test_data()

    # IDs in the recrod file
    IDs = []

    # check whether vel_WSE contains WSE or it is (u,v) only
    b_uv_only = True

    for record in test_dataset:
        ID, zb, vel_WSE = record
        IDs.append(ID.numpy())

        if vel_WSE.numpy().shape[2] == 3:  # This in fact only needs to be checked once
            b_uv_only = False

    print("There are total of ", len(IDs), " records in the dataset. They are: ", IDs)

    #hack: to visualize the ID you want as the first one
    choices = [592]

    counter = 0

    # loop over all records
    for record in test_dataset:
        ID, zb_test, vel_WSE_test = record

        # if the current case is in the chosen list for visualization
        if ID.numpy() in choices:
            counter = counter + 1

            print("Visualizing ID =", ID.numpy(), ",", counter, "out of", len(IDs))

            #specific_utilities.plot_feature_maps(modelWrapper.model, zb_test)

            specific_utilities.plot_feature_maps_for_publication(modelWrapper.model, zb_test, data_loader)


def invert(config_filename, trained_model_data_filename, zb_inverted_save_filename):
    """The inversion step

    :param config_filename: str
        file name for the configuration
    :param trained_model_data_filename: str
        file name for the trained model in checkpoints
    :param zb_inverted_save_filename: str
        file name for saving the inverted zb in npz format

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    modelWrapper.load(trained_model_data_filename)

    print('Create the inverter.')
    inverter = SWEs2DModelInverter(modelWrapper, data_loader, config)

    #get zb_truth and uvWSE_target
    zb_truth = np.squeeze(data_loader.get_zb_truth())
    uvWSE_target = np.squeeze(inverter.get_uvWSE_target())

    #arrays to hold inverted zbs
    zb_init_all = None
    zb_inverted_all = None
    uvWSE_pred_all = []
    losses_all = None

    #whether the inversion initial zb is from a file or zero
    b_zb_init_from_file = config.inverter.b_zb_init_from_file

    if not b_zb_init_from_file: #zb_init = 0
        nSamples = 1

        zb_init = np.zeros(inverter.get_input_data_shape())

    else:    #zb_init is provided in a file
         #load the initial zb for multiple inversions
         sampled_elevations_for_inversion_init = np.load("sampled_elevations_for_inversion_init.npz")

         sampled_elevations_for_inversion_init = sampled_elevations_for_inversion_init['elevations']

         # zero initial zb + those provided in file
         nSamples = 1 + sampled_elevations_for_inversion_init.shape[-1]

    #loop over each initial zb to do inversion
    for i in range(nSamples):
        print("Inversion #: ", i)

        if i == 0: #always use the zero zb_init regardless whether the zb_init file is provided or not
            zb_init = (np.random.random(inverter.get_input_data_shape()) - 0.5) * 0.1
        else:
            zb_init = sampled_elevations_for_inversion_init[:,:,i-1]
            zb_init = zb_init[:,:,np.newaxis]

        inverter.initialize_variables(zb_init=zb_init)

        inverter.invert()
        #inverter.invert_scipy()

        # get inverted zb
        zb_inverted = np.squeeze(inverter.get_zb())

        # get uvWSE output from NN (based on the inverted zb)
        uvWSE_pred = inverter.get_uvWSE_pred()
        uvWSE_pred = uvWSE_pred[:,:,:,np.newaxis]

        # get inversion loss history (converted to a numpy array)
        losses = np.array(inverter.get_inversion_loss_history())

        if i == 0:
            zb_init_all = np.squeeze(zb_init)
            zb_inverted_all = zb_inverted
            uvWSE_pred_all.append(uvWSE_pred)
            losses_all = losses
        else:
            zb_init_all = np.dstack((zb_init_all, np.squeeze(zb_init)))
            zb_inverted_all = np.dstack((zb_inverted_all, zb_inverted))
            uvWSE_pred_all.append(uvWSE_pred)
            losses_all = np.dstack((losses_all, losses))

        #save intermediate zb (the inversion process)
        if i == 0 or i == 1: #specify which realization you want to save
            np.savez("zb_intermediate_"+str(i)+".npz", zb_intermediate=inverter.get_zb_intermediate())

    #put all uvWSE_pred together a 3D numpy array
    uvWSE_pred_all = np.concatenate(uvWSE_pred_all, axis=-1)

    #save zb_inverted and zb_truth
    np.savez(zb_inverted_save_filename, zb_init=zb_init_all, zb_inverted=zb_inverted_all, zb_truth=zb_truth,
             uvWSE_target=uvWSE_target, uvWSE_pred=uvWSE_pred_all, losses=losses_all)


def invert_L_curve(config_filename, trained_model_data_filename):
    """To produce the L-curve for inversion: find the optimal slope loss parameter alpha_slope

    :param config_filename: str
        file name for the configuration
    :param trained_model_data_filename: str
        file name for the trained model in checkpoints
    :param zb_inverted_save_filename: str
        file name for saving the inverted zb in npz format

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    modelWrapper.load(trained_model_data_filename)

    print('Create the inverter.')
    inverter = SWEs2DModelInverter(modelWrapper, data_loader, config)

    #alpha_slope values to try
    nAlpha = 11   #number of alpha_slope values
    alpha_slope = np.logspace(-2, 1.5, num=nAlpha)  #linearly distributed in log space

    #inital bed
    #zb_init = (np.random.random(inverter.get_input_data_shape()) - 0.5) * 0.1
    # load the initial zb for multiple inversions
    sampled_elevations_for_inversion_init = np.load("sampled_elevations_for_inversion_init.npz")
    sampled_elevations_for_inversion_init = sampled_elevations_for_inversion_init['elevations']

    #number of inital beds
    nBeds = sampled_elevations_for_inversion_init.shape[-1]
    print("There are ", nBeds, " initial beds for inversion.")

    # array to hold All (nAlpha x nBeds x 4) loss values
    losses_all = np.zeros([nAlpha, nBeds, 4])

    #loop over each alpha_slope to do inversion
    for i in range(nAlpha):
        print("Inversion #: ", i, " with alpha_slope = ", alpha_slope[i])

        config.inverter.slope_regularization_factor = alpha_slope[i]

        for init_bed_ID in range(nBeds):
            print("\tusing bed #", init_bed_ID, "out of", nBeds)

            zb_init = sampled_elevations_for_inversion_init[:, :, init_bed_ID]
            zb_init = zb_init[:, :, np.newaxis]

            inverter.initialize_variables(zb_init=zb_init)

            inverter.invert()

            # get inversion loss history (converted to a numpy array)
            losses = np.array(inverter.get_inversion_loss_history())

            #take the mean of loss at the end; con't take the last one because it fluctuates
            losses_all[i, init_bed_ID, 0] = losses[800:, 0].mean()
            losses_all[i, init_bed_ID, 1] = losses[800:, 1].mean()
            losses_all[i, init_bed_ID, 2] = losses[800:, 2].mean()/(config.inverter.value_regularization_factor)   #need to divided by factor
            losses_all[i, init_bed_ID, 3] = losses[800:, 3].mean()/(config.inverter.slope_regularization_factor)

    #save all losses for L-curve
    np.savez("losses_L_curve.npz", alpha_slope=alpha_slope, losses_all=losses_all)

def invert_uv_uncertainty(config_filename, trained_model_data_filename, zb_inverted_save_filename):
    """The inversion step for just one zb. However, the (u,v) is perturbed with random Gaussian noise.

    :param config_filename: str
        file name for the configuration
    :param trained_model_data_filename: str
        file name for the trained model in checkpoints
    :param zb_inverted_save_filename: str
        file name for saving the inverted zb in npz format

    :return:
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        #args = get_args()
        #config = process_config(args.config)

        #hard-wired the JSON configuration file name
        config = process_config(config_filename)
    except:
        raise Exception("missing or invalid arguments")

    print('Create the data loader/generator.')
    data_loader = SWEs2DDataLoader(config)

    print('Create the model wrapper.')
    modelWrapper = SWEs2DModel(config, data_loader)

    print('Load model weights from checkpoint.')
    modelWrapper.load(trained_model_data_filename)

    print('Create the inverter.')
    inverter = SWEs2DModelInverter(modelWrapper, data_loader, config)

    #get zb_truth and uvWSE_target
    zb_truth = np.squeeze(data_loader.get_zb_truth())
    #uvWSE_target = np.squeeze(inverter.get_uvWSE_target())

    #arrays to hold inverted zbs
    zb_init_all = None
    zb_inverted_all = None
    uvWSE_target_all = []
    uvWSE_pred_all = []
    losses_all = None

    #here we only need one zb_init
    zb_init = (np.random.random(inverter.get_input_data_shape()) - 0.5) * 0.1

    #how many (u,v) samples to invert from
    nSamples = 200

    #perturbation level (normalized value)
    perturb_level = 0.05  #=0.5*10%

    #loop over each initial zb to do inversion
    for i in range(nSamples):
        print("Inversion #: ", i, "out of", nSamples)

        inverter.initialize_variables(zb_init=zb_init)

        #add pertubation to (u,v,WSE)
        inverter.perturb_uvWSE(perturb_level)

        inverter.invert()

        # get inverted zb
        zb_inverted = np.squeeze(inverter.get_zb())

        # get uvWSE output from NN (based on the inverted zb)
        uvWSE_pred = inverter.get_uvWSE_pred()
        uvWSE_pred = uvWSE_pred[:,:,:,np.newaxis]

        # get uvWSE_target
        uvWSE_target = inverter.get_uvWSE_target()
        uvWSE_target = uvWSE_target[:,:,:,np.newaxis]

        # get inversion loss history (converted a numpy array)
        losses = np.array(inverter.get_inversion_loss_history())

        if i == 0:
            zb_init_all = np.squeeze(zb_init)
            zb_inverted_all = zb_inverted
            uvWSE_pred_all.append(uvWSE_pred)
            uvWSE_target_all.append(uvWSE_target)
            losses_all = losses
        else:
            zb_init_all = np.dstack((zb_init_all, np.squeeze(zb_init)))
            zb_inverted_all = np.dstack((zb_inverted_all, zb_inverted))
            uvWSE_pred_all.append(uvWSE_pred)
            uvWSE_target_all.append(uvWSE_target)
            losses_all = np.dstack((losses_all, losses))

        #save intermediate zb (the inversion process)
        #if i == 0: #specify which realization you want to save
        #    np.savez("zb_intermediate_"+str(i)+"_uv_uncertainty.npz", zb_intermediate=inverter.get_zb_intermediate())

    #put all uvWSE_pred together in a 3D numpy array
    uvWSE_pred_all = np.concatenate(uvWSE_pred_all, axis=-1)

    uvWSE_target_all = np.concatenate(uvWSE_target_all, axis=-1)

    #save zb_inverted and zb_truth
    np.savez(zb_inverted_save_filename, zb_init=zb_init_all, zb_inverted=zb_inverted_all, zb_truth=zb_truth,
             uvWSE_target=uvWSE_target_all, uvWSE_pred=uvWSE_pred_all, losses=losses_all)

def prior_info():

    inversion_data = np.load("inversion_case_uvWSE_0206.npz")

    zb_truth = inversion_data['zb'].squeeze()

    print("min, max of zb =", zb_truth.min(), zb_truth.max())

    dzb_dx = (np.diff(zb_truth, axis=1))
    dzb_dy = (np.diff(zb_truth, axis=0))

    print("min, max of slope_x =", dzb_dx.min(), dzb_dx.max())
    print("min, max of slope_y =", dzb_dy.min(), dzb_dy.max())

    bathymetry_data = np.load("twoD_bathymetry_data.npz")

    elevation = bathymetry_data['elevation']

    print("Done.")

if __name__ == '__main__':

    config_filename = 'surrogate_bathymetry_inversion_2D_config_uvWSE.json'
    #config_filename = 'surrogate_bathymetry_inversion_2D_config_uv.json'

    #train(config_filename=config_filename)
    specific_utilities.plot_training_validation_losses("training_history_uvWSE.json")

    trained_model_data_filename = "./experiments/2022-01-23/uvWSE/checkpoints/uvWSE.hdf5"
    #trained_model_data_filename = "./experiments/2022-01-25/uv/checkpoints/uv.hdf5"

    zb_inverted_save_filename = "zb_inverted_result_0592.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0592_no_value_reg.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0592_no_slope_reg.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0592_no_regs.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0592_uv_uncertainty.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0.npz"
    #zb_inverted_save_filename = "zb_inverted_result_uvWSE_NN_uvWSE.npz"
    #zb_inverted_save_filename = "zb_inverted_result_0592_uv_NN_uv.npz"

    #visualize filters and feature maps
    #visualize_filters_feature_maps(config_filename=config_filename, trained_model_data_filename=trained_model_data_filename)

    #specific_utilities.calculate_relative_and_absolute_errors_for_all_test_cases()

    #predict(config_filename=config_filename, trained_model_data_filename=trained_model_data_filename, nPlotSamples=1)

    #obtain the L-curve for inversion:
    #invert_L_curve(config_filename='surrogate_bathymetry_inversion_2D_config.json', trained_model_data_filename=trained_model_data_filename)
    #specific_utilities.plot_L_curve(losses_L_curve_file_name="losses_L_curve_0592.npz")

    #print_now_time(string_before="Inversion start:")
    #invert(config_filename=config_filename, trained_model_data_filename=trained_model_data_filename, zb_inverted_save_filename=zb_inverted_save_filename)
    #print_now_time(string_before="Inversion end:")

    #invert_uv_uncertainty(config_filename=config_filename, trained_model_data_filename=trained_model_data_filename, zb_inverted_save_filename=zb_inverted_save_filename)

    #specific_utilities.plot_zb_inversion_result(zb_inverted_save_filename, config_filename=config_filename)
    #specific_utilities.plot_zb_inversion_result_profiles(zb_inverted_save_filename, config_filename=config_filename)
    #specific_utilities.plot_zb_inversion_result_profiles_uv_uncertainty(zb_inverted_save_filename, config_filename=config_filename)
    #specific_utilities.plot_zb_inversion_WSE_comparison(zb_inverted_save_filename)
    #specific_utilities.plot_inversion_losses(zb_inverted_save_filename)

    #specific_utilities.plot_zb_inversion_regularization_effects(config_filename)
    #specific_utilities.plot_zb_inversion_result_profiles_regularization_effects(config_filename)

    #specific_utilities.plot_zb_inversion_loss_components_cnn_structure()

    #specific_utilities.animate_zb_inversion_process(config_filename=config_filename)
    #specific_utilities.image_sequence_to_animation()
    #specific_utilities.zb_inversion_process_for_publication(config_filename)

    #specific_utilities.plot_uvWSE_masking_vs_original(zb_inverted_save_filename)

    #specific_utilities.plot_prediction_l2norms_histogram()

    #prior_info()

    print("All done!")
