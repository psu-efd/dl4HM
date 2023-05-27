# Bathymetry inversion with a deep-learning-based surrogate model

### by Xiaofeng Liu

This folder contains the code for bathymetry inversion work done with a deep-learning-based surrogate model. A manuscript underview:

Xiaofeng Liu, Yalan Song, and Chaopeng Shen (2022). Bathymetry Inversion using a Deep-Learning-Based Surrogate for Shallow Water Equations Solvers.

## Description of files:

The main file:
- "main_train_predict_invert.py": this is the file where the training, prediction, and inversion happen. 
  - It needs a JSON configuration file, e.g, "surrogate_bathymetry_inversion_2D_config_uvWSE.json".
  - It also needs the training, validation, and test datasets (train_uv.tfrecords, validation_uv.tfrecords, test_uv.tfrecords). These files are in TensorFlow Records format. They exceed the size limit set by GitHub. They can be obtained from [CUAHSI Hydroshare](http://www.hydroshare.org/resource/357c3c413622460a91d29cc61d0ba084). 

If you just want to use the example dataset, there is no need to run other Python scripts in this folder. 

Other files for training data generation and processing. 
- "generate_tf_models.py": creates the CNN network architecture and save it as a JSON file.
- "generate_bathymetries.py": this file is where the bathymetry generation happens. A Gaussian process is used to draw random bathymetry realizations.
    - It needs a JSON configuration file, e.g., "swe_2D_training_data_generation_config".
- "generate_training_data_from_SRH-2D.py": generates training data (in TensorFlow record format).     
    - It also needs a JSON configuration file, e.g., "swe_2D_training_data_generation_config".
	- SRH-2D needs to be installed.
- "specific_utilities.py": utility functions for plotting, postprocessing, etc. 
- "swe_solvers_module.py": a special Python module to group functions interacting with SWE solvers such as SRH-2D.   