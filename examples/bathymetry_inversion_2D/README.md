# Bathymetry inversion with a deep-learning-based surrogate model

by Xiaoeng LIu

This folder contains the code for bathymetry inversion work done with a deep-learning-based surrogate model. 

## Description of files:

The main file:
- "main_train_predict_invert.py": this is the file where the training, prediction, and inversion happens. 
  - It needs a JSON configuration file, e.g, "surrogate_bathymetry_inversion_2D_config_uvWSE.json".

Example training data can be obtained from this link (to be updated once our paper is published). If you just want to use the example dataset, there is no need to run other Python scripts in this folder. 

Other files for training data generation and processing. 
- "generate_tf_models.py": creates the CNN network architecture and save it as a JSON file.
- "generate_bathymetries.py": this file is where the bathymetry generation happens. A Gaussian process is used to draw random bathymetry realizations.
    - It needs a JSON configuration file, e.g., "swe_2D_training_data_generation_config".
- "generate_training_data_from_SRH-2D.py": generates training data (in TensorFlow record format).     
    - It also needs a JSON configuration file, e.g., "swe_2D_training_data_generation_config".
- "specific_utilities.py": utility functions for plotting, postprocessing, etc. 
- "swe_solvers_module.py": a special Python module to group functions interacting with SWE solvers such as SRH-2D.   