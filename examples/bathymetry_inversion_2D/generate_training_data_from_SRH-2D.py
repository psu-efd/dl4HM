"""
This Python script does the following:
    - using Gmsh to generate a mesh: a rectangular channel with one inlet and one outlet.
    - interpolate the bathymetries to the 2D meshes: number of meshes = number of bathymetries
    - convert Gmsh meshes to SRH-2D cases
    - run all SRH-2D cases
    - convert SRH-2D cases results (u,v,h) and bed elevation (zb) to training data

What you will need:
    - installation of SRH-2D. You can download a community version of SMS from https://www.aquaveo.com/downloads-sms.
      It comes with SRH-2D.
    - pyHMT2D (install from GitHub to get the latest version; see https://github.com/psu-efd/pyHMT2D for instruction
    - some knowledge about SRH-2D which solves the shallow water equations with the finite volume method

"""

import numpy as np
from scipy import interpolate
import pygmsh
import gmsh
import vtk
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
import meshio
import json
import os
import shutil
import time

import matplotlib.ticker as tick

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

import multiprocessing

from swe_solvers_module import run_one_SRH_2D_case


from pyHMT2D.Hydraulic_Models_Data.SRH_2D import SRH_2D_Data, SRH_2D_Model, SRH_2D_SRHHydro
from pyHMT2D.Misc import gmsh2d_to_srh

np_config.enable_numpy_behavior()

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', size=18, family='serif') #specify the default font family to be "serif"

def generate_gmsh_mesh(mshFineName, bathymetry_inversion_2D_config):
    """ Generate meshes using pygmsh

    A rectangular channel.

    pygmsh can also add circle, or any other irregular shape as a hole.

    :param bathymetry_inversion_2D_config: dictionary containing information from the config json file

    Returns
    -------

    """

    print("Generate mesh with pygmsh")

    # Characteristic length (resolution)
    resolution = bathymetry_inversion_2D_config['SRH-2D cases']['Gmsh']['resolution']

    # Coordinates of lower-left and upper-right vertices of the channel domain
    xmin = bathymetry_inversion_2D_config['SRH-2D cases']['Gmsh']['xmin']
    xmax = bathymetry_inversion_2D_config['SRH-2D cases']['Gmsh']['xmax']
    ymin = bathymetry_inversion_2D_config['SRH-2D cases']['Gmsh']['ymin']
    ymax = bathymetry_inversion_2D_config['SRH-2D cases']['Gmsh']['ymax']

    #center of the rectangular hole
    #hole_center_x = 5.0
    #hole_center_y = 2.1

    #side length of the rectangular hole
    #hole_lx = 1.5
    #hole_ly = 1.2

    # Initialize empty geometry using the builtin kernel in GMSH
    geometry = pygmsh.geo.Geometry()

    # Fetch model we would like to add data to
    model = geometry.__enter__()

    # Add the rectangular hole (elevation z = 0)
    #rectangle = model.add_rectangle(hole_center_x - hole_lx / 2, hole_center_x + hole_lx / 2,
    #                               hole_center_y - hole_ly / 2, hole_center_y + hole_ly / 2,
    #                               z=0.0,
    #                               mesh_size=resolution)

    # Add points for the channel
    points = [model.add_point((xmin, ymin, 0), mesh_size=resolution),
              model.add_point((xmax, ymin, 0), mesh_size=resolution),
              model.add_point((xmax, ymax, 0), mesh_size=resolution),
              model.add_point((xmin, ymax, 0), mesh_size=resolution)]

    # Add lines between all points creating the rectangle channel
    channel_lines = [model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)]

    # Create a line loop and plane surface for meshing
    channel_loop = model.add_curve_loop(channel_lines)
    #plane_surface = model.add_plane_surface(channel_loop, holes=[rectanle.curve_loop])
    plane_surface = model.add_plane_surface(channel_loop)

    # Call gmsh kernel before add physical entities
    model.synchronize()

    # Add physcial boundaries
    # For SRH-2D, if we want to explicitly specify boundary conditions for wall,
    # we can not combine disconnected lines to one boundary. They have to be separated.
    model.add_physical([plane_surface], "channel")
    model.add_physical([channel_lines[0]], "inlet")
    model.add_physical([channel_lines[2]], "outlet")
    #model.add_physical(rectanle.curve_loop.curves, "obstacle")

    # Generate the mesh using the pygmsh
    geometry.generate_mesh(dim=2)

    # Specify the mesh version as 2 (no need for latest version)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    #write the Gmsh MSH file
    gmsh.write(mshFineName)

    #clean up and exit
    gmsh.clear()
    geometry.__exit__()

def gmsh2d_to_3d(gmsh2d_fileName, gmsh3d_fileName, bathymetry_data_file_name, ID_bathymetry):
    """
    Assign the elevation of points so the 2D mesh becomes 3D (still surface though)

    :param gmsh2d_fileName:
    :param gmsh3d_fileName:
    :param bathymetry_data_file_name: file name for the bathymetries
    :param ID_bathymetry: ID of the bathymetry in the array
    :return:
    """

    print("Assign elevation to points in Gmsh 2D mesh to make it 3D ...")

    #load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    xArray = bathymetry_data['xArray']
    yArray = bathymetry_data['yArray']
    elevation = bathymetry_data['elevation']

    #total number of bathymetry in the data
    nBathy = elevation.shape[2]

    #read in the Gmsh MSH with meshio
    mesh = meshio.read(gmsh2d_fileName)

    #output the 2D mesh as vtk for checking
    #mesh.write("check_mesh_before.vtk")

    #construct the 2d interpolator
    bilinterp = interpolate.interp2d(xArray, yArray, elevation[:,:,ID_bathymetry], kind='linear')

    #loop over all points in the mesh
    for i in range(mesh.points.shape[0]):
        mesh.points[i,2] = bilinterp(mesh.points[i,0], mesh.points[i,1])

    #output the 3D mesh as vtk for checking
    #mesh.write("check_mesh_after.vtk")

    #write out the new mesh
    mesh.write(gmsh3d_fileName, file_format="gmsh22", binary=False)

def generate_all_SRH_2D_cases(bathymetry_inversion_2D_config):
    """
    Generate all SRH-2D cases

    :return:
    """

    print("Generate all SRH-2D cases ...")

    # generate mesh using pygmsh
    gmsh2d_fileName = "twoD_channel.msh"
    generate_gmsh_mesh(gmsh2d_fileName, bathymetry_inversion_2D_config)

    # get the bathymetry data file
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']

    # get the number of bathymetries
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # get the base srhhydro file name
    srhhydro_filename = bathymetry_inversion_2D_config['SRH-2D cases']['srhhydro_filename']

    # all cases will be in the "cases" directory (make sure delete this directory before run the script)
    if os.path.isdir("cases"):
        raise Exception("The directory cases already exists. Make sure to remove it before run this script.")
    else:
        os.mkdir("cases")

    # loop over all bathymetries
    for iBathy in range(nBathy):
        print("Creating SRH-2D case ", iBathy, " out of ", nBathy)

        gmsh3d_fileName = "twoD_channel_with_elevation_" + str(iBathy) + ".msh"

        # make the elevation assignment
        gmsh2d_to_3d(gmsh2d_fileName, gmsh3d_fileName, bathymetry_data_file_name, ID_bathymetry=iBathy)

        # convert Gmsh mesh to SRH-2D
        srh_caseName = "twoD_channel_" + str(iBathy)

        gmsh2d_to_srh(gmsh3d_fileName, srh_caseName, units="Meters")

        # save the corresponding srhhydro file
        my_srh_2d_srhhydro = SRH_2D_SRHHydro(srhhydro_filename)

        # new grid file name, HydroMat file name, and srhhydro file name
        newGridFileName = srh_caseName + ".srhgeom"
        newHydroMatFileName = srh_caseName + ".srhmat"
        newSRHHydroFileName = srh_caseName + ".srhhydro"

        my_srh_2d_srhhydro.modify_Case_Name(srh_caseName)
        my_srh_2d_srhhydro.modify_Grid_FileName(newGridFileName)
        my_srh_2d_srhhydro.modify_HydroMat_FileName(newHydroMatFileName)

        # save the srhhydro file
        my_srh_2d_srhhydro.write_to_file(newSRHHydroFileName)

        # make a case directory inside "cases"
        os.mkdir("cases/case_" + str(iBathy))

        # move the three case files to the current case directory
        shutil.move(newSRHHydroFileName, "cases/case_" + str(iBathy))
        shutil.move(newHydroMatFileName, "cases/case_" + str(iBathy))
        shutil.move(newGridFileName, "cases/case_" + str(iBathy))
        shutil.move(gmsh3d_fileName, "cases/case_" + str(iBathy))

def run_SRH_2D(srh_caseName):
    """Run SRH-2D simulation

    Parameters
    ----------
    srh_caseName : str
        SRH-2D case name (without the extension .srhhydro)

    Returns
    -------

    """

    #the follow should be modified based on your installation of SRH-2D
    version = "3.3"
    srh_pre_path = r"C:\Program Files\SMS 13.1 64-bit\Python36\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
    srh_path = r"C:\Program Files\SMS 13.1 64-bit\Python36\Lib\site-packages\srh2d_exe\SRH-2D_V330_Console.exe"
    extra_dll_path = r"C:\Program Files\SMS 13.1 64-bit\Python36\Lib\site-packages\srh2d_exe"

    #create a SRH-2D model instance
    my_srh_2d_model = SRH_2D_Model(version, srh_pre_path, srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project(srh_caseName+".srhhydro")

    #run SRH-2D Pre to preprocess the case
    my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project
    bRunSucessful = my_srh_2d_model.run_model(bShowProgress=False)
    if not bRunSucessful:
        raise Exception("SRH-2D run failed.")

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()


def convert_SRH_2D_to_VTK(srh_caseName, hdf_fileName):
    """ Convert SRH-2D results to VTK

    Parameters
    ----------
    srh_caseName : str
        SRH-2D case name (without the extension .srhhydro)

    Returns
    -------

    """

    my_srh_2d_data = SRH_2D_Data(srh_caseName+".srhhydro")

    #read SRH-2D result in XMDF format (*.h5)
    #wether the XMDF result is nodal or cell center
    bNodal = False

    my_srh_2d_data.readSRHXMDFFile(hdf_fileName, bNodal)

    #export to VTK
    vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')

    return vtkFileNameList

def sample_vtk_on_grid(n_rows, n_cols, vtkFileName):
    """Sample the vtk results on a grid of n_rows x n_cols

    Returns
    -------
           vel_x, vel_y, water_depth, bInDomain: all in numpy 2D array (n_rows, n_cols)

    """

    #load the vtk file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtkFileName)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    #read the data in
    vtk_data = reader.GetOutput()

    #get the bounds of the data domain
    bounds = vtk_data.GetBounds()

    #print(bounds)

    ### flatten the vtk source data (can't do "resample to image" if it is 3D)

    # a transform for flattening
    flattener = vtk.vtkTransform()
    flattener.Scale(1.0, 1.0, 0.0)

    s_flat = vtk.vtkTransformFilter()
    s_flat.SetInputData(vtk_data)
    s_flat.SetTransform(flattener)

    # do the "resample to image" filter on the flattened vtk data
    resample = vtk.vtkResampleToImage()
    resample.SetInputConnection(s_flat.GetOutputPort())
    resample.SetSamplingDimensions(n_cols, n_rows, 1)  #make sure n_cols is x-dimension and n_rows is y-dimension
    resample.Update()

    result = resample.GetOutput()

    #get the result values (names should be adjusted depending on SI or EN units)
    zb = result.GetPointData().GetArray("Bed_Elev_m")
    velocity = result.GetPointData().GetArray("Velocity_m_p_s")
    water_depth = result.GetPointData().GetArray("Water_Depth_m")
    WSE = result.GetPointData().GetArray("Water_Elev_m")
    bInDomain = result.GetPointData().GetArray("vtkValidPointMask") #this is the mask for in or out of domain (1 is in and 0 is out)

    #convert to numpy array and reshape to 2D array (n_rows x n_cols)
    zb = VN.vtk_to_numpy(zb).reshape(n_rows, n_cols)

    velocity = VN.vtk_to_numpy(velocity)

    vel_x = velocity[:, 0].reshape(n_rows, n_cols)
    vel_y = velocity[:, 1].reshape(n_rows, n_cols)

    water_depth = VN.vtk_to_numpy(water_depth).reshape(n_rows, n_cols)
    WSE = VN.vtk_to_numpy(WSE).reshape(n_rows, n_cols)
    bInDomain = VN.vtk_to_numpy(bInDomain).reshape(n_rows, n_cols)

    return zb, vel_x, vel_y, water_depth, WSE, bInDomain, bounds

def run_all_SRH_2D_cases_parallel(bathymetry_inversion_2D_config, nProcess):
    """
    Run all cases in parallel using multiple CPU cores. Each core
    simulates one case at a time.

    Steps:
    1. create python multiprocessing envinroment
    2. (manually) assign simulcate case ranges to each process
    3. each process runs their assigned cases
    3. all processes join back and done.


    Parameters
    ----------
    bathymetry_inversion_2D_config

    Returns
    -------

    """

    print("Run SRH-2D cases in parallel with ", nProcess, " processes.")

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=nProcess)

    caseIDs = [*range(2000, nBathy, 1)]
    #caseIDs = [*range(0, 4, 1)]

    outputs = pool.map(run_one_SRH_2D_case, caseIDs)

    #save the outputs as a json file
    with open('parallel_run_srh2d_outputs.json', 'w') as f:
        json.dump(outputs, f)

    print("Parallel run of cases done. Outputs = ", outputs)


def run_all_SRH_2D_cases(bathymetry_inversion_2D_config, bRange=False, iStart=0, iEnd=0):
    """
    Run all SRH-2D cases.

    Optionally, the caller can specify the range of cases to run (this
    is for parallel processing).

    Parameters
    ----------
    bathymetry_inversion_2D_config
    bRange: bool. Whether set the range of cases. Default is false
    iStart: int. Starting ID of cases
    iEnd: int. Ending ID of cases

    :return:
    """

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    if not bRange:
        print("Run SRH-2D cases without specifying a range (default). All cases will be ran ...")
        iStart = 0
        iEnd = nBathy
    else:
        #make sure the specified range makes sense
        if iStart >= iEnd:
            raise Exception("Specified iStart = ", iStart, " is larger or equal to iEnd = ", iEnd)

        if iStart < 0 or iEnd > nBathy:
            raise Exception("Specified iStart or iEnd is out of range.")

        print("Run SRH-2D cases with the specified range of ", iStart, " to ", iEnd)

    # loop over all bathymetries (cases)
    for iBathy in range(iStart,iEnd):
        print("Running SRH-2D case: ", iBathy, "out of", nBathy-1)

        #go into the case's directory
        os.chdir("./cases/case_" + str(iBathy))

        srh_caseName = "twoD_channel_" + str(iBathy)

        #run the current case
        run_SRH_2D(srh_caseName)

        #go back to the root
        os.chdir("../..")

def sample_all_SRH_2D_cases(bathymetry_inversion_2D_config):
    """
    Sample (convert) all SRH-2D cases results to training data

    :return:
    """

    print("Sample all SRH-2D cases ...")

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # amplitudes of result variables for all cases (for normalization purpose)
    zb_min = 1E6
    zb_max = -1E6
    vel_x_min = 1E6
    vel_x_max = -1E6
    vel_y_min = 1E6
    vel_y_max = -1E6
    water_depth_min = 1E6
    water_depth_max = -1E6
    WSE_min = 1E6
    WSE_max = -1E6

    slope_x_min = 1E6
    slope_x_max = -1E6
    slope_y_min = 1E6
    slope_y_max = -1E6

    # loop over all bathymetries (cases)
    for iBathy in range(nBathy):
        print("Sampling SRH-2D case: ", iBathy, "out of", nBathy-1)

        #go into the case's directory
        os.chdir("./cases/case_" + str(iBathy))

        srh_caseName = "twoD_channel_" + str(iBathy)

        # convert SRH-2D result to VTK
        hdf_fileName = srh_caseName + "_XMDFC.h5"
        vtkFileNameList = convert_SRH_2D_to_VTK(srh_caseName, hdf_fileName)

        # print the VTK file name list. It should only have one file name in it
        # because by default we only export the last time step
        print(vtkFileNameList)

        # sample the VTK result on grid

        # get number of rows (in y-direction)
        n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
        # get number of colmuns (in x-direction)
        n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

        zb, vel_x, vel_y, water_depth, WSE, bInDomain, bounds = \
            sample_vtk_on_grid(n_rows, n_cols, vtkFileNameList[-1])

        zb_min = np.min([zb_min, np.min(zb)])
        zb_max = np.max([zb_max, np.max(zb)])
        vel_x_min = np.min([vel_x_min, np.min(vel_x)])
        vel_x_max = np.max([vel_x_max, np.max(vel_x)])
        vel_y_min = np.min([vel_y_min, np.min(vel_y)])
        vel_y_max = np.max([vel_y_max, np.max(vel_y)])

        water_depth_min = np.min([water_depth_min, np.min(water_depth)])
        water_depth_max = np.max([water_depth_max, np.max(water_depth)])

        WSE_min = np.min([WSE_min, np.min(WSE)])
        WSE_max = np.max([WSE_max, np.max(WSE)])

        dzb_dx = (np.diff(zb, axis=1))
        dzb_dy = (np.diff(zb, axis=0))

        if slope_x_min > dzb_dx.min():
            slope_x_min = dzb_dx.min()

        if slope_x_max < dzb_dx.max():
            slope_x_max = dzb_dx.max()

        if slope_y_min > dzb_dy.min():
            slope_y_min = dzb_dy.min()

        if slope_y_max < dzb_dy.max():
            slope_y_max = dzb_dy.max()

        #save the sampled result to file
        np.savez(srh_caseName+".npz", n_rows=n_rows, ncols=n_cols, zb = zb, vel_x=vel_x, vel_y=vel_y,
                 water_depth=water_depth, WSE=WSE, bInDomain=bInDomain)

        #make a copy of the sampled result file to "cases" directory for convience
        shutil.copy(srh_caseName +".npz", "../")

        #go back to the root
        os.chdir("../..")

    #dictionary to hold all varialbe min and max values
    varialbes_min_max = {}
    varialbes_min_max['zb_min'] = zb_min
    varialbes_min_max['zb_max'] = zb_max
    varialbes_min_max['vel_x_min'] = vel_x_min
    varialbes_min_max['vel_x_max'] = vel_x_max
    varialbes_min_max['vel_y_min'] = vel_y_min
    varialbes_min_max['vel_y_max'] = vel_y_max

    varialbes_min_max['water_depth_min'] = water_depth_min
    varialbes_min_max['water_depth_max'] = water_depth_max

    varialbes_min_max['WSE_min'] = WSE_min
    varialbes_min_max['WSE_max'] = WSE_max

    varialbes_min_max['bounds'] = bounds

    varialbes_min_max['slope_x_min'] = slope_x_min
    varialbes_min_max['slope_x_max'] = slope_x_max
    varialbes_min_max['slope_y_min'] = slope_y_min
    varialbes_min_max['slope_y_max'] = slope_y_max

    print("bed slope range in x and y: ", slope_x_min, slope_x_max, slope_y_min, slope_y_max)

    #save the min/max dictionary to JSON file
    varialbes_min_max_file_name = bathymetry_inversion_2D_config['SRH-2D cases']['varialbes_min_max_file_name']
    with open(varialbes_min_max_file_name, "w") as outfile:
        json.dump(varialbes_min_max, outfile, indent=4)

def createTFRecords(bathymetry_inversion_2D_config):
    """
    Create TFRecord for training data

    Reference:
    https://github.com/loliverhennigh/Steady-State-Flow-With-Neural-Nets/blob/master/utils/createTFRecords.py


    :param bathymetry_inversion_2D_config:
    :return:
    """

    # helper functions: to convert a value to a type compatible with tf.train.Example
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_array(array):
        array = tf.io.serialize_tensor(array)
        return array

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # get number of rows (in y-direction)
    n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
    # get number of colmuns (in x-direction)
    n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

    # get min and max of all result variables
    varialbes_min_max_file_name = bathymetry_inversion_2D_config['SRH-2D cases']['varialbes_min_max_file_name']

    with open(varialbes_min_max_file_name) as json_file:
        variables_min_max = json.load(json_file)

    # min and max of variables
    zb_min = variables_min_max['zb_min']
    zb_max = variables_min_max['zb_max']
    vel_x_min = variables_min_max['vel_x_min']
    vel_x_max = variables_min_max['vel_x_max']
    vel_y_min = variables_min_max['vel_y_min']
    vel_y_max = variables_min_max['vel_y_max']

    WSE_min = variables_min_max['WSE_min']
    WSE_max = variables_min_max['WSE_max']

    # fractions for training data, validation data, and test data
    training_fraction = bathymetry_inversion_2D_config['data generation']['training_fraction']
    validation_fraction = bathymetry_inversion_2D_config['data generation']['validation_fraction']
    test_fraction = bathymetry_inversion_2D_config['data generation']['test_fraction']

    if abs(training_fraction+validation_fraction+test_fraction - 1.0) > 0.0001:
        raise Exception("The specified training, validation, and test fractions do not sum up to 1.0")

    # create tf writers

    # read the bool for whether to export (u, v) only or (u, v, WSE)
    b_uv_only = bathymetry_inversion_2D_config['data generation']['uv_only']

    if b_uv_only: # if only exporting (u,v)
        training_record_filename = 'train_uv.tfrecords'
        validation_record_filename = 'validation_uv.tfrecords'
        test_record_filename = 'test_uv.tfrecords'

        print("Creating TFRecords files with (u, v). And the records files are: ", training_record_filename,
              validation_record_filename, test_record_filename)

    else:         # if exporting (u, v, WSE)
        training_record_filename = 'train_uvWSE.tfrecords'
        validation_record_filename = 'validation_uvWSE.tfrecords'
        test_record_filename = 'test_uvWSE.tfrecords'

        print("Creating TFRecords files with (u, v, WSE). And the records files are: ", training_record_filename,
              validation_record_filename, test_record_filename)

    training_writer = tf.io.TFRecordWriter(training_record_filename)
    validation_writer = tf.io.TFRecordWriter(validation_record_filename)
    test_writer = tf.io.TFRecordWriter(test_record_filename)

    # seperating index among training, validation and test record
    iTraining_validation = int(nBathy*training_fraction)
    iValidation_test = int(nBathy*(training_fraction+validation_fraction))

    print("The split among ", nBathy, "data sets for training, validation, and test: ")
    print("\t training: 0 to ", iTraining_validation - 1)
    print("\t validation: ", iTraining_validation, " to ", iValidation_test - 1)
    print("\t test: ", iValidation_test, " to ", nBathy - 1)

    # randomly shuffle the cases
    caseIDs_list = [*range(nBathy)]

    rng = np.random.default_rng()
    rng.shuffle(caseIDs_list, axis=0)

    #print(caseIDs_list)

    # loop over all bathymetries (cases)
    for iBathy, caseID in zip(range(nBathy), caseIDs_list):
        print("Processing SRH-2D case: ", iBathy, "out of", nBathy - 1, ", using caseID = ", caseID)

        data_fileName = "./cases/twoD_channel_" + str(caseID) + ".npz"

        data = np.load(data_fileName)

        #unpack the results
        zb = data['zb']
        vel_x = data['vel_x']
        vel_y = data['vel_y']
        WSE = data['WSE']

        #normalize to [-0.5, 0.5]
        zb_norm = (zb - zb_min)/(zb_max - zb_min) - 0.5
        WSE_norm = (WSE - WSE_min) / (WSE_max - WSE_min) - 0.5

        vel_x_norm = (vel_x - vel_x_min) / (vel_x_max - vel_x_min) - 0.5  #normalization is component-wise
        vel_y_norm = (vel_y - vel_y_min) / (vel_y_max - vel_y_min) - 0.5

        if b_uv_only:  # if only exporting (u,v)
            vel_WSE_norm = np.dstack([vel_x_norm, vel_y_norm ])  # stack x and y velocity to form 3D array
        else:
            vel_WSE_norm = np.dstack([vel_x_norm, vel_y_norm, WSE_norm])  # stack x and y velocity to form 3D array

        #expand one more dimension to the zb array, e.g., shape=[21, 121] to shape=[21, 121, 1]
        zb_norm = zb_norm[:,:,np.newaxis]

        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
            'iBathy': _int64_feature(caseID),
            'zb':  _bytes_feature(serialize_array(zb_norm)),
            'vel_WSE': _bytes_feature(serialize_array(vel_WSE_norm))}))

        if iBathy < iTraining_validation:
            training_writer.write(example.SerializeToString())
        elif iBathy < iValidation_test:
            validation_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

    #save the IDs in training, validation and test dataset
    IDs_seperation = {}
    IDs_seperation['training'] = caseIDs_list[0:iTraining_validation]
    IDs_seperation['validation'] = caseIDs_list[iTraining_validation:iValidation_test]
    IDs_seperation['test'] = caseIDs_list[iValidation_test:]

    # save the IDs dictionary to JSON file
    with open("caseIDs_in_train_validation_test.json", "w") as outfile:
        json.dump(IDs_seperation, outfile, indent=4)


def createTFRecords_with_specified_IDs(bathymetry_inversion_2D_config):
    """
    Create TFRecord for training data with specified case IDs in each group (training, validation, test)

    Reference:
    https://github.com/loliverhennigh/Steady-State-Flow-With-Neural-Nets/blob/master/utils/createTFRecords.py


    :param bathymetry_inversion_2D_config:
    :return:
    """

    with open('caseIDs_in_train_validation_test.json') as json_file:
        IDs_seperation = json.load(json_file)

    IDs_training = IDs_seperation['training']
    IDs_validation = IDs_seperation['validation']
    IDs_test = IDs_seperation['test']

    # helper functions: to convert a value to a type compatible with tf.train.Example
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_array(array):
        array = tf.io.serialize_tensor(array)
        return array

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # get number of rows (in y-direction)
    n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
    # get number of colmuns (in x-direction)
    n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

    # get min and max of all result variables
    varialbes_min_max_file_name = bathymetry_inversion_2D_config['SRH-2D cases']['varialbes_min_max_file_name']

    with open(varialbes_min_max_file_name) as json_file:
        variables_min_max = json.load(json_file)

    # min and max of variables
    zb_min = variables_min_max['zb_min']
    zb_max = variables_min_max['zb_max']
    vel_x_min = variables_min_max['vel_x_min']
    vel_x_max = variables_min_max['vel_x_max']
    vel_y_min = variables_min_max['vel_y_min']
    vel_y_max = variables_min_max['vel_y_max']

    WSE_min = variables_min_max['WSE_min']
    WSE_max = variables_min_max['WSE_max']

    # read the bool for whether to export (u, v) only or (u, v, WSE)
    b_uv_only = bathymetry_inversion_2D_config['data generation']['uv_only']

    if b_uv_only: # if only exporting (u,v)
        training_record_filename = 'train_uv.tfrecords'
        validation_record_filename = 'validation_uv.tfrecords'
        test_record_filename = 'test_uv.tfrecords'

        print("Creating TFRecords files with (u, v). And the records files are: ", training_record_filename,
              validation_record_filename, test_record_filename)

    else:         # if exporting (u, v, WSE)
        training_record_filename = 'train_uvWSE.tfrecords'
        validation_record_filename = 'validation_uvWSE.tfrecords'
        test_record_filename = 'test_uvWSE.tfrecords'

        print("Creating TFRecords files with (u, v, WSE). And the records files are: ", training_record_filename,
              validation_record_filename, test_record_filename)

    training_writer = tf.io.TFRecordWriter(training_record_filename)
    validation_writer = tf.io.TFRecordWriter(validation_record_filename)
    test_writer = tf.io.TFRecordWriter(test_record_filename)

    # loop over all bathymetries (cases)
    for iBathy in range(nBathy):
        print("Processing SRH-2D case: ", iBathy, "out of", nBathy - 1)

        data_fileName = "./cases/twoD_channel_" + str(iBathy) + ".npz"

        data = np.load(data_fileName)

        #unpack the results
        zb = data['zb']
        vel_x = data['vel_x']
        vel_y = data['vel_y']
        WSE = data['WSE']

        #normalize to [-0.5, 0.5]
        zb_norm = (zb - zb_min)/(zb_max - zb_min) - 0.5
        WSE_norm = (WSE - WSE_min) / (WSE_max - WSE_min) - 0.5

        vel_x_norm = (vel_x - vel_x_min) / (vel_x_max - vel_x_min) - 0.5  #normalization is component-wise
        vel_y_norm = (vel_y - vel_y_min) / (vel_y_max - vel_y_min) - 0.5

        if b_uv_only:  # if only exporting (u,v)
            vel_WSE_norm = np.dstack([vel_x_norm, vel_y_norm ])  # stack x and y velocity to form 3D array
        else:
            vel_WSE_norm = np.dstack([vel_x_norm, vel_y_norm, WSE_norm])  # stack x and y velocity to form 3D array

        #expand one more dimension to the zb array, e.g., shape=[21, 121] to shape=[21, 121, 1]
        zb_norm = zb_norm[:,:,np.newaxis]

        # create example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
            'iBathy': _int64_feature(iBathy),
            'zb':  _bytes_feature(serialize_array(zb_norm)),
            'vel_WSE': _bytes_feature(serialize_array(vel_WSE_norm))}))

        if iBathy in IDs_training:
            training_writer.write(example.SerializeToString())
        elif iBathy in IDs_validation:
            validation_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

def checkTFRecords(record_filename, bathymetry_inversion_2D_config, nPlotSamples=1):
    """
    Check the TFRecord

    :param bathymetry_inversion_2D_config:
    :param nPlotSamples: int
        number of samples to plot. If -1, plot all in the records. Default = 1
    :return:
    """

    # Set up our dataset
    global max
    dataset = tf.data.TFRecordDataset(record_filename)

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # get number of rows (in y-direction)
    n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
    # get number of colmuns (in x-direction)
    n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

    def parse_flow_data(serialized_example):
        features = {
            'iBathy': tf.io.FixedLenFeature([], tf.int64),
            'zb': tf.io.FixedLenFeature([], tf.string),
            'vel_WSE': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        iBathy = parsed_features['iBathy']

        zb = parsed_features['zb']  # get byte string
        zb = tf.io.parse_tensor(zb, out_type=tf.float64)  # restore 2D array from byte string

        vel_WSE = parsed_features['vel_WSE']  # get byte string
        vel_WSE = tf.io.parse_tensor(vel_WSE, out_type=tf.float64)  # restore 2D array from byte string

        return iBathy, zb, vel_WSE

    # Transform binary data into image arrays
    dataset = dataset.map(parse_flow_data)

    # IDs in the recrod file
    IDs = []

    # check whether vel_WSE contains WSE or it is (u,v) only
    b_uv_only = True

    dzb_dx_max = -1E6
    dzb_dy_max = -1E6

    for record in dataset:
        ID, zb, vel_WSE = record
        IDs.append(ID.numpy())

        if vel_WSE.numpy().shape[2] == 3: # This in fact only needs to be checked once
            b_uv_only = False

        # check whether the dimensions of the result data are compatible with the config file
        if n_rows != vel_WSE.numpy().shape[0] or n_cols != vel_WSE.numpy().shape[1]:
            raise Exception("The dimensions of the data in config file and TFRecords files are not compatible. "
                            "In config file: n_rows, n_cols = ", n_rows, n_cols,
                            "In TFRecords files: n_rows, n_cols = ", vel_WSE.numpy().shape[0], vel_WSE.numpy().shape[1])

        #calculate the "slope" maximums
        dzb_dx_max = max(tf.math.reduce_max(tf.math.abs(tf.experimental.numpy.diff(zb, axis=1))).numpy(), dzb_dx_max)
        dzb_dy_max = max(tf.math.reduce_max(tf.math.abs(tf.experimental.numpy.diff(zb, axis=0))).numpy(), dzb_dy_max)

    print("There are total of ", len(IDs), " records in the dataset. They are: ", IDs)

    print("maximum absolutute slopes in x and y are: ", dzb_dx_max, dzb_dy_max)

    # if WSE data is not in, create a WSE with zero values
    if b_uv_only:
        WSE = np.zeros((n_rows, n_cols))

    # randonly draw cases
    nExamples = 0
    if nPlotSamples == -1:
        nExamples = len(IDs)
    else:
        if nPlotSamples <= len(IDs):
            nExamples = nPlotSamples
        else:
            raise Exception("The specified number of plots nPlotSamples, ", nPlotSamples, " is larger than the number of records.")

    choices = np.sort(np.random.choice(IDs, size=nExamples, replace=False))

    print("Chosen case IDs for plotting: ", choices)

    fig, axs = plt.subplots(2, 2, figsize=(2 * 10, 2 * 2), sharex=True, sharey=True, facecolor='w',
                            edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.01)

    counter = 0

    min = -0.5
    max = 0.5
    levels = np.linspace(min, max, 51)
    ticks = np.linspace(min, max, 7)

    # loop over all records
    for record in dataset:
        ID, zb, vel_WSE = record

        # if the current case is in the chosen list for plotting
        if ID.numpy() in choices:
            counter = counter + 1

            print("Plotting ID =", ID.numpy(), ",", counter, "out of", len(IDs))

            # a list to contain all colorbars (to be cleared at the end)
            clbs = []

            # plot zb
            cf_zb = axs[0, 0].contourf(np.squeeze(zb[:, :, :]), levels, vmin=min, vmax=max, cmap=plt.cm.terrain)
            axs[0, 0].set_xlim([0, n_cols])
            axs[0, 0].set_ylim([0, n_rows])
            axs[0, 0].set_aspect('equal')
            axs[0, 0].set_title("Bed elevation", fontsize=14)
            clb_zb = fig.colorbar(cf_zb, ticks=ticks, ax=axs[0, 0])
            clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
            clb_zb.ax.tick_params(labelsize=12)
            clbs.append(clb_zb)

            # plot WSE
            if b_uv_only:    # if the TFRecords only have u and v, use zero WSE
                cf_vel_x = axs[0, 1].contourf(WSE, levels, vmin=min, vmax=max,
                                              cmap=plt.cm.jet)
                axs[0, 1].set_xlim([0, n_cols])
                axs[0, 1].set_ylim([0, n_rows])
                axs[0, 1].set_aspect('equal')
                axs[0, 1].set_title("WSE (zeros, not in dataset)", fontsize=14)
                clb_WSE = fig.colorbar(cf_vel_x, ticks=ticks, ax=axs[0, 1])
                clb_WSE.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                clb_WSE.ax.tick_params(labelsize=12)
                clbs.append(clb_WSE)
            else:
                cf_vel_x = axs[0, 1].contourf(np.squeeze(vel_WSE[:, :, 2]), levels, vmin=min, vmax=max,
                                              cmap=plt.cm.jet)
                axs[0, 1].set_xlim([0, n_cols])
                axs[0, 1].set_ylim([0, n_rows])
                axs[0, 1].set_aspect('equal')
                axs[0, 1].set_title("WSE", fontsize=14)
                clb_WSE = fig.colorbar(cf_vel_x, ticks=ticks, ax=axs[0, 1])
                clb_WSE.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
                clb_WSE.ax.tick_params(labelsize=12)
                clbs.append(clb_WSE)

            # plot vel_x
            cf_vel_x = axs[1, 0].contourf(np.squeeze(vel_WSE[:, :, 0]), levels, vmin=min, vmax=max,
                                          cmap=plt.cm.jet)
            axs[1, 0].set_xlim([0, n_cols])
            axs[1, 0].set_ylim([0, n_rows])
            axs[1, 0].set_aspect('equal')
            axs[1, 0].set_title("x-velocity", fontsize=14)
            clb_vel_x = fig.colorbar(cf_vel_x, ticks=ticks, ax=axs[1, 0])
            clb_vel_x.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
            clb_vel_x.ax.tick_params(labelsize=12)
            clbs.append(clb_vel_x)

            # plot vel_y
            cf_vel_y = axs[1, 1].contourf(np.squeeze(vel_WSE[:, :, 1]), levels, vmin=min, vmax=max,
                                          cmap=plt.cm.jet)
            axs[1, 1].set_xlim([0, n_cols])
            axs[1, 1].set_ylim([0, n_rows])
            axs[1, 1].set_aspect('equal')
            axs[1, 1].set_title("y-velocity", fontsize=14)
            clb_vel_y = fig.colorbar(cf_vel_y, ticks=ticks, ax=axs[1, 1])
            clb_vel_y.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
            clb_vel_y.ax.tick_params(labelsize=12)
            clbs.append(clb_vel_y)

            # set labels
            plt.setp(axs[-1, :], xlabel='x')
            plt.setp(axs[:, 0], ylabel='y')

            plt.savefig("./plots/srh_2d_results_from_tfrecrod_" + str(ID.numpy()).zfill(4) + ".png", dpi=300, bbox_inches='tight', pad_inches=0)

            #plt.show()

            # clear all subplots
            for axs_1 in axs:
                for axs_2 in axs_1:
                    axs_2.clear()

            # remove all created colorbars
            for clb in clbs:
                clb.remove()

    plt.close(fig)


def plot_one_example_result(ID, example_results, variables_min_max):
    """
    Plot one example result

    :param example_results:
    :return:
    """

    #unpack the result
    zb = example_results['zb']
    vel_x=example_results['vel_x']
    vel_y=example_results['vel_y']
    WSE=example_results['WSE']

    # min and max of variables
    zb_min = variables_min_max['zb_min']
    zb_max = variables_min_max['zb_max']
    vel_x_min = variables_min_max['vel_x_min']
    vel_x_max = variables_min_max['vel_x_max']
    vel_y_min = variables_min_max['vel_y_min']
    vel_y_max = variables_min_max['vel_y_max']

    water_depth_min = variables_min_max['water_depth_min']
    water_depth_max = variables_min_max['water_depth_max']

    WSE_min = variables_min_max['WSE_min']
    WSE_max = variables_min_max['WSE_max']

    bounds = variables_min_max['bounds']

    ny, nx = zb.shape[0], zb.shape[1]

    xArray = np.linspace(bounds[0], bounds[1], nx)
    yArray = np.linspace(bounds[2], bounds[3], ny)

    fig, axs = plt.subplots(2, 2, figsize=(2*10, 2*2), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.01)

    # plot zb
    levels = np.linspace(zb_min, zb_max, 51)
    cf_zb = axs[0,0].contourf(xArray, yArray, zb, levels, vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain)
    axs[0,0].set_xlim([bounds[0], bounds[1]])
    axs[0,0].set_ylim([bounds[2], bounds[3]])
    axs[0,0].set_aspect('equal')
    axs[0,0].set_title("Bed elevation (m)", fontsize=14)
    clb_zb = fig.colorbar(cf_zb, ticks=np.linspace(zb_min, zb_max, 7), ax=axs[0,0])
    clb_zb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_zb.ax.tick_params(labelsize=12)
    #clb_zb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # plot WSE
    levels = np.linspace(WSE_min, WSE_max, 51)
    cf_WSE = axs[0, 1].contourf(xArray, yArray, WSE, levels, vmin=WSE_min, vmax=WSE_max, cmap=plt.cm.jet)
    axs[0, 1].set_xlim([bounds[0], bounds[1]])
    axs[0, 1].set_ylim([bounds[2], bounds[3]])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("WSE (m)", fontsize=14)
    clb_WSE = fig.colorbar(cf_WSE, ticks=np.linspace(WSE_min, WSE_max, 7), ax=axs[0, 1])
    clb_WSE.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb_WSE.ax.tick_params(labelsize=12)
    #clb_WSE.set_label('WSE (m)', labelpad=0.3, fontsize=24)

    # plot vel_x
    levels = np.linspace(vel_x_min, vel_x_max, 51)
    cf_vel_x = axs[1, 0].contourf(xArray, yArray, vel_x, levels, vmin=vel_x_min, vmax=vel_x_max, cmap=plt.cm.jet)
    axs[1, 0].set_xlim([bounds[0], bounds[1]])
    axs[1, 0].set_ylim([bounds[2], bounds[3]])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("x-velocity (m/s)", fontsize=14)
    clb_vel_x = fig.colorbar(cf_vel_x, ticks=np.linspace(vel_x_min, vel_x_max, 7), ax=axs[1, 0])
    clb_vel_x.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_x.ax.tick_params(labelsize=12)
    #clb_vel_x.set_label('Ux (m/s)', labelpad=0.3, fontsize=24)

    # plot vel_y
    levels = np.linspace(vel_y_min, vel_y_max, 51)
    cf_vel_y = axs[1, 1].contourf(xArray, yArray, vel_y, levels, vmin=vel_y_min, vmax=vel_y_max, cmap=plt.cm.jet)
    axs[1, 1].set_xlim([bounds[0], bounds[1]])
    axs[1, 1].set_ylim([bounds[2], bounds[3]])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("y-velocity (m/s)", fontsize=14)
    clb_vel_y = fig.colorbar(cf_vel_y, ticks=np.linspace(vel_y_min, vel_y_max, 7), ax=axs[1, 1])
    clb_vel_y.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb_vel_y.ax.tick_params(labelsize=12)
    #clb_vel_y.set_label('Uy (m/s)', labelpad=0.3, fontsize=24)

    # set labels
    plt.setp(axs[-1, :], xlabel='x (m)')
    plt.setp(axs[:, 0], ylabel='y (m)')

    plot_file_name = "sample_srh_2d_results_"+str(ID).zfill(4)+".png"

    plt.savefig(plot_file_name, dpi=300, bbox_inches='tight', pad_inches=0)

    # move the figure to the root directory
    shutil.copy(plot_file_name, "../plots/")

    #plt.show()

def plot_example_results(bathymetry_inversion_2D_config, nExamples=4):
    """
    Make plots for some example SRH-2D cases

    :param bathymetry_inversion_2D_config:
    :param nExamples: int
        number of examples to plot. if -1, plot all. Default = 4 (randomly chosen)
    :return:
    """

    print("Plot example SRH-2D results ...")

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # randonly draw (nrows * ncolumns) cases from all SRH-2D cases
    if nExamples == -1:
        nExamples = nBathy

    choices = np.sort(np.random.choice(nBathy, size=nExamples, replace=False))

    #hack:
    choices[0] = 493

    # get min and max of all result variables
    varialbes_min_max_file_name = bathymetry_inversion_2D_config['SRH-2D cases']['varialbes_min_max_file_name']

    with open(varialbes_min_max_file_name) as json_file:
        variables_min_max = json.load(json_file)

    # go into the case's directory
    os.chdir("cases")

    # make "plots" directory if not existing yet
    if not os.path.exists("../plots"):
        os.mkdir("../plots")

    #debug
    id_min_vel_y = -1
    min_vel_y = 1E6

    # loop over the chosen cases to gather results and plot
    for choice in choices:
        print("Plotting results for case: ", choice)

        example_results = {}   #an dictionary to hold results for the current case

        result_fileName = "twoD_channel_" + str(choice)+".npz"

        result = np.load(result_fileName)

        # add bathymetry, vel_x, vel_y, and WSE
        example_results['zb'] = result['zb']
        example_results['vel_x'] = result['vel_x']
        example_results['vel_y'] = result['vel_y']
        example_results['WSE'] = result['WSE']

        if min_vel_y > np.min(result['vel_y']):
            id_min_vel_y = choice
            min_vel_y = np.min(result['vel_y'])

        # call the plot function
        plot_one_example_result(choice, example_results, variables_min_max)

    print("Minimum vel_y happends at ", id_min_vel_y, " with min value of ", min_vel_y )

    # go back to the root
    os.chdir("../")

def generate_inversion_data(record_filename, bathymetry_inversion_2D_config, nInversionCases=1):
    """
    Genearte inversion data from TFRecord file

    :param bathymetry_inversion_2D_config:
    :param nInversionCases: int
        number of inversion cases to generate. If -1, general all in the records. Default = 1
    :return:
    """

    # Set up our dataset
    dataset = tf.data.TFRecordDataset(record_filename)

    # get the number of bathymetries (= number of cases)
    nBathy = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # get number of rows (in y-direction)
    n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
    # get number of colmuns (in x-direction)
    n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

    def parse_flow_data(serialized_example):
        features = {
            'iBathy': tf.io.FixedLenFeature([], tf.int64),
            'zb': tf.io.FixedLenFeature([], tf.string),
            'vel_WSE': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        iBathy = parsed_features['iBathy']

        zb = parsed_features['zb']  # get byte string
        zb = tf.io.parse_tensor(zb, out_type=tf.float64)  # restore 2D array from byte string

        vel_WSE = parsed_features['vel_WSE']  # get byte string
        vel_WSE = tf.io.parse_tensor(vel_WSE, out_type=tf.float64)  # restore 2D array from byte string

        return iBathy, zb, vel_WSE

    # Transform binary data into image arrays
    dataset = dataset.map(parse_flow_data)

    # IDs in the recrod file
    IDs = []

    # check whether vel_WSE contains WSE or it is (u,v) only
    b_uv_only = True

    for record in dataset:
        ID, zb, vel_WSE = record
        IDs.append(ID.numpy())

        if vel_WSE.numpy().shape[2] == 3: # This in fact only needs to be checked once
            b_uv_only = False

        # check whether the dimensions of the result data are compatible with the config file
        if n_rows != vel_WSE.numpy().shape[0] or n_cols != vel_WSE.numpy().shape[1]:
            raise Exception("The dimensions of the data in config file and TFRecords files are not compatible. "
                            "In config file: n_rows, n_cols = ", n_rows, n_cols,
                            "In TFRecords files: n_rows, n_cols = ", vel_WSE.numpy().shape[0], vel_WSE.numpy().shape[1])

    print("There are total of ", len(IDs), " records in the dataset. They are: ", IDs)

    # randonly draw cases
    nExamples = 0
    if nInversionCases == -1:
        nExamples = len(IDs)
    else:
        if nInversionCases <= len(IDs):
            nExamples = nInversionCases
        else:
            raise Exception("The specified number of cases nInversionCases, ", nInversionCases, " is larger than the number of records.")

    choices = np.sort(np.random.choice(IDs, size=nExamples, replace=False))

    #hack: force the choice
    #if 145 not in choices:
    #    choices[0] = 145

    print("Chosen case IDs for generating inversion cases: ", choices)

    counter = 0

    # loop over all records
    for record in dataset:
        ID, zb, vel_WSE = record

        # if the current case is in the chosen list for creating inversion case
        if ID.numpy() in choices:
            counter = counter + 1

            zb = zb.numpy()
            uvWSE=vel_WSE.numpy()

            print("Generating inversion case ID =", ID.numpy(), ",", counter, "out of", len(IDs))

            if b_uv_only:
                np.savez("./inversion_case_uv_" + str(ID.numpy()).zfill(4) + ".npz", zb=zb, uvWSE=uvWSE)
            else:
                np.savez("./inversion_case_uvWSE_" + str(ID.numpy()).zfill(4) + ".npz", zb=zb, uvWSE=uvWSE)


def generate_subset_masks(bathymetry_inversion_2D_config):
    """
    Genearte subset masks for inversion (the masks indicate the subset of (u,v,WSE) used in inversion.

    :param bathymetry_inversion_2D_config:

    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']

    # load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    xArray = bathymetry_data['xArray']
    yArray = bathymetry_data['yArray']
    X, Y = np.meshgrid(xArray, yArray)

    # get number of rows (in y-direction)
    n_rows = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_rows']
    # get number of colmuns (in x-direction)
    n_cols = bathymetry_inversion_2D_config['SRH-2D cases']['sample_result_n_cols']

    # we need cases for selecting 20, 40, 60, 80%
    samples = 2   #either 0 or 1

    masks_80 = np.random.choice(samples, size=[n_rows, n_cols], replace=True, p=[0.2, 0.8])
    masks_60 = np.random.choice(samples, size=[n_rows, n_cols], replace=True, p=[0.4, 0.6])
    masks_40 = np.random.choice(samples, size=[n_rows, n_cols], replace=True, p=[0.6, 0.4])
    masks_20 = np.random.choice(samples, size=[n_rows, n_cols], replace=True, p=[0.8, 0.2])
    masks_05 = np.random.choice(samples, size=[n_rows, n_cols], replace=True, p=[0.9999, 0.0001])

    #check
    print("masks_80: ", masks_80.sum()/(n_rows*n_cols))
    print("masks_05: ", masks_05.sum() / (n_rows * n_cols))

    #plot for visualization
    fig, axs = plt.subplots(2, 2, figsize=(8, 2.5), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.0, wspace=.1)

    cmap = plt.cm.get_cmap('gray', 2)  # 2 discrete colors; PiYG, viridis

    cf = axs[0, 0].contourf(X, Y, masks_05, cmap=cmap)
    axs[0, 0].set_xlim([np.min(X), np.max(X)])
    axs[0, 0].set_ylim([np.min(Y), np.max(Y)])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title("20 \%", fontsize=12)

    cf = axs[0, 1].contourf(X, Y, masks_40, cmap=cmap)
    axs[0, 1].set_xlim([np.min(X), np.max(X)])
    axs[0, 1].set_ylim([np.min(Y), np.max(Y)])
    axs[0, 1].set_aspect('equal')
    axs[0, 1].set_title("40 \%", fontsize=12)

    cf = axs[1, 0].contourf(X, Y, masks_60, cmap=cmap)
    axs[1, 0].set_xlim([np.min(X), np.max(X)])
    axs[1, 0].set_ylim([np.min(Y), np.max(Y)])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("60 \%", fontsize=12)

    cf = axs[1, 1].contourf(X, Y, masks_80, cmap=cmap)
    axs[1, 1].set_xlim([np.min(X), np.max(X)])
    axs[1, 1].set_ylim([np.min(Y), np.max(Y)])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("80 \%", fontsize=12)

    # hack for 2x2 plot (for publication)
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)

    axs[1, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='y', labelsize=14)

    axs[1, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=14)

    clb = fig.colorbar(cf, ticks=[0,1], ax=axs.ravel().tolist(), shrink=0.75)
    clb.set_label('Mask', labelpad=0.3, fontsize=14)

    plt.savefig("masks.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    np.savez("masks_05.npz", masks=masks_05)
    np.savez("masks_20.npz", masks=masks_20)
    np.savez("masks_40.npz", masks=masks_40)
    np.savez("masks_60.npz", masks=masks_60)
    np.savez("masks_80.npz", masks=masks_80)


if __name__ == "__main__":

    #The following shows the whole process of:
    # - using Gmsh to generate a flat 2D mesh
    # - interpolate bathymetries to the Gmsh 2D meshes
    # - convert Gmsh meshes to SRH-2D cases
    # - run all SRH-2D cases
    # - sample (convert) all SRH-2D cases results to training data

    #open the configuration JSON file
    f_json = open('swe_2D_training_data_generation_config.json')
    bathymetry_inversion_2D_config = json.load(f_json)

    #generate all SRH-2D cases
    #generate_all_SRH_2D_cases(bathymetry_inversion_2D_config)

    #run all SRH-2D cases
    #start = time.time()
    #run_all_SRH_2D_cases(bathymetry_inversion_2D_config)
    #run_all_SRH_2D_cases_parallel(bathymetry_inversion_2D_config,nProcess=6)
    #end = time.time()
    #print("Elapsed time for running SRH-2D cases: ", end - start)

    #convert (sample) all SRH-2D cases results to training data
    #sample_all_SRH_2D_cases(bathymetry_inversion_2D_config)

    #plot results and make visual check on some example SRH-2D cases
    #plot_example_results(bathymetry_inversion_2D_config, nExamples=1)

    #create and optionally check TFRecords
    #createTFRecords(bathymetry_inversion_2D_config)
    createTFRecords_with_specified_IDs(bathymetry_inversion_2D_config)
    #checkTFRecords('test_uv.tfrecords', bathymetry_inversion_2D_config, nPlotSamples=2)
    #checkTFRecords('test_uvWSE.tfrecords', bathymetry_inversion_2D_config, nPlotSamples=2)
    #checkTFRecords('validation_uv.tfrecords', bathymetry_inversion_2D_config, nPlotSamples=2)
    #checkTFRecords('train_uvWSE.tfrecords', bathymetry_inversion_2D_config, nPlotSamples=1)

    #generate some inversion data
    #generate_inversion_data('test_uv.tfrecords', bathymetry_inversion_2D_config, nInversionCases=-1)
    #generate_inversion_data('test_uvWSE.tfrecords', bathymetry_inversion_2D_config, nInversionCases=10)

    # generate masks to only use subset of (u,v,WSE) for inversion
    #generate_subset_masks(bathymetry_inversion_2D_config)

    #close the JSON config file
    f_json.close()

    print("Done!")
