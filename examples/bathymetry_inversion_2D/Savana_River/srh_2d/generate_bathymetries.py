"""
This script is mainly to generate bathymetry data for the Savana case.
1. Convert AdH mesh to srhgeom
2.
"""

import os
import sys
import py2dm
import json
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import shapefile
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
import shapely
import shutil

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ConstantKernel as C
from sklearn.utils.optimize import _check_optimize_result

from scipy import interpolate
from scipy.stats import qmc

import scipy.stats as st
import scipy
from scipy import ndimage

import pyHMT2D
from pyHMT2D.Hydraulic_Models_Data.SRH_2D import SRH_2D_Data, SRH_2D_Model, SRH_2D_SRHHydro
from pyHMT2D.Misc import gmsh2d_to_srh

import vtk
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy

import meshio

import time

class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        print("theta_opt, func_min=", theta_opt, func_min)

        return theta_opt, func_min


def convert_AdH_mesh_to_srhgeom(srhgeomFileName, AdH_units, shift_x=0.0, shift_y=0.0, srhgeom_units="Meters"):
    """

    Parameters
    ----------
    srhgeomFileName
    AdH_units
    shift_x: shift in x direction (in AdH's unit)
    shift_y: shift in y direction (in AdH's unit)
    srhgeom_units

    Returns
    -------

    """

    #unit check
    if AdH_units != 'Meters' and AdH_units != 'Feet' and \
        srhgeom_units != 'Meters' and srhgeom_units != 'Feet':
        raise "The units specified are wrong."

    fname = srhgeomFileName + ".srhgeom"

    try:
        fid = open(fname, 'w')
    except IOError:
        print('.srhgeom error')
        sys.exit()

    fid.write('SRHGEOM 30\n')
    fid.write('Name \"Converted from Gmsh 2D Mesh \"\n')

    fid.write('\n')

    fid.write('GridUnit \"%s\" \n' % srhgeom_units)

    #with py2dm.Reader('simple.2dm') as mesh:
    with py2dm.Reader('savannah_gridgen_true.3dm') as mesh:

        # all cells
        cellI = 0

        for elem in mesh.iter_elements():
            if elem.num_nodes == 3:    #triangles
                cellI += 1
                fid.write("Elem ")
                fid.write("%d %d %d %d \n" % (cellI,
                                              elem.nodes[0],
                                              elem.nodes[1],
                                              elem.nodes[2]
                                              ))
            elif elem.num_nodes == 4:   #quad
                cellI += 1
                fid.write("Elem ")
                fid.write("%d %d %d %d %d\n" % (cellI,
                                              elem.nodes[0],
                                              elem.nodes[1],
                                              elem.nodes[2],
                                              elem.nodes[3]
                                              ))
            else:
                raise Exception("2DM element type is not supported.")


        # all points
        nodeI = 0

        for node in mesh.iter_nodes():
            nodeI += 1

            fid.write("Node %d " % (nodeI))

            if AdH_units=='Feet' and srhgeom_units=='Meters':
                curr_point_coordinates = [(node.x+shift_x)*0.3048, (node.y+shift_y)*0.3048, node.z*0.3048]
            elif AdH_units=='Meters' and srhgeom_units=='Feet':
                curr_point_coordinates = [(node.x+shift_x)*3.28084, (node.y+shift_y)*3.28084, node.z*3.28084]
            else:
                curr_point_coordinates = [node.x+shift_x, node.y+shift_y, node.z]

            fid.write(" ".join(map(str, curr_point_coordinates)))
            fid.write("\n")


        # NodeString
        nsI = 0

        for ns in mesh.iter_node_strings():
            nsI += 1

            fid.write("NodeString %d " % nsI)

            # line break counter (start a new line every 10 nodes)
            line_break_counter = 0

            # loop over each node ID in the current NodeString
            for nodeID in ns.nodes:

                line_break_counter += 1

                fid.write(" %d" % (nodeID))

                # 10 numbers per line
                if ((line_break_counter % 10) == 0):
                    fid.write("\n")

                    line_break_counter = 0

            fid.write("\n")

    fid.close()

def calc_distance_to_center_line_srhgeom():
    """
    Calculate the distance from srhgeom mesh node to the centerline

    This function reads in the mesh in srhgeom format

    :return:
    """


    # create the SRH_2D_Data object
    my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data("SMS\SI\savana_SI\SRH-2D\Sim\savana_SI.srhhydro")

    # read the center line
    sf = shapefile.Reader("SMS\SI\center_line.shp")

    # we could have multiple polylines. But here we only use the first one.
    polylines = sf.shapes()

    polyline = polylines[0]

    points = polyline.points
    #print("Center line points: ", points)

    line = LineString(points)

    # read the lower half polygon
    sf = shapefile.Reader("SMS\SI\lower_half_polygon.shp")
    lower_half_polygon = Polygon( sf.shapes()[0].points )

    #streamwise and cross-sectional distance to the center line
    resultVarNames = ['distance_s', 'distance_t']
    resultData = []

    #loop over each node in the mesh and calculate its distance to the center line
    for nodeI in range(my_srh_2d_data.srhgeom_obj.numOfNodes):
        a_point = Point(my_srh_2d_data.srhgeom_obj.nodeCoordinates[nodeI, 0],
                        my_srh_2d_data.srhgeom_obj.nodeCoordinates[nodeI, 1])

        #check whether the node is in the lower or upper half of the channel
        bLower = shapely.contains(lower_half_polygon, a_point)

        #the following will return two points: first one is a_point and the second one is the nearest point
        nrst_points = nearest_points(a_point, line)

        distance_s = shapely.line_locate_point(line, nrst_points[1])

        distance_t = a_point.distance(nrst_points[1])

        if bLower:
            resultData.append([distance_s, -distance_t])
        else:
            resultData.append([distance_s, distance_t])

        #print("Distance = ", distance)

    #save to vtk
    bNodal = True
    my_srh_2d_data.outputVTK("distance.vtk", resultVarNames, resultData, bNodal)

    #save srhgeom node's (s,t) coordinates to file
    srhgeom_node_st = np.array(resultData)

    np.savez('srhgeom_node_st.npz', node_st=srhgeom_node_st)

    #draw nodes from the mesh
    chosen_nodes = []
    chosen_distance = []
    for nodeI in range(my_srh_2d_data.srhgeom_obj.numOfNodes):
        #select nodes
        if (nodeI % 20) != 0:
            continue

        nodeCorrdinate = my_srh_2d_data.srhgeom_obj.nodeCoordinates[nodeI,:]
        chosen_nodes.append(nodeCorrdinate)

        chosen_distance.append(resultData[nodeI])

    #print(chosen_nodes)

    save_nodes_to_vtk(chosen_nodes, chosen_distance, "chosen_nodes")

def calc_distance_to_center_line_gmsh(bathymetry_inversion_2D_config):
    """
    Calculate the distance from srhgeom mesh node to the centerline

    This function reads in the mesh in gmsh format

    :return:
    """

    # read in the Gmsh MSH with meshio
    gmsh_flat_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['gmsh_flat_file_name']
    mesh = meshio.read(gmsh_flat_file_name)

    mesh.write('flat_mesh.vtk')

    # read the center line
    sf = shapefile.Reader("SMS\SI\center_line.shp")

    # we could have multiple polylines. But here we only use the first one.
    polylines = sf.shapes()

    polyline = polylines[0]

    points = polyline.points
    #print("Center line points: ", points)

    line = LineString(points)

    # read the lower half polygon
    sf = shapefile.Reader("SMS\SI\lower_half_polygon.shp")
    lower_half_polygon = Polygon( sf.shapes()[0].points )

    #streamwise and cross-sectional distance to the center line
    resultVarNames = ['distance_s', 'distance_t']
    resultData = []

    #loop over each node in the mesh and calculate its distance to the center line
    for nodeI in range(mesh.points.shape[0]):
        a_point = Point(mesh.points[nodeI, 0], mesh.points[nodeI, 1])

        #check whether the node is in the lower or upper half of the channel
        bLower = shapely.contains(lower_half_polygon, a_point)

        #the following will return two points: first one is a_point and the second one is the nearest point
        nrst_points = nearest_points(a_point, line)

        distance_s = shapely.line_locate_point(line, nrst_points[1])

        distance_t = a_point.distance(nrst_points[1])

        if bLower:
            resultData.append([distance_s, -distance_t])
        else:
            resultData.append([distance_s, distance_t])

        #print("Distance = ", distance)

    #add the distance_s and distance_t to gmsh point data
    distance_s = []
    distance_t = []
    for distanceI in resultData:
        distance_s.append(distanceI[0])
        distance_t.append(distanceI[1])

    mesh.point_data['distance_s'] = np.array(distance_s)
    mesh.point_data['distance_t'] = np.array(distance_t)

    #save to the gmsh with distance_st data
    meshio.gmsh.write('gmsh_distance_st.msh', mesh, fmt_version='2.2', binary=False)

    mesh.write('3d_mesh.vtk')

def save_nodes_to_vtk(chosen_nodes, chosen_distance, nodeVTKFileName,dir=''):
    """

    Parameters
    ----------
    chosen_nodes

    Returns
    -------

    """

    vtkFileName_xy = ''   #in x-y space
    vtkFileName_st = ''   #in s-t space
    if len(dir) == 0:
        vtkFileName_xy = nodeVTKFileName + '_xy.vtk'
        vtkFileName_st = nodeVTKFileName + '_st.vtk'
    else:
        vtkFileName_xy = dir + "/" + nodeVTKFileName + '_xy.vtk'
        vtkFileName_st = dir + "/" + nodeVTKFileName + '_st.vtk'

    try:
        fid_xy = open(vtkFileName_xy, 'w')
        fid_st = open(vtkFileName_st, 'w')
    except IOError:
        print('vtk file open error')
        sys.exit()

    # write the header
    fid_xy.write("# vtk DataFile Version 3.0\n")
    fid_xy.write("sampled points\n")
    fid_xy.write("ASCII\n")
    fid_xy.write("DATASET POLYDATA\n")

    fid_st.write("# vtk DataFile Version 3.0\n")
    fid_st.write("sampled points\n")
    fid_st.write("ASCII\n")
    fid_st.write("DATASET POLYDATA\n")

    npoints = len(chosen_nodes)

    fid_xy.write("POINTS %d double\n" % npoints)
    fid_st.write("POINTS %d double\n" % npoints)

    #for node in chosen_nodes:
    #for distance in chosen_distance:
    for node, distance in zip(chosen_nodes, chosen_distance):
        fid_xy.write("%f %f %f\n" %(node[0], node[1], node[2]))
        fid_st.write("%f %f %f\n" % (distance[0], distance[1], node[2]))

    fid_xy.close()
    fid_st.close()

# read point data (sampled from the true bathymetry)
def read_vtk_point_data(vtkFileName):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFileName)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    numpy_nodes = vtk_to_numpy(array)

    return numpy_nodes

def generate_2D_bed(bathymetry_inversion_2D_config):
    """

    :param amplitude:
    :return:
    """

    # random number generator initialization
    rng = np.random.RandomState(1)

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']
    chosen_nodes_st_vtk_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['chosen_nodes_st_vtk_file_name']
    chosen_nodes_xy_vtk_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['chosen_nodes_xy_vtk_file_name']

    target_ele_low = bathymetry_inversion_2D_config['bathymetry parameters']['target_ele_low']
    target_ele_high = bathymetry_inversion_2D_config['bathymetry parameters']['target_ele_high']

    nSamples = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # Grid
    nGrids = bathymetry_inversion_2D_config['bathymetry parameters']['nGrids']
    nGridt = bathymetry_inversion_2D_config['bathymetry parameters']['nGridt']
    min_s = bathymetry_inversion_2D_config['bathymetry parameters']['min_s']
    max_s = bathymetry_inversion_2D_config['bathymetry parameters']['max_s']
    min_t = bathymetry_inversion_2D_config['bathymetry parameters']['min_t']
    max_t = bathymetry_inversion_2D_config['bathymetry parameters']['max_t']
    lim_s = bathymetry_inversion_2D_config['bathymetry parameters']['lim_s']
    lim_t = bathymetry_inversion_2D_config['bathymetry parameters']['lim_t']

    #lin_s = np.linspace(-lim_s, lim_s, nGrids)
    #lin_t = np.linspace(-lim_t, lim_t, nGridt)

    lin_s = np.linspace(min_s, max_s, nGrids)
    lin_t = np.linspace(min_t, max_t, nGridt)

    x1, x2 = np.meshgrid(lin_s, lin_t)
    xx = np.vstack((x1.flatten(), x2.flatten())).T

    # Observed data from the chosen nodes in the mesh
    numpy_nodes_st = read_vtk_point_data(chosen_nodes_st_vtk_file_name)
    numpy_nodes_xy = read_vtk_point_data(chosen_nodes_xy_vtk_file_name)

    X_st = numpy_nodes_st[:, 0:2]  # s and t
    z_obs = numpy_nodes_st[:, 2] # elevation

    # scale X_st to: s in [-lim_s, lim_s] and t in [-lim_t, lim_t]
    #X_st[:, 0] = ((X_st[:, 0] - min_s) / (max_s - min_s) - 0.5) * 2 * lim_s
    #X_st[:, 1] = ((X_st[:, 1] - min_t) / (max_t - min_t) - 0.5) * 2 * lim_t

    # scale elevation to be in [0, 1]
    max_ele = np.max(z_obs)
    min_ele = np.min(z_obs)
    print("max and min observed bed elevation: ", max_ele, min_ele)

    #z_obs = (z_obs - min_ele) / (max_ele - min_ele)

    # add some random noise to data
    noise_std = 0.00

    z_obs_noisy = z_obs + rng.normal(loc=0.0, scale=noise_std, size=z_obs.shape)

    #kernel = RBF()
    ls = 2
    lt = 1
    kernel = C(1.0, (1e-02, 1e4)) * RBF(length_scale=(ls, lt), length_scale_bounds=(1e-3, 1e3)) + WhiteKernel()

    #gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)
    gp = MyGPR(kernel=kernel, n_restarts_optimizer=15)

    gp.fit(X_st, z_obs_noisy)
    print("Learned kernel", gp.kernel_)

    z_mean, z_cov = gp.predict(xx, return_cov=True)

    elevationArray = st.multivariate_normal.rvs(mean=z_mean, cov=z_cov, size=nSamples)

    #elevationArray = gp.sample_y(xx, nSamples, random_state=None)

    max_ele_gen = np.max(elevationArray)
    min_ele_gen = np.min(elevationArray)
    print("max and min generated bed elevation: ", max_ele_gen, min_ele_gen)

    #x1 = min_s + (x1 + lim_s) / 2.0 / lim_s * (max_s - min_s)
    #x2 = min_t + (x2 + lim_t) / 2.0 / lim_t * (max_t - min_t)

    #scale the elevation to [target_ele_low, target_ele_high]

    elevationArray = (elevationArray - min_ele_gen)/(max_ele_gen - min_ele_gen) * (target_ele_high - target_ele_low) + target_ele_low

    #levels = np.linspace(min_ele_gen, max_ele_gen, 51)
    levels = np.linspace(target_ele_low, target_ele_high, 51)

    bPlot = False

    if bPlot:
        fig, axs = plt.subplots(nSamples + 1)

        ax = axs[0]
        # ax.contourf(x1, x2, y_analytic)
        # ax.plot(X[:, 0], X[:, 1], "r.", ms=12)

        for i, post in enumerate(elevationArray, 1):
            # scale the elevation back
            # post = min_ele + (max_ele - min_ele) * post

            # axs[i].plot(X_st[:, 0], X_st[:, 1], "r.", ms=2, alpha=0.9)
            cf = axs[i].contourf(x1, x2, np.flip(post.reshape(nGridt, nGrids)), levels, vmin=target_ele_low,
                                 vmax=target_ele_high, cmap=plt.cm.terrain, extend='both')

        clb = fig.colorbar(cf, ticks=np.linspace(target_ele_low, target_ele_high, 11), ax=axs.ravel().tolist())
        clb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

        # plt.tight_layout()
        # plt.legend()

        # plt.savefig("fit_GP.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

    #return

    elevationArray = elevationArray.reshape(nSamples, nGridt, nGrids)

    elevationArray_list = []
    for i in np.arange(nSamples):
        #elevationArray_list.append(elevationArray[i,:,:])
        elevationArray_list.append(ndimage.zoom(elevationArray[i, :, :], 4))  #resample to 640x80

    #compute the s and t in physical unit
    #sArray = np.linspace(min_s, max_s, nGrids)
    #tArray = np.linspace(min_t, max_t, nGridt)
    sArray = np.linspace(min_s, max_s, nGrids*4)
    tArray = np.linspace(min_t, max_t, nGridt*4)

    np.savez(bathymetry_data_file_name, sArray=sArray, tArray=tArray,
             elevation=elevationArray_list)

def generate_2D_bed_numpy_roll(bathymetry_inversion_2D_config):
    """
    Generate 2D bed using numpy.roll function: shifting the bathymetry left-right and up-down in s-t space
    1. build a 2d interpolator on scatter data points
    2. use the 2d interpolator to sample on regular grid
    3. save data


    :param amplitude:
    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']
    chosen_nodes_st_vtk_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['chosen_nodes_st_vtk_file_name']
    chosen_nodes_xy_vtk_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['chosen_nodes_xy_vtk_file_name']

    target_ele_low = bathymetry_inversion_2D_config['bathymetry parameters']['target_ele_low']
    target_ele_high = bathymetry_inversion_2D_config['bathymetry parameters']['target_ele_high']

    nSamples = bathymetry_inversion_2D_config['bathymetry parameters']['nSamples']

    # Grid
    nGrids = bathymetry_inversion_2D_config['bathymetry parameters']['nGrids']
    nGridt = bathymetry_inversion_2D_config['bathymetry parameters']['nGridt']
    min_s = bathymetry_inversion_2D_config['bathymetry parameters']['min_s']
    max_s = bathymetry_inversion_2D_config['bathymetry parameters']['max_s']
    min_t = bathymetry_inversion_2D_config['bathymetry parameters']['min_t']
    max_t = bathymetry_inversion_2D_config['bathymetry parameters']['max_t']

    lin_s = np.linspace(min_s, max_s, nGrids)
    lin_t = np.linspace(min_t, max_t, nGridt)

    x1, x2 = np.meshgrid(lin_s, lin_t)
    xx = np.vstack((x1.flatten(), x2.flatten())).T

    # Observed data from the chosen nodes in the mesh
    numpy_nodes_st = read_vtk_point_data(chosen_nodes_st_vtk_file_name)
    numpy_nodes_xy = read_vtk_point_data(chosen_nodes_xy_vtk_file_name)

    X_st = numpy_nodes_st[:, 0:2]  # s and t
    X_xy = numpy_nodes_xy[:, 0:2]  # x and y
    z_obs = numpy_nodes_st[:, 2] # elevation

    #build the 2D interpolator
    #interpolator = interp.interp2d(X_st[:,0], X_st[:,1], z_obs, kind='linear')
    #interpolator = interp.CloughTocher2DInterpolator(X_st, z_obs)
    #elevationArray_org = interpolator(xx[:,0], xx[:,1])

    #elevationArray_org = interpolate.griddata(X_st, z_obs, (x1, x2), method='linear', fill_value=0.0)

    #use the combination of 'linear' and 'nearest' in griddata
    elevation_linear = interpolate.griddata(X_st, z_obs, (x1, x2), method='linear')
    elevation_nearest = interpolate.griddata(X_st, z_obs, (x1, x2), method='nearest')
    elevation_linear[np.isnan(elevation_linear)] = elevation_nearest[np.isnan(elevation_linear)]

    elevationArray_org = elevation_linear

    #max_ele_gen = np.max(elevationArray_org)
    #min_ele_gen = np.min(elevationArray_org)
    max_ele_gen = np.max(z_obs)
    min_ele_gen = np.min(z_obs)
    print("max and min bed elevation: ", max_ele_gen, min_ele_gen)

    elevationArray = []

    delta_s = int((max_s - min_s)/nGrids)
    delta_t = int((max_t - min_t)/nGridt)
    print("delta_s, delta_t = ", delta_s, delta_t)

    #roll (in s or t)
    for i in np.arange(nSamples):
        #elevationArray.append(np.roll(elevationArray_org, delta_s*i, axis=1))
        elevationArray.append(np.roll(elevationArray_org, delta_t * i, axis=0))

    #roll in both s and t (#randomly nSample samples the s-t space)
    #sampler = qmc.LatinHypercube(d=2)
    #samples = sampler.integers(l_bounds=[0,0],u_bounds=[nGridt, nGrids], n=nSamples)

    #for i in np.arange(nSamples):
        #i_t = samples[i][0]
        #i_s = samples[i][1]

        #elevationArray.append(np.roll(elevationArray_org, (delta_t * i_t, delta_s * i_s), axis=(0, 1)))

    bPlot = True

    if bPlot:
        levels = np.linspace(min_ele_gen, max_ele_gen, 51)

        fig, axs = plt.subplots(nSamples + 1)

        ax = axs[0]
        # ax.contourf(x1, x2, y_analytic)
        # ax.plot(X[:, 0], X[:, 1], "r.", ms=12)

        for i, post in enumerate(elevationArray, 1):

            #axs[i].plot(X_st[:, 0], X_st[:, 1], "r.", ms=2, alpha=0.9)
            cf = axs[i].contourf(x1, x2, post.reshape(nGridt, nGrids), levels, vmin=min_ele_gen,
                             vmax=max_ele_gen, cmap=plt.cm.terrain, extend='both')

        clb = fig.colorbar(cf, ticks=np.linspace(min_ele_gen, max_ele_gen, 11), ax=axs.ravel().tolist())
        clb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

        plt.tight_layout()
        # plt.legend()

        # plt.savefig("fit_GP.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

    #elevationArray = elevationArray.reshape(nSamples, nGridt, nGrids).transpose((1, 2, 0))
    #elevationArray = elevationArray.reshape(nSamples, nGridt, nGrids)

    np.savez(bathymetry_data_file_name, sArray=lin_s, tArray=lin_t,
             elevation=elevationArray)

def plot_sample_bathymetries(nrows, ncolumns, bathymetry_inversion_2D_config):
    """
    Plot some randomly selected bathymetries

    :param nrows:
    :param ncolumns:
    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']

    #load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    sArray = bathymetry_data['sArray']
    tArray = bathymetry_data['tArray']
    elevation = bathymetry_data['elevation']

    #total number of bathymetry in the data
    nBathy = elevation.shape[2]

    #randonly draw (nrows * ncolumns) bathymetries from the data
    choices = np.sort(np.random.choice(nBathy, size=nrows*ncolumns, replace=False))

    #force the first choice to be 0
    #choices[0] = 0
    #choices[-1] = nBathy-1

    #hack: put the ID of the bathymetry you want to plot here
    #choices[0] = 592
    #choices[1] = 1170
    #choices[2] = 1184
    #choices[3] = 2321

    #amplitude of the bathymetry elevation
    #amplitude = bathymetry_inversion_2D_config['amplitude']

    zMax = np.max(elevation)
    zMin = np.min(elevation)
    #zMax = amplitude
    #zMin = -amplitude
    levels = np.linspace(zMin, zMax, 51)

    fig, axs = plt.subplots(nrows, ncolumns, figsize=(ncolumns*6, nrows*2),
                            sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.05)

    x1, x2 = np.meshgrid(sArray, tArray)

    for i, ax, choice in zip(range(nrows*ncolumns), axs.ravel(), choices):
        cf = ax.contourf(x1, x2, elevation[:,:,choice], levels, vmin=zMin,
                         vmax=zMax, cmap=plt.cm.terrain, extend='both')
        #ax.set_xlim([np.min(sArray), np.max(sArray)])
        #ax.set_ylim([np.min(tArray), np.max(tArray)])
        ax.set_aspect('equal')
        ax.set_title("Sample " + str(i))


    clb = fig.colorbar(cf, ticks=np.linspace(zMin, zMax, 11), ax=axs.ravel().tolist())
    clb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # set labels
    #plt.setp(axs[-1, :], xlabel='x (m)')
    #plt.setp(axs[:, 0], ylabel='y (m)')

    #hack for 2x2 plot (for publication)
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='y', labelsize=14)

    axs[1, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='y', labelsize=14)

    axs[1, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=14)

    #plot the lines for profiles (one longitudinal and the other cross section)
    #axs[0, 0].plot([0,26],[3.2,3.2], 'k--')
    #axs[0, 0].plot([12.8, 12.8], [0, 6.4], 'k--')

    #plt.savefig("sample_bathymetries.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def update_node_elevation_in_srhgeom(bathymetry_inversion_2D_config):
    """
    Update the node elevation of srhgeom file

    Parameters
    ----------
    bathymetry_inversion_2D_config

    Returns
    -------

    """

    # output file name for the generated bathymetry data
    #bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry parameters']['bathymetry_data_file_name']
    bathymetry_data_file_name = 'all_bathy_data.npz'

    # load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    sArray = bathymetry_data['sArray']
    tArray = bathymetry_data['tArray']
    elevation = bathymetry_data['elevation']

    # total number of bathymetry in the data
    nBathy = elevation.shape[0]

    # read the distance.vtk file: (x,y) <-> (s,t) mapping
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName('distance.vtk')
    reader.ReadAllScalarsOn()
    #reader.ReadAllVectorsOn()
    reader.Update()

    ugrid = reader.GetOutput()
    points = ugrid.GetPoints()
    array = points.GetData()
    numpy_nodes = vtk_to_numpy(array)

    distance_s = vtk_to_numpy(ugrid.GetPointData().GetArray('distance_s'))
    distance_t = vtk_to_numpy(ugrid.GetPointData().GetArray('distance_t'))

    #debug
    bDebugShow = False
    if bDebugShow:
        min_ele_gen = elevation.min()
        max_ele_gen = elevation.max()
        levels = np.linspace(min_ele_gen, max_ele_gen, 51)

        fig, axs = plt.subplots(2)

        axs[0].plot(distance_s, distance_t, "r.", ms=2, alpha=0.9)
        axs[0].contourf(sArray, tArray, elevation[:, :, 0], levels, vmin=min_ele_gen,
                        vmax=max_ele_gen, cmap=plt.cm.terrain, extend='both')

        axs[1].plot(distance_s, distance_t, "r.", ms=2, alpha=0.9)
        axs[1].contourf(sArray, tArray, elevation[:, :, 1], levels, vmin=min_ele_gen,
                        vmax=max_ele_gen, cmap=plt.cm.terrain, extend='both')

        plt.tight_layout()
        plt.show()

    # loop over the bathymetry data
    for iBathy in range(nBathy):
        print("Modify srhgeom file: %d out of %d\n" % (iBathy+1, nBathy))

        # create the SRH_2D_Data object
        my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data("SMS/SI/savana_SI/SRH-2D/Sim/savana_SI.srhhydro")

        # construct the 2d interpolator
        #bilinterp = interpolate.interp2d(sArray, tArray, elevation[iBathy, :, :], kind='linear')
        #ele_on_nodes = bilinterp(distance_s, distance_t)

        #ele_on_nodes = interpolate.griddata((sArray, tArray), elevation[iBathy, :, :], (distance_s, distance_t), method='linear')

        interp_spline = interpolate.RectBivariateSpline(tArray, sArray, elevation[iBathy, :, :])
        ele_on_nodes = interp_spline(distance_t, distance_s, grid=False)

        # loop over all nodes in the mesh
        for i in range(numpy_nodes.shape[0]):
            my_srh_2d_data.srhgeom_obj.nodeCoordinates[i, 2] = \
               ele_on_nodes[i]

        # save to new srhgeom file
        my_srh_2d_data.srhgeom_obj.save_as('cases/savana_SI_' + str(iBathy) + '.srhgeom')

        my_srh_2d_data.srhgeom_obj.output_2d_mesh_to_vtk('savana_SI_' + str(iBathy) + '.vtk', dir='cases')

def combine_all_bathy_files():
    """
    Combine all bed bathy files into one file

    :return:
    """

    # load the bathymetry data from files
    bathymetry_data_GP1 = np.load('twoD_bathymetry_data_GP1.npz')
    bathymetry_data_GP2 = np.load('twoD_bathymetry_data_GP2.npz')
    bathymetry_data_GP3 = np.load('twoD_bathymetry_data_GP3.npz')
    bathymetry_data_GP4 = np.load('twoD_bathymetry_data_GP4.npz')
    bathymetry_data_GP5 = np.load('twoD_bathymetry_data_GP5.npz')

    bathymetry_data_roll_s = np.load('twoD_bathymetry_data_roll_s.npz')
    bathymetry_data_roll_t = np.load('twoD_bathymetry_data_roll_t.npz')
    bathymetry_data_roll_st = np.load('twoD_bathymetry_data_roll_st.npz')

    elevation_all = []
    nBathy_all = 0

    # add bathymetry_data_GP1
    sArray_GP1 = bathymetry_data_GP1['sArray']
    tArray_GP1 = bathymetry_data_GP1['tArray']
    elevation = bathymetry_data_GP1['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_GP2
    sArray_GP2 = bathymetry_data_GP2['sArray']
    tArray_GP2 = bathymetry_data_GP2['tArray']
    elevation = bathymetry_data_GP2['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_GP3
    sArray_GP3 = bathymetry_data_GP3['sArray']
    tArray_GP3 = bathymetry_data_GP3['tArray']
    elevation = bathymetry_data_GP3['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_GP4
    sArray_GP4 = bathymetry_data_GP4['sArray']
    tArray_GP4 = bathymetry_data_GP4['tArray']
    elevation = bathymetry_data_GP4['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_GP5
    sArray_GP5 = bathymetry_data_GP5['sArray']
    tArray_GP5 = bathymetry_data_GP5['tArray']
    elevation = bathymetry_data_GP5['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_roll_s
    sArray_roll_s = bathymetry_data_roll_s['sArray']
    tArray_roll_s = bathymetry_data_roll_s['tArray']
    elevation = bathymetry_data_roll_s['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_roll_t
    sArray_roll_t = bathymetry_data_roll_t['sArray']
    tArray_roll_t = bathymetry_data_roll_t['tArray']
    elevation = bathymetry_data_roll_t['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    # add bathymetry_data_roll_st
    sArray_roll_st = bathymetry_data_roll_st['sArray']
    tArray_roll_st = bathymetry_data_roll_st['tArray']
    elevation = bathymetry_data_roll_st['elevation']

    nBathy_all += elevation.shape[0]

    for elevation_i in elevation:
        elevation_all.append(elevation_i)

    print("Total number of bathy: ", nBathy_all)

    np.savez("all_bathy_data.npz", sArray=sArray_GP1, tArray=tArray_GP1,
             elevation=elevation_all)


def sample_bathymetries_as_inversion_init(nSamples, f_json):
    """
    Sample some randomly selected bathymetries as initial zb for inversion

    :param nSamples:
    :return:
    """

    config = json.load(f_json)

    # output file name for the generated bathymetry data
    bathymetry_inversion_2D_config = config['bathymetry parameters']
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

    srh_2d_config = config['SRH-2D cases']
    xmin = srh_2d_config['Gmsh']['xmin']
    xmax = srh_2d_config['Gmsh']['xmax']
    ymin = srh_2d_config['Gmsh']['ymin']
    ymax = srh_2d_config['Gmsh']['ymax']

    n_rows = srh_2d_config['sample_result_n_rows']
    n_cols = srh_2d_config['sample_result_n_cols']

    x_interp = np.linspace(xmin, xmax, n_cols)
    y_interp = np.linspace(ymin, ymax, n_rows)
    xx_interp, yy_interp = np.meshgrid(x_interp, y_interp)

    #load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    xArray = bathymetry_data['xArray']
    yArray = bathymetry_data['yArray']
    elevation = bathymetry_data['elevation']

    #plt.imshow(elevation[:,:,0],aspect=0.25)
    #plt.show()

    grid_x_org, grid_y_org = np.meshgrid(xArray, yArray)

    grid_x_org = grid_x_org.flatten()
    grid_y_org = grid_y_org.flatten()


    #total number of bathymetry in the data
    nBathy = elevation.shape[2]

    #randomly draw nSamples bathymetries from the data
    #choices = np.sort(np.random.choice(nBathy, size=nSamples, replace=False))
    #evenly draw nSamples bathymetries from the data
    choices = np.linspace(0,nBathy-1,nSamples).astype(int)

    print("Chosen samples: ", choices)

    #force the first choice to be 0
    #choices[0] = 0
    #choices[-1] = nBathy-1

    selected_elevations = np.zeros((n_rows, n_cols, nSamples))

    for choice, i in zip(choices, range(nSamples)):
        print("i = ", i)
        f = interpolate.interp2d(xArray, yArray, elevation[:,:,choice], kind='linear')

        selected_elevations[:,:,i] = f(x_interp, y_interp)

        #plt.imshow(selected_elevations[:,:,i])
        #plt.show()

    np.savez("sampled_elevations_for_inversion_init.npz", elevations=selected_elevations)



if __name__ == "__main__":

    #convert AdH mesh to srhgeom format for SRH-2D
    #convert_AdH_mesh_to_srhgeom("savana", AdH_units="Feet", shift_x=-733563.2, shift_y=-1209696.994, srhgeom_units="Meters")

    #create SRH-2D case and mesh in SMS and export center line and lower half polygon

    #calculate distance of all nodes to the center line in s-t space
    #calc_distance_to_center_line_srhgeom()

    # load bathymetry generation parameters from json config file
    f_json = open('swe_2D_training_data_generation_config.json')
    bathymetry_inversion_2D_config = json.load(f_json)

    # generate the bed bathymetires
    #generate_2D_bed(bathymetry_inversion_2D_config)
    #generate_2D_bed_numpy_roll(bathymetry_inversion_2D_config)

    # combine all bed bathymetry files into one file
    #combine_all_bathy_files()

    # plot some sample bed bathymetries to visually check (the first two numbers are rows and columns of subplots)
    #plot_sample_bathymetries(2, 2, bathymetry_inversion_2D_config)

    # animate the bathymetry contours
    # animate_bathymetry_contours(bathymetry_inversion_2D_config)

    # select some random bathymetries as the initial zb for inversion
    # nSamples = 3 #10
    # sample_bathymetries_as_inversion_init(nSamples, f_json)

    # update node elevation in srhgeom file and save as different srhgeom files
    update_node_elevation_in_srhgeom(bathymetry_inversion_2D_config)

    # close the JSON file
    f_json.close()

    print("All done!")