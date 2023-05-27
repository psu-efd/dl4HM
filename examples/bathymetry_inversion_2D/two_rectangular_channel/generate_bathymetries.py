"""
Generate 2D bathymetry data using Gaussian process (sampled from a prior)
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C

from scipy import interpolate

import pyvista as pv

import json


plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', size=14, family='serif') #specify the default font family to be "serif"

def smooth_and_plot(sampled : pv.core.grid.UniformGrid):
    mesh = sampled.warp_by_scalar('elevation')
    mesh = mesh.extract_surface()

    # clean and smooth a little to reduce perlin noise artifacts
    #mesh = mesh.smooth(n_iter=100, inplace=True, relaxation_factor=0.07)
    mesh.plot(cmap=plt.cm.terrain)

def generate_2D_bed(bathymetry_inversion_2D_config):
    """

    :param amplitude:
    :return:
    """

    # get parameter values from the config file
    amplitude = bathymetry_inversion_2D_config['amplitude']    #bed elevation amplitude (for scaling purpose)
    nGrid = bathymetry_inversion_2D_config['nGrid']         #number of grid in x and y
    nSamples = bathymetry_inversion_2D_config['nSamples']   #number of samples to generate
    xstart = bathymetry_inversion_2D_config['xstart']
    xend = bathymetry_inversion_2D_config['xend']
    ystart = bathymetry_inversion_2D_config['ystart']
    yend = bathymetry_inversion_2D_config['yend']

    sigma_zb = bathymetry_inversion_2D_config['sigma_zb']
    lx = bathymetry_inversion_2D_config['lx']     #length scale in x
    ly = bathymetry_inversion_2D_config['ly']     #length scale in y


    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

    xArray = np.linspace(0, 1, nGrid)
    yArray = np.linspace(0, 1, nGrid)
    x1, x2 = np.meshgrid(xArray, yArray)

    xx = np.vstack((x1.flatten(), x2.flatten())).T

    #error std
    #sigma_n = 0.5

    #z = 0.0*x1.flatten() + np.random.normal(loc=0, scale=sigma_n, size=nGrid*nGrid)
    #z = 0.001 * np.sin(x1.flatten())

    kernel = sigma_zb ** 2 * RBF(length_scale=(lx, ly), length_scale_bounds=(1e-1, 10.0))
    #gpr = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, random_state=1)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=1)

    #fit
    #gpr.fit(xx, z)

    elevationArray = gpr.sample_y(xx, nSamples, random_state=None)

    elevationArray = elevationArray.reshape(nGrid, nGrid, nSamples).transpose((1, 0, 2))

    #scale the elevation to [-amplitude, amplitude] (as a whole)
    #elevationArray = ((elevationArray - elevationArray.min())/(elevationArray.max()-elevationArray.min()) - 0.5)*2*amplitude

    #scale the elevation to [-amplitude, amplitude] (individually to reduce the variation of elevation range)
    #This effectively embeds some prior into the training data, i.e., we believe the bed elevation is in the
    #specified range AND every bathymetry fully spans the range. This reduces one more degree of freedom. Otherwise
    #the training data size should be further increased.
    for i in range(nSamples):
        #zero the mean first
        elevationArray[:, :, i] = elevationArray[:,:,i] - elevationArray[:,:,i].mean()

        #scale
        elevationArray[:,:,i] = ((elevationArray[:,:,i] - elevationArray[:,:,i].min()) / (
                    elevationArray[:,:,i].max() - elevationArray[:,:,i].min()) - 0.5) * 2 * amplitude

    xArray = np.linspace(xstart, xend, nGrid)
    yArray = np.linspace(ystart, yend, nGrid)

    # normalized elevation in [-0.5, 0.5]
    elevationArray_norm = (
                (elevationArray - elevationArray.min()) / (elevationArray.max() - elevationArray.min()) - 0.5)

    #calculate the range of slope as prior information (should use the normalized elevation here)
    dzb_dx_min = 1E6
    dzb_dx_max = 1E-6
    dzb_dy_min = 1E6
    dzb_dy_max = 1E-6

    for i in range(nSamples):
        dzb_dx = (np.diff(elevationArray_norm[:,:,i].squeeze(), axis=1))
        dzb_dy = (np.diff(elevationArray_norm[:,:,i].squeeze(), axis=0))

        if dzb_dx_min > dzb_dx.min():
            dzb_dx_min = dzb_dx.min()

        if dzb_dx_max < dzb_dx.max():
            dzb_dx_max = dzb_dx.max()

        if dzb_dy_min > dzb_dy.min():
            dzb_dy_min = dzb_dy.min()

        if dzb_dy_max < dzb_dy.max():
            dzb_dy_max = dzb_dy.max()

    slope_range = np.array([dzb_dx_min, dzb_dx_max, dzb_dy_min, dzb_dy_max])
    print("slope_range = ", slope_range)

    np.savez(bathymetry_data_file_name, xArray=xArray, yArray=yArray, elevation=elevationArray, slope_range=slope_range)

def plot_sample_bathymetries(nrows, ncolumns, bathymetry_inversion_2D_config):
    """
    Plot some randomly selected bathymetries

    :param nrows:
    :param ncolumns:
    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

    #load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    xArray = bathymetry_data['xArray']
    yArray = bathymetry_data['yArray']
    elevation = bathymetry_data['elevation']

    #total number of bathymetry in the data
    nBathy = elevation.shape[2]

    #randonly draw (nrows * ncolumns) bathymetries from the data
    choices = np.sort(np.random.choice(nBathy, size=nrows*ncolumns, replace=False))

    #force the first choice to be 0
    #choices[0] = 0
    #choices[-1] = nBathy-1

    #hack: put the ID of the bathymetry you want to plot here
    choices[0] = 592
    choices[1] = 1170
    choices[2] = 1184
    choices[3] = 2321

    #amplitude of the bathymetry elevation
    amplitude = bathymetry_inversion_2D_config['amplitude']

    #zMax = np.max(elevation)
    #zMin = np.min(elevation)
    zMax = amplitude
    zMin = -amplitude
    levels = np.linspace(zMin, zMax, 51)

    fig, axs = plt.subplots(nrows, ncolumns, figsize=(ncolumns*8, nrows*2),
                            sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.05)

    for i, ax, choice in zip(range(nrows*ncolumns), axs.ravel(), choices):
        cf = ax.contourf(xArray, yArray, elevation[:,:,choice], levels, vmin=zMin,
                         vmax=zMax, cmap=plt.cm.terrain, extend='both')
        ax.set_xlim([np.min(xArray), np.max(xArray)])
        ax.set_ylim([np.min(yArray), np.max(yArray)])
        ax.set_aspect('equal')
        ax.set_title("Sample " + str(i))


    clb = fig.colorbar(cf, ticks=np.linspace(zMin, zMax, 11), ax=axs.ravel().tolist())
    clb.ax.tick_params(labelsize=16)
    clb.set_label('Elevation (m)', labelpad=0.3, fontsize=20)

    # set labels
    #plt.setp(axs[-1, :], xlabel='x (m)')
    #plt.setp(axs[:, 0], ylabel='y (m)')

    #hack for 2x2 plot (for publication)
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=18)
    axs[0, 0].tick_params(axis='y', labelsize=16)

    axs[1, 0].set_xlabel('$x$ (m)', fontsize=18)
    axs[1, 0].tick_params(axis='x', labelsize=16)
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=18)
    axs[1, 0].tick_params(axis='y', labelsize=16)

    axs[1, 1].set_xlabel('$x$ (m)', fontsize=18)
    axs[1, 1].tick_params(axis='x', labelsize=16)

    #plot the lines for profiles (one longitudinal and the other cross section)
    axs[0, 0].plot([0,26],[3.2,3.2], 'k--')
    axs[0, 0].plot([12.8, 12.8], [0, 6.4], 'k--')

    plt.savefig("sample_bathymetries.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def animate_bathymetry_contours(bathymetry_inversion_2D_config):
    """
    Animate the bathymetry contours

    :param bathymetry_inversion_2D_config:
    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

    #load the bathymetry data from file
    bathymetry_data = np.load(bathymetry_data_file_name)

    xArray = bathymetry_data['xArray']
    yArray = bathymetry_data['yArray']
    elevation = bathymetry_data['elevation']

    #total number of bathymetry in the data
    nBathy = elevation.shape[2]

    #amplitude of the bathymetry elevation
    amplitude = bathymetry_inversion_2D_config['amplitude']

    zMax = amplitude
    zMin = -amplitude
    levels = np.linspace(zMin, zMax, 51)

    fig = plt.figure(figsize=(10, 5))

    axis = plt.axes(xlim=(np.min(xArray), np.max(xArray)), ylim=(np.min(yArray), np.max(yArray)))
    axis.set_aspect('equal')

    # set labels
    axis.set_xlabel('x (m)')
    axis.set_ylabel('y (m)')

    cf = axis.contourf(xArray, yArray, elevation[:,:,0], levels, vmin=zMin, vmax=zMax, cmap=plt.cm.terrain)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)

    clb = plt.colorbar(cf, ticks=np.linspace(zMin, zMax, 7), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title('Unit: m', fontsize=12)

    def animate(i):
        print(i)
        cf = axis.contourf(xArray, yArray, elevation[:,:,i], levels, vmin=zMin, vmax=zMax, cmap=plt.cm.terrain)

        axis.set_title("Bathymetry " + str(i))

        return cf

    anim = animation.FuncAnimation(fig, animate, frames=100)  #nBathy

    # save the animation to file
    FFwriter = animation.FFMpegWriter(bitrate=1500, fps=10)
    anim.save('bathymetry_animation.mp4', writer=FFwriter)

    #plt.show()


def sample_bathymetries_as_inversion_init(nSamples, bathymetry_inversion_2D_config, srh_2d_config):
    """
    Sample some randomly selected bathymetries as initial zb for inversion

    :param nSamples:
    :return:
    """

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

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



if __name__ == '__main__':

    #load bathymetry generation parameters from json config file
    f_json = open('swe_2D_training_data_generation_config.json')
    config = json.load(f_json)

    bathymetry_inversion_2D_config = config['bathymetry parameters']
    srh_2d_config = config['SRH-2D cases']

    #generate the bed bathymetires
    #generate_2D_bed(bathymetry_inversion_2D_config)

    #plot some sample bed bathymetries to visually check (the first two numbers are rows and columns of subplots)
    plot_sample_bathymetries(2, 2, bathymetry_inversion_2D_config)

    #animate the bathymetry contours
    #animate_bathymetry_contours(bathymetry_inversion_2D_config)

    #select some random bathymetries as the initial zb for inversion
    #nSamples = 10
    #sample_bathymetries_as_inversion_init(nSamples, bathymetry_inversion_2D_config, srh_2d_config)

    #close the JSON file
    f_json.close()

    print("Done!")

