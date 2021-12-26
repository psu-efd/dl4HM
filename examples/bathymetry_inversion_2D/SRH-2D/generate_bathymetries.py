"""
Generate 2D bathymetry data
"""

import math

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tick
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy import interpolate

import pyvista as pv
import meshio

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
    amplitude = bathymetry_inversion_2D_config['amplitude']
    nx = bathymetry_inversion_2D_config['nx']
    ny = bathymetry_inversion_2D_config['ny']
    nz = bathymetry_inversion_2D_config['nz']
    xstart = bathymetry_inversion_2D_config['xstart']
    xend = bathymetry_inversion_2D_config['xend']
    ystart = bathymetry_inversion_2D_config['ystart']
    yend = bathymetry_inversion_2D_config['yend']
    zstart = bathymetry_inversion_2D_config['zstart']  # z does not matter (for visualization purpuse)
    zend = bathymetry_inversion_2D_config['zend']
    xfreq = bathymetry_inversion_2D_config['xfreq']
    yfreq = bathymetry_inversion_2D_config['yfreq']
    zfreq = bathymetry_inversion_2D_config['zfreq']
    xphase = bathymetry_inversion_2D_config['xphase']
    yphase = bathymetry_inversion_2D_config['yphase']
    zphase = bathymetry_inversion_2D_config['zphase']

    # output file name for the generated bathymetry data
    bathymetry_data_file_name = bathymetry_inversion_2D_config['bathymetry_data_file_name']

    freq = [xfreq, yfreq, zfreq]
    dim = [nx, ny, nz]
    noise = pv.perlin_noise(amplitude, freq, (xphase, yphase, zphase))
    sampled = pv.sample_function(noise,
                                 bounds=(xstart, xend, ystart, yend, zstart, zend),
                                 dim=dim,
                                 scalar_arr_name='elevation')

    smooth_and_plot(sampled)

    #save the elevation array to file
    elevationArray = sampled.get_array('elevation').reshape(nz, ny, nx)
    elevationArray = elevationArray.transpose(2,1,0)

    xArray = np.linspace(xstart, xend, nx)
    yArray = np.linspace(ystart, yend, ny)

    np.savez(bathymetry_data_file_name, xArray=xArray, yArray=yArray, elevation=elevationArray)

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

    #amplitude of the bathymetry elevation
    amplitude = bathymetry_inversion_2D_config['amplitude']

    #zMax = np.max(elevation)
    #zMin = np.min(elevation)
    zMax = amplitude
    zMin = -amplitude
    levels = np.linspace(zMin, zMax, 51)

    fig, axs = plt.subplots(nrows, ncolumns, figsize=(ncolumns*10, nrows*2),
                            sharex=True, sharey=True, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=.05)

    for ax, choice in zip(axs.ravel(), choices):
        cf = ax.contourf(elevation[:,:,choice].T, levels, vmin=zMin,
                         vmax=zMax, cmap=plt.cm.terrain)
        ax.set_xlim([np.min(xArray), np.max(xArray)])
        ax.set_ylim([np.min(yArray), np.max(yArray)])
        ax.set_aspect('equal')
        ax.set_title("Sample " + str(choice))

    clb = fig.colorbar(cf, ticks=np.linspace(zMin, zMax, 11), ax=axs.ravel().tolist())
    clb.set_label('Elevation (m)', labelpad=0.3, fontsize=24)

    # set labels
    plt.setp(axs[-1, :], xlabel='x (m)')
    plt.setp(axs[:, 0], ylabel='y (m)')

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

    cf = axis.contourf(elevation[:,:,0].T, levels, vmin=zMin, vmax=zMax, cmap=plt.cm.terrain)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)

    clb = plt.colorbar(cf, ticks=np.linspace(zMin, zMax, 7), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title('Unit: m', fontsize=12)

    def animate(i):
        cf = axis.contourf(elevation[:,:,i].T, levels, vmin=zMin, vmax=zMax, cmap=plt.cm.terrain)

        axis.set_title("Bathymetry " + str(i))

        return cf

    anim = animation.FuncAnimation(fig, animate, frames=nBathy)

    # save the animation to file
    FFwriter = animation.FFMpegWriter(bitrate=1500, fps=1)
    anim.save('bathymetry_animation.mp4', writer=FFwriter)

    #plt.show()


if __name__ == '__main__':

    #load bathymetry generation parameters from json config file
    f_json = open('swe_2D_training_data_generation_config.json')
    bathymetry_inversion_2D_config = json.load(f_json)['bathymetry parameters']

    #generate the bed bathymetires
    generate_2D_bed(bathymetry_inversion_2D_config)

    #plot some sample bed bathymetries to visually check (the first two numbers are rows and columns of subplots)
    plot_sample_bathymetries(2, 2, bathymetry_inversion_2D_config)

    #animate the bathymetry contours
    #animate_bathymetry_contours(bathymetry_inversion_2D_config)

    #close the JSON file
    f_json.close()

    print("Done!")

