"""
Generate 2D bathymetry data
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pyvista as pv


plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def generate_2D_bed(nx, ny, xstart, xend, ystart, yend):
    """

    :param nx:
    :param ny:
    :param xstart:
    :param xend:
    :param ystart:
    :param yend:
    :return:
    """

    return None

def smooth_and_plot(sampled : pv.core.grid.UniformGrid):
    mesh = sampled.warp_by_scalar('scalars')
    mesh = mesh.extract_surface()

    # clean and smooth a little to reduce perlin noise artifacts
    mesh = mesh.smooth(n_iter=100, inplace=True, relaxation_factor=0.07)
    mesh.plot()


def gravel_plane():
    freq = [180, 180, 50]
    noise = pv.perlin_noise(0.2, freq, (0, 0, 0))
    sampled = pv.sample_function(noise,
                                 bounds=(-10, 2, -10, 10, -10, 10),
                                 dim=(500, 500, 1))

    smooth_and_plot(sampled)


def bumpy_plane():
    freq = [1, 0.2, 0.3]
    dim = [100, 40, 100]
    noise = pv.perlin_noise(0.5, freq, (-1, -10, -10))
    sampled = pv.sample_function(noise,
                                 bounds=(0, 50, 0, 20, -10, 10),
                                 dim=dim)

    smooth_and_plot(sampled)

    #save the elevation array to file
    elevationArray_org = np.array(sampled.active_scalars).reshape(dim)

    print("shape before transpose = ", elevationArray_org.shape)

    elevationArray = elevationArray_org.transpose((2,0,1))

    print("shape after transpose = ", elevationArray.shape)

    elevationArray = elevationArray.reshape(-1, (elevationArray.shape[1]*elevationArray.shape[2]))

    print(elevationArray.shape)

    #print(elevationArray)

    #plt.imshow(elevationArray, interpolation='none',cmap='jet')
    #plt.colorbar()
    #plt.show()

    pca = PCA(0.95)

    lower_dimensional_data = pca.fit_transform(elevationArray)

    print(pca.n_components_)

    approximation = pca.inverse_transform(lower_dimensional_data)
    print("approximation.shape = ", approximation.shape)

    plt.figure(figsize=(8, 4))

    # 1D line
    #plt.plot(elevationArray[1], label="original")
    #plt.plot(approximation[1], label="approximation")
    #plt.xlabel('x', fontsize=14)
    #plt.ylabel('z', fontsize=14)

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(elevationArray_org[:,:,1])
    plt.xlabel('x', fontsize=14)
    plt.title('Original Image', fontsize=20)

    # 154 principal components
    plt.subplot(1, 3, 2)
    plt.imshow(approximation[1,:].reshape(dim[0], dim[1]))
    plt.xlabel('x', fontsize=14)
    plt.title('95% of Explained Variance', fontsize=20)

    # difference
    plt.subplot(1, 3, 3)
    plt.imshow(approximation[1,:].reshape(dim[0], dim[1]) - elevationArray_org[:,:,1])
    plt.xlabel('x', fontsize=14)
    plt.title('diff', fontsize=20)
    plt.colorbar()

    plt.show()


def bumpy_gravel_plane():
    bounds = (-10, 2, -10, 10, -10, 10)
    dim = (500, 500, 1)

    freq = [180, 180, 50]
    noise = pv.perlin_noise(0.2, freq, (0, 0, 0))
    sampled_gravel = pv.sample_function(noise, bounds=bounds, dim=dim)

    freq = [0.5, 0.7, 0]
    noise = pv.perlin_noise(0.5, freq, (-10, -10, -10))
    sampled_bumps = pv.sample_function(noise, bounds=bounds, dim=dim)

    sampled = sampled_gravel
    sampled['scalars'] += sampled_bumps['scalars']

    smooth_and_plot(sampled)

if __name__ == '__main__':

    bumpy_plane()

    #bumpy_gravel_plane()