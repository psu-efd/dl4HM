"""
Generate training data for backwater curve solver surrogate
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise1, pnoise3
import random
from scipy.interpolate import interp1d
import json
from numpy import unravel_index

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def generate_bed_profiles(nProfiles, numPoints, xstart, xend):
    """generate bed profiles

    :param nProfiles: int, number of profiles to generate
    :return:
    """

    print("Generating bed profiles ...")

    #t coordinate (only used by the Perlin noise function; not the x cooridinate)
    t = np.linspace(0, 10, num=numPoints)

    # bed elevation for all profiles
    zb = np.zeros((nProfiles, numPoints))

    for i in range(nProfiles):
        #randomize each profile
        base = random.randint(0, 100)
        octaves = random.randint(9, 12)
        persistence = random.uniform(1.2, 5.8)
        lacunarity = random.uniform(2.1, 2.9)

        #print("base, octaves, persistence, lacunarity = ", base, octaves, persistence, lacunarity)

        for j in range(numPoints):
            zb[i, j] = pnoise1(t[j], octaves=octaves, persistence=persistence, base=base, lacunarity=lacunarity)

    # x coordinates
    x_bed = np.linspace(xstart, xend, numPoints)

    # make bed plots
    print_bed_profiles(x_bed, zb)

    # save data
    np.save("x_bed.npy", x_bed)
    np.save("zb.npy", zb)

    # test: load back the data
    # x_bed = np.load("x_bed.npy")
    # zb = np.load("zb.npy")

def print_bed_profiles(x, zb):
    """Print the bed profiles

    :param x:
    :param zb:
    :return:
    """

    # plot with the simulated results and compare with target
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    # plot bed profiles
    for i in range(zb.shape[0]):
        ax.plot(x, zb[i,:])

    ax.set_xlabel('x (m)', fontsize=16)
    ax.set_ylabel('elevation (m)', fontsize=16)

    # set the limit for the x and y axes
    # plt.xlim([0,1.0])
    # plt.ylim([5,45])

    # show the ticks on both axes and set the font size
    plt.tick_params(axis='both', which='major', labelsize=14)

    # show title and set font size
    # plt.title('Backwater curve', fontsize=16)

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    # plt.savefig("Bed inversion", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

# the RK4 method to solve an ODE
# f is the dydx slope function
# x_0 and y_0 are IC
# h is step size and n is the number of steps
# The backwater curve equation can't handle transcritical flow. If that happends, return transcritical = True.
def RK4(f, x_0, y_0, h, n, slope, Cf, qw):
    x_res = np.zeros(n + 1)
    y_res = np.zeros(n + 1)
    x, y = x_0, y_0
    x_res[0] = x_0
    y_res[0] = y_0
    i = 0

    #whether transcritical happened (Fr ~ 1)
    transcritical = False

    while i < n:
        k1, transcritical = f(x, y, slope[i], Cf, qw)
        k2, transcritical = f(x + 0.5 * h, y + 0.5 * k1 * h, slope[i], Cf, qw)
        k3, transcritical = f(x + 0.5 * h, y + 0.5 * k2 * h, slope[i], Cf, qw)
        k4, transcritical = f(x + h, y + k3 * h, slope[i], Cf, qw)
        x += h
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        i = i + 1
        x_res[i] = x
        y_res[i] = y

    return x_res, y_res, transcritical

def F_H(x,H,slope,Cf,qw): # F(H) function
    "F(H) function in the backwater curve. It can't handle transcritical."

    H = max(H, 0.01)    #to make sure we don't have a tiny water depth

    transcritical = False

    #Froude number
    Fr = qw/H/math.sqrt(9.81*H)

    if math.fabs(Fr - 1.0) < 1E-3:
        transcritical = True

    return -(slope-Cf*qw**2/9.81/H**3)/(1-qw**2/9.81/H**3), transcritical

def simulate_one_backwater_curve(zb_bed, slope, x0, L, H0, Cz, qw, N):
    "Solve the backwater curve equation with a variable bed"

    #convert the Chezy coefficient to friction coefficient
    Cf = 1 / Cz ** 2

    #step size
    step_size = L/(N-1)

    # call the ODE solvers
    x, H, transcritical = RK4(F_H, x0, H0, step_size, N-1, slope, Cf, qw)

    #water surface elevation
    WSE = zb_bed + H

    # whether the solver diverged
    diverged = False

    #sanity check
    if np.min(H) <= 0 or np.max(WSE) > 3.5 or (np.max(WSE) - np.min(WSE)) > 2.0:   #here the max is kind of arbitrary; need to adjust
        diverged = True
        #raise Exception("Negative or zero water depth. The solver diverged.")

    #mean velocity
    U = qw/(H+1e-3)

    return WSE, H, U, (diverged or transcritical)

def simulate_all_backwater_curves(nProfiles, x_bed, zb_beds, x0, L, H0, Cz, qw, N):
    """Solve all backwater curves for all the given bed profiles

    :param nProfiles:
    :param x_bed:
    :param zb_beds:
    :param x0:
    :param L:
    :param H0:
    :param Cz:
    :param qw:
    :param N:
    :return:
    """

    print("Simulating all backwater curves ...")

    x = np.linspace(x0, L, N)
    zb_all = np.zeros((nProfiles, N))

    WSE = np.zeros((nProfiles, N))
    H = np.zeros((nProfiles, N))
    U = np.zeros((nProfiles, N))

    #a list to store diverged backwater curve
    diverged_curve_IDs = []

    #loop over each bed profile
    for i in range(nProfiles):
        print("\tProfile i = ", i)

        #interpolate the bed profile to the grids
        bedInterpolator = interp1d(x_bed, zb_beds[i,:], fill_value="extrapolate")

        #bed profile on grid for the current profile
        zb_all[i,:] = bedInterpolator(x)

        #calcualte the slope
        slope = np.zeros(N)

        for j in range(N):
            if j < (N - 1):
                slope[j] = ((zb_all[i, j + 1] - zb_all[i, j]) / (x[j + 1] - x[j]))
            else:
                slope[j] = ((zb_all[i, j] - zb_all[i, j - 1]) / (x[j] - x[j - 1]))

        #simulate the current backwater curve
        WSE[i,:], H[i,:], U[i,:], diverged = simulate_one_backwater_curve(zb_all[i,:], slope, x0, L, H0, Cz, qw, N)

        if diverged:
            print("Transcritical, negative or zero water depth for profile ID ", i)
            diverged_curve_IDs.append(i)

    #deal with those diverged backwater curves (remove them from the data)
    if diverged_curve_IDs:
        print("The following profiles will be removed due to divergence: ", diverged_curve_IDs)

        zb_all = np.delete(zb_all, diverged_curve_IDs, axis=0)
        WSE = np.delete(WSE, diverged_curve_IDs, axis=0)
        H = np.delete(H, diverged_curve_IDs, axis=0)
        U = np.delete(U, diverged_curve_IDs, axis=0)

    #make plot backwater profiles (diverged profiles not excluded yet)
    plot_backwater_profiles(x, zb_all, WSE)

    return x, WSE, H, U, diverged_curve_IDs

def plot_backwater_profiles(x, zb_all, WSE):
    """Print the bed profiles

    :param x:
    :param zb:
    :return:
    """

    # plot with the simulated results and compare with target
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    # plot profiles (randomly select some to plot)
    selection = random.sample(range(zb_all.shape[0]), zb_all.shape[0])    #4

    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

    for i, profile_ID in enumerate(selection):
        ax.plot(x, WSE[i, :], color=colors[i%7], )  # WSE
        ax.plot(x, zb_all[i, :], color=colors[i%7])  # bed

    #also plot the one with the largest WSE
    indices_max_WSE = unravel_index(WSE.argmax(), WSE.shape)
    print("Maximum WSE happes at ", indices_max_WSE)

    ax.plot(x, WSE[indices_max_WSE[0],:], 'k--')
    ax.plot(x, zb_all[indices_max_WSE[0], :], 'k--')

    ax.set_xlabel('x (m)', fontsize=16)
    ax.set_ylabel('elevation (m)', fontsize=16)

    # set the limit for the x and y axes
    # plt.xlim([0,1.0])
    # plt.ylim([5,45])

    # show the ticks on both axes and set the font size
    plt.tick_params(axis='both', which='major', labelsize=14)

    # show title and set font size
    # plt.title('Backwater curve', fontsize=16)

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    # plt.savefig("Bed inversion", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_one_backwater_profile(x_bed, zb, x, WSE):
    """Print the bed profiles

    :param x:
    :param zb:
    :return:
    """

    # plot with the simulated results and compare with target
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    ax.plot(x, WSE, 'r' )  # WSE
    ax.plot(x_bed, zb, 'k')  # bed

    ax.set_xlabel('x (m)', fontsize=16)
    ax.set_ylabel('elevation (m)', fontsize=16)

    # set the limit for the x and y axes
    # plt.xlim([0,1.0])
    # plt.ylim([5,45])

    # show the ticks on both axes and set the font size
    plt.tick_params(axis='both', which='major', labelsize=14)

    # show title and set font size
    # plt.title('Backwater curve', fontsize=16)

    # show legend, set its location, font size, and turn off the frame
    # plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.savefig("Bed_inversion_truth", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def minMaxScale2D(twoDArray):
    """
    linearly scale all elements in a 2D array to (-1,1) based on the min and max of the array.

    :param twoDArray:
    :return:
    """

    min = np.min(twoDArray)
    max = np.max(twoDArray)

    return min, max, (twoDArray - min)/(max - min)

if __name__ == '__main__':

    #parameters
    nProfiles = 1000  #number of bed profiles
    numPoints = 20   #number of points on each bed profile
    xstart = 0.0     #starting x
    L = 1000         #river length
    xend = xstart + L  #end x
    H0 = 2.6    # starting water depth
    #S0 = 2E-4  # background bed slope
    Cz = 22.0  # Chezy friction coefficient
    L = 1000.0  # River length in m
    qw = 6.0  # specific discharge (discharge per unit width)

    NGrid = 200  # number of grids

    split_ratio = 0.9  # percentage for training and the rest for testing

    #generate bed profiles
    generate_bed_profiles(nProfiles, numPoints, xstart, xend)

    #load back the bed data
    x_bed = np.load("x_bed.npy")
    zb_beds = np.load("zb.npy")

    #solve 1D backwater curve ODE based on the generated bed profiles
    x, WSE, H, U, diverged_curve_IDs = simulate_all_backwater_curves(nProfiles, x_bed, zb_beds, xstart, L, H0, Cz, qw, NGrid)

    print("Before zb_beds.shape =", zb_beds.shape)

    #deal with those bed profiles that generate diverged backwater curves (remove them from the data)
    if diverged_curve_IDs:
        print("The following profiles will be removed due to divergence: ", diverged_curve_IDs)

        zb_beds = np.delete(zb_beds, diverged_curve_IDs, axis=0)

    print("After zb_beds.shape =", zb_beds.shape)

    #scaling to (-1,1) for all fields
    zb_beds_min, zb_beds_max, zb_beds = minMaxScale2D(zb_beds)
    WSE_min, WSE_max, WSE = minMaxScale2D(WSE)
    H_min, H_max, H = minMaxScale2D(H)
    U_min, U_max, U = minMaxScale2D(U)

    minMax_dict = {"zb_beds_min": zb_beds_min, \
                   "zb_beds_max": zb_beds_max, \
                   "WSE_min": WSE_min, \
                   "WSE_max": WSE_max, \
                   "H_min": H_min, \
                   "H_max": H_max, \
                   "U_min": U_min, \
                   "U_max": U_max}

    with open('minMaxVars.json', 'w') as convert_file:
        convert_file.write(json.dumps(minMax_dict))

    #split the generated data into training data and test data
    total_data_len = zb_beds.shape[0]
    training_len = int(total_data_len * split_ratio)

    #save data in one file to be used as training data
    np.savez("backwater_curve_training_data.npz", x_bed=x_bed, zb_beds=zb_beds[0:training_len,:],
             x=x, WSE=WSE[0:training_len,:], H=H[0:training_len,:], U=U[0:training_len,:])

    # save data in one file to be used as testing data
    np.savez("backwater_curve_testing_data.npz", x_bed=x_bed, zb_beds=zb_beds[training_len:total_data_len, :],
             x=x, WSE=WSE[training_len:total_data_len, :], H=H[training_len:total_data_len, :],
             U=U[training_len:total_data_len, :])

    #check: read back data
    training_data = np.load("backwater_curve_training_data.npz")
    print(training_data.files)
    print(training_data['WSE'])

    testing_data = np.load("backwater_curve_testing_data.npz")
    print(testing_data.files)
    print(testing_data['WSE'])

    #save one profile for inversion
    np.savez("backwater_curve_inversion_data.npz", zb_beds=zb_beds[-1, :], WSE=WSE[-1, :])  #take the last one

    plot_one_backwater_profile(x_bed, zb_beds[-1, :]*(zb_beds_max-zb_beds_min)+zb_beds_min, x, \
                               WSE[-1, :]*(WSE_max-WSE_min)+WSE_min)

    print("Done!")