"""
Some specific utility functions, such as plotting.

"""

import matplotlib.pyplot as plt
import numpy as np

import shapefile

import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable

import meshio

import json

from pathlib import Path

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_scheme():
    """
    Make plot for scheme:
    1. bathy in xy (with river center line)
    2. bathy in st
    3. s and t contour plots in xy

    :return:
    """

    fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=False, facecolor='w', edgecolor='k')

    #fig.subplots_adjust(hspace=.15, wspace=.01)

    zb_min = 21
    zb_max = 29

    local_levels = np.linspace(zb_min, zb_max, 51)

    #1. plot bathy in xy and center line
    vtk_fileName = 'SRH2D_savana_SI_1000_C_0005_tri.vtk'

    #read data from vtk file
    vtk_result = meshio.read(vtk_fileName)

    # number of triangles
    nTri = vtk_result.cells[0].data.shape[0]

    tri = np.zeros((nTri, 3))
    for i in range(nTri):
        tri[i, 0] = vtk_result.cells[0].data[i, 0]
        tri[i, 1] = vtk_result.cells[0].data[i, 1]
        tri[i, 2] = vtk_result.cells[0].data[i, 2]

    xCoord = vtk_result.points[:, 0]
    yCoord = vtk_result.points[:, 1]

    xl = xCoord.min()
    xh = xCoord.max()
    yl = yCoord.min()
    yh = yCoord.max()

    zb = np.squeeze(vtk_result.point_data['Bed_Elev_m'])

    cf_zb = axs[0, 0].tricontourf(xCoord, yCoord, tri, zb, local_levels,
                                        vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain, extend='neither')
    axs[0, 0].set_xlim([xl, xh])
    axs[0, 0].set_ylim([yl, yh])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].set_title("Bathymetry in $x$-$y$ space", fontsize=14)

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(cf_zb, ticks=np.linspace(21, 29, 7), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title("(m)", loc='center', fontsize=12)

    # read the center line
    sf = shapefile.Reader("center_line.shp")

    # we could have multiple polylines. But here we only use the first one.
    polylines = sf.shapes()
    polyline = polylines[0]
    points = polyline.points

    xCoord_cl = np.zeros(len(points))
    yCoord_cl = np.zeros(len(points))

    for pointI, point in enumerate(points):
        xCoord_cl[pointI] = point[0]
        yCoord_cl[pointI] = point[1]

    axs[0, 0].plot(xCoord_cl, yCoord_cl, color='r', linewidth=1.5)

    axs[0, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[0, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[0, 0].tick_params(axis='x', labelsize=14)
    axs[0, 0].tick_params(axis='y', labelsize=14)

    # plot bathy in st space
    vtk_fileName = 'SRH2D_savana_SI_1000_C_0005_st_tri.vtk'

    # read data from vtk file
    vtk_result = meshio.read(vtk_fileName)

    # number of triangles
    nTri = vtk_result.cells[0].data.shape[0]

    tri = np.zeros((nTri, 3))
    for i in range(nTri):
        tri[i, 0] = vtk_result.cells[0].data[i, 0]
        tri[i, 1] = vtk_result.cells[0].data[i, 1]
        tri[i, 2] = vtk_result.cells[0].data[i, 2]

    sCoord = vtk_result.points[:, 0]
    tCoord = vtk_result.points[:, 1]

    sl = sCoord.min()
    sh = sCoord.max()
    tl = tCoord.min()
    th = tCoord.max()

    zb = np.squeeze(vtk_result.point_data['Bed_Elev_m'])

    cf_zb = axs[0, 1].tricontourf(sCoord, tCoord, tri, zb, local_levels,
                                  vmin=zb_min, vmax=zb_max, cmap=plt.cm.terrain, extend='neither')
    axs[0, 1].set_xlim([sl, sh])
    axs[0, 1].set_ylim([tl, th])
    #axs[0, 1].set_aspect('equal')
    axs[0, 1].set_aspect(1)
    axs[0, 1].set_title("Bathymetry in $s$-$t$ space", fontsize=14)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(cf_zb, ticks=np.linspace(21, 29, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title("(m)", loc='center', fontsize=12)

    axs[0, 1].set_xlabel('$s$ (m)', fontsize=16)
    axs[0, 1].tick_params(axis='x', labelsize=14)
    axs[0, 1].set_ylabel('$t$ (m)', fontsize=16)
    axs[0, 1].tick_params(axis='y', labelsize=14)

    #plot s and t contour in xy
    vtk_fileName = 'distance_tri.vtk'

    # read data from vtk file
    vtk_result = meshio.read(vtk_fileName)

    # number of triangles
    nTri = vtk_result.cells[0].data.shape[0]

    tri = np.zeros((nTri, 3))
    for i in range(nTri):
        tri[i, 0] = vtk_result.cells[0].data[i, 0]
        tri[i, 1] = vtk_result.cells[0].data[i, 1]
        tri[i, 2] = vtk_result.cells[0].data[i, 2]

    xCoord = vtk_result.points[:, 0]
    yCoord = vtk_result.points[:, 1]

    xl = xCoord.min()
    xh = xCoord.max()
    yl = yCoord.min()
    yh = yCoord.max()

    distance_s = np.squeeze(vtk_result.point_data['distance_s'])
    distance_t = np.squeeze(vtk_result.point_data['distance_t'])

    s_min = distance_s.min()
    s_max = distance_s.max()
    local_levels = np.linspace(s_min, s_max, 51)

    cf_zb = axs[1, 0].tricontourf(xCoord, yCoord, tri, distance_s, local_levels,
                                  vmin=s_min, vmax=s_max, cmap=plt.cm.jet, extend='neither')
    axs[1, 0].set_xlim([xl, xh])
    axs[1, 0].set_ylim([yl, yh])
    axs[1, 0].set_aspect('equal')
    axs[1, 0].set_title("$s$ coordinate", fontsize=14)
    axs[1, 0].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 0].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 0].tick_params(axis='x', labelsize=14)
    axs[1, 0].tick_params(axis='y', labelsize=14)

    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(cf_zb, ticks=np.linspace(s_min, s_max, 7), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title("(m)", loc='center', fontsize=12)

    #t
    t_min = distance_t.min()
    t_max = distance_t.max()
    local_levels = np.linspace(t_min, t_max, 51)

    cf_zb = axs[1, 1].tricontourf(xCoord, yCoord, tri, distance_t, local_levels,
                                  vmin=t_min, vmax=t_max, cmap=plt.cm.jet, extend='neither')
    axs[1, 1].set_xlim([xl, xh])
    axs[1, 1].set_ylim([yl, yh])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].set_title("$t$ coordinate", fontsize=14)
    axs[1, 1].set_xlabel('$x$ (m)', fontsize=16)
    axs[1, 1].set_ylabel('$y$ (m)', fontsize=16)
    axs[1, 1].tick_params(axis='x', labelsize=14)
    axs[1, 1].tick_params(axis='y', labelsize=14)

    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(cf_zb, ticks=np.linspace(t_min, t_max, 7), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.0f'))
    clb.ax.tick_params(labelsize=12)
    clb.ax.set_title("(m)", loc='center', fontsize=12)

    #add caption
    axs[0,0].text(-0.1, 1.05, "(a)", size=16, ha="center", transform=axs[0,0].transAxes)  # upper left
    axs[0,1].text(-0.1, 1.05, "(b)", size=16, ha="center", transform=axs[0,1].transAxes)
    axs[1, 0].text(-0.1, 1.05, "(c)", size=16, ha="center", transform=axs[1, 0].transAxes)  # upper left
    axs[1, 1].text(-0.1, 1.05, "(d)", size=16, ha="center", transform=axs[1, 1].transAxes)

    plt.savefig("Savana_scheme_mapping.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':

    plot_scheme()

    print("Done!")