"""
This script is mainly to preprocess the Savana case.
1. Convert AdH mesh to srhgeom
2.
"""

import pyHMT2D

import numpy as np

import vtk
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import vtk_to_numpy

import meshio

def try_pyHMT2D():
    # convert SRH-2D result to VTK
    hdf_fileName = "cases/case_0/savana_SI_0_XMDFC.h5"
    srh_caseName = "cases/case_0/savana_SI_0"

    my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data(srh_caseName + ".srhhydro")

    # read SRH-2D result in XMDF format (*.h5)
    # wether the XMDF result is nodal or cell center
    bNodal = False

    my_srh_2d_data.readSRHXMDFFile(hdf_fileName, bNodal)

    # export to VTK
    vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')

def try_contourf():
    # Implementation of matplotlib function
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import ma
    from matplotlib import ticker, cm

    x = np.linspace(-6.0, 6.0, 200)
    y = np.linspace(-7.0, 7.0, 300)
    X, Y = np.meshgrid(x, y)

    Z1 = np.exp(X * Y)
    z = 50 * Z1
    z[:5, :5] = -1
    z = ma.masked_where(z <= 0, z)

    cs = plt.contourf(X, Y, z,
                      locator=ticker.LogLocator(),
                      cmap="bone")

    cbar = plt.colorbar(cs)

    plt.title('matplotlib.pyplot.contourf() Example')
    plt.show()

def change_vtk_coordinates_xy_to_st(vtkFileName_xy, xy_st_vtkFileName, vtkFileName_st):
    """
    Change a vtk file's xy coordinates to st.

    :return:
    """

    vtk_xy = meshio.read(vtkFileName_xy)
    vtk_distance = meshio.read(xy_st_vtkFileName)

    zCoord = vtk_xy.points[:,2]

    distance_s = np.squeeze(vtk_distance.point_data['distance_s'])
    distance_t = np.squeeze(vtk_distance.point_data['distance_t'])

    #stack the coordinates together (s, t, zCoord)
    stz_coord = np.stack([distance_s, distance_t, zCoord]).T

    vtk_xy.points = stz_coord

    vtk_xy.write(vtkFileName_st)



if __name__ == "__main__":
    #try_contourf()

    change_vtk_coordinates_xy_to_st("SRH2D_savana_SI_0_C_0005.vtk", "distance.vtk", "SRH2D_savana_SI_0_C_0005_st.vtk")

    print("All done!")