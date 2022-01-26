import multiprocessing
import os

from pyHMT2D.Hydraulic_Models_Data.SRH_2D import SRH_2D_Data, SRH_2D_Model, SRH_2D_SRHHydro
from pyHMT2D.Misc import gmsh2d_to_srh

def run_one_SRH_2D_case(iBathy):
    """

    Returns
    -------

    """

    processID = multiprocessing.current_process()

    print(processID, ": running SRH-2D case: ", iBathy)

    # go into the case's directory
    os.chdir("./cases/case_" + str(iBathy))

    srh_caseName = "twoD_channel_" + str(iBathy)

    # run the current case
    bRunSucessful = my_run_SRH_2D(srh_caseName)

    # go back to the root
    os.chdir("../..")

    #if successful, return iBathy; otherwise, return -iBathy
    if bRunSucessful:
        return iBathy
    else:
        return -iBathy



def my_run_SRH_2D(srh_caseName):
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
    #if not bRunSucessful:
    #    raise Exception("SRH-2D run failed.")

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    return bRunSucessful