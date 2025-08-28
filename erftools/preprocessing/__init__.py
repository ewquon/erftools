from .wrf_inputs import WRFInputDeck
from .grids import LambertConformalGrid

try:
    import cdsapi
except ModuleNotFoundError:
    print("Note: Need to install package 'cdsapi' to work with reanalysis data")
else:
    # ERA5 related functions
    from .era5.Download_ERA5Data import Download_ERA5_Data
    from .era5.Download_ERA5Data import Download_ERA5_ForecastData
    from .era5.Plot_1D import plot_1d
    from .era5.ReadERA5DataAndWriteERF_IC import ReadERA5_3DData

    # GFS related functions
    from .gfs.Download_GFSData import Download_GFS_Data
    from .gfs.Download_GFSData import Download_GFS_ForecastData
    from .gfs.Plot_1D import plot_1d
    from .gfs.ReadGFSDataAndWriteERF_IC import ReadGFS_3DData
    from .gfs.ReadGFSDataAndWriteERF_IC_FourCastNetGFS import ReadGFS_3DData_FourCastNetGFS

try:
    from herbie import Herbie
except ModuleNotFoundError:
    print("Note: Need to install package 'herbie-data' to work with HRRR data")
else:
    from .hrrr import NativeHRRR, hrrr_projection
