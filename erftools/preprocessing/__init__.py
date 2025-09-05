from .wrf_inputs import WRFInputDeck
from .grids import LambertConformalGrid

from .plotting import plot_1d

try:
    import cdsapi
except ModuleNotFoundError:
    print("Note: Need to install package 'cdsapi' to work with reanalysis data")
else:
    # ERA5 related funrcions
    from .era5.IO import write_binary_vtk_cartesian
    from .era5.ReadERA5DataAndWriteERF_IC import ReadERA5_3DData
    from .era5.ReadERA5DataAndWriteERF_SurfBC import Download_ERA5_SurfaceData
    from .era5.ReadERA5DataAndWriteERF_SurfBC import Download_ERA5_ForecastSurfaceData
    from .era5.ReadERA5DataAndWriteERF_SurfBC import ReadERA5_SurfaceData
    from .era5.Download_ERA5Data import Download_ERA5_Data
    from .era5.Download_ERA5Data import Download_ERA5_ForecastData

# GFS related funrcions
from ._gfs.Download_GFSData import Download_GFS_Data
from ._gfs.Download_GFSData import Download_GFS_ForecastData
from ._gfs.IO import write_binary_vtk_cartesian
from ._gfs.ReadGFSDataAndWriteERF_IC import ReadGFS_3DData
from ._gfs.ReadGFSDataAndWriteERF_IC_FourCastNetGFS import ReadGFS_3DData_FourCastNetGFS
from ._gfs.ReadGFSDataAndWriteERF_IC_OnlyUVW import ReadGFS_3DData_UVW

from .gfs import GFSDataset

try:
    from herbie import Herbie
except ModuleNotFoundError:
    print("Note: Need to install package 'herbie-data' to work with HRRR data")
else:
    from .hrrr import NativeHRRR, hrrr_projection
