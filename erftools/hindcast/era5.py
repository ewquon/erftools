"""ERA5 hindcast preprocessing workflow."""
from __future__ import annotations

import glob
import logging
import os
from typing import Any, List, Optional, Tuple, Union

from .base import HindcastBase

logger = logging.getLogger(__name__)

# Output directories created by the ERA5 workflow.
_ERA5_OUTPUT_DIRS: List[str] = [
    "Output/ERA5Data_3D",
    "Output/ERA5Data_Surface",
    "Output/VTK/3D/ERA5Domain",
    "Output/VTK/3D/ERFDomain",
    "Output/VTK/Surface/ERA5Domain",
    "Output/VTK/Surface/ERFDomain",
]

# Glob patterns for downloaded GRIB files.
_SURFACE_FILE_PATTERN = "era5_surf_*.grib"
_3D_FILE_PATTERN = "era5_3d_*.grib"


class ERA5Hindcast(HindcastBase):
    """ERA5 hindcast preprocessing workflow.

    Downloads ERA5 pressure-level and surface reanalysis/forecast data from
    the Copernicus CDS and produces ERF initial condition files.

    Parameters
    ----------
    config_file:
        Path to the key:value input file (e.g. ``input_for_Laura``).
    do_forecast:
        When ``True``, download a multi-timestep forecast instead of a
        single reanalysis snapshot.
    forecast_time_hours:
        Forecast length in hours (required when *do_forecast* is ``True``).
    interval_hours:
        Interval between forecast timesteps in hours (required when
        *do_forecast* is ``True``).
    """

    def __init__(
        self,
        config_file: Optional[Union[str, os.PathLike]] = None,
        do_forecast: bool = False,
        forecast_time_hours: Optional[int] = None,
        interval_hours: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_file, **kwargs)
        self.do_forecast = do_forecast
        self.forecast_time_hours = forecast_time_hours
        self.interval_hours = interval_hours
        if do_forecast and (forecast_time_hours is None or interval_hours is None):
            raise ValueError(
                "do_forecast=True requires both forecast_time_hours and interval_hours"
            )

    def _prepare_workspace(self) -> None:
        """Remove stale domain_extents file and create output directories."""
        if os.path.exists("Output/domain_extents.txt"):
            os.remove("Output/domain_extents.txt")
        self._setup_output_dirs(*_ERA5_OUTPUT_DIRS)

    def download(self) -> Tuple[List[str], List[float]]:
        """Download ERA5 surface data.

        Returns
        -------
        tuple
            ``(filenames, area)`` where *filenames* is a list of downloaded
            GRIB file paths and *area* is ``[lat_max, lon_min, lat_min, lon_max]``.
        """
        from erftools.preprocessing import (
            Download_ERA5_ForecastSurfaceData,
            Download_ERA5_SurfaceData,
        )

        if self.do_forecast:
            logger.info("Downloading ERA5 forecast surface data...")
            return Download_ERA5_ForecastSurfaceData(
                self.config_file, self.forecast_time_hours, self.interval_hours
            )
        logger.info("Downloading ERA5 reanalysis surface data...")
        return Download_ERA5_SurfaceData(self.config_file)

    def process(self, download_result: Tuple[List[str], List[float]]) -> None:
        """Process ERA5 surface and 3-D files and write ERF outputs.

        Parameters
        ----------
        download_result:
            The ``(filenames, area)`` tuple returned by :meth:`download`.
        """
        from erftools.preprocessing import ReadERA5_SurfaceData, ReadERA5_3DData

        surf_files = sorted(glob.glob(_SURFACE_FILE_PATTERN))
        for filename in surf_files:
            logger.info("Processing surface file: %s", filename)
            ReadERA5_SurfaceData(filename, self.lambert_conformal)

        if self.do_forecast:
            from erftools.preprocessing import Download_ERA5_ForecastData

            Download_ERA5_ForecastData(
                self.config_file, self.forecast_time_hours, self.interval_hours
            )
            for filename in sorted(glob.glob(_3D_FILE_PATTERN)):
                logger.info("Processing 3D file: %s", filename)
                ReadERA5_3DData(filename, self.lambert_conformal)
        else:
            from erftools.preprocessing import Download_ERA5_Data

            filename, _ = Download_ERA5_Data(self.config_file)
            logger.info("Processing 3D file: %s", filename)
            ReadERA5_3DData(filename, self.lambert_conformal)

    def run(self, comm: Any = None) -> None:  # type: ignore[override]
        """Download and process ERA5 data, with optional MPI parallelism.

        Parameters
        ----------
        comm:
            An ``mpi4py.MPI.Comm`` communicator.  When provided, surface
            file downloads are distributed across ranks and processing is
            split evenly.  When ``None`` (default) the workflow runs
            single-process.
        """
        if comm is None:
            rank, size = 0, 1
        else:
            rank = comm.Get_rank()
            size = comm.Get_size()

        if rank == 0:
            self._prepare_workspace()
        if comm is not None:
            comm.Barrier()

        # --- Surface download (MPI-aware inside the download functions) ---
        self.download()
        if comm is not None:
            comm.Barrier()

        # --- Surface processing (distributed across all ranks) ---
        from erftools.preprocessing import ReadERA5_SurfaceData

        surf_files = sorted(glob.glob(_SURFACE_FILE_PATTERN))
        for filename in surf_files[rank::size]:
            logger.info("[Rank %d] Processing surface file: %s", rank, filename)
            ReadERA5_SurfaceData(filename, self.lambert_conformal)

        if comm is not None:
            comm.Barrier()

        # --- 3-D download + processing ---
        if self.do_forecast:
            from erftools.preprocessing import (
                Download_ERA5_ForecastData,
                ReadERA5_3DData,
            )

            Download_ERA5_ForecastData(
                self.config_file, self.forecast_time_hours, self.interval_hours
            )
            if comm is not None:
                comm.Barrier()
            for filename in sorted(glob.glob(_3D_FILE_PATTERN))[rank::size]:
                logger.info("[Rank %d] Processing 3D file: %s", rank, filename)
                ReadERA5_3DData(filename, self.lambert_conformal)
        else:
            from erftools.preprocessing import Download_ERA5_Data, ReadERA5_3DData

            filename, _ = Download_ERA5_Data(self.config_file)
            logger.info("Processing 3D file: %s", filename)
            ReadERA5_3DData(filename, self.lambert_conformal)

        if comm is not None:
            comm.Barrier()
        logger.info("Finished ERA5Hindcast run")
