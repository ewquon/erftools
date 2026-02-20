"""GFS hindcast preprocessing workflow."""
from __future__ import annotations

import logging
import os
from typing import Any, List, Tuple, Union

from .base import HindcastBase

logger = logging.getLogger(__name__)


class GFSHindcast(HindcastBase):
    """GFS hindcast preprocessing workflow.

    Downloads GFS forecast or FNL reanalysis data from NCAR's OSDF and
    produces ERF initial condition files.

    Parameters
    ----------
    config_file:
        Path to the key:value input file (e.g. ``input_for_Laura``).
    do_forecast:
        When ``True``, download the full multi-timestep forecast series.
        When ``False`` (default), download a single analysis snapshot.
    product:
        GFS product type: ``'forecast'`` (default) or ``'fnl'``.
    """

    def __init__(
        self,
        config_file: str,
        do_forecast: bool = False,
        product: str = "forecast",
        **kwargs: Any,
    ) -> None:
        super().__init__(config_file, **kwargs)
        self.do_forecast = do_forecast
        self.product = product
        self._setup_output_dirs("Output")

    def download(self) -> Tuple[Union[str, List[str]], List[float]]:
        """Download GFS data.

        Returns
        -------
        tuple
            ``(filename_or_filenames, area)`` where the first element is
            either a single file path (reanalysis) or a list of paths
            (forecast series).
        """
        from erftools.preprocessing import Download_GFS_Data, Download_GFS_ForecastData

        if self.do_forecast:
            logger.info("Downloading GFS forecast data series...")
            return Download_GFS_ForecastData(self.config_file)
        logger.info("Downloading GFS analysis data...")
        return Download_GFS_Data(self.config_file, product=self.product)

    def process(
        self,
        download_result: Tuple[Union[str, List[str]], List[float]],
    ) -> None:
        """Process GFS files and write ERF outputs.

        Parameters
        ----------
        download_result:
            The ``(filename_or_filenames, area)`` tuple returned by
            :meth:`download`.
        """
        from erftools.preprocessing import ReadGFS_3DData

        filenames_or_file, area = download_result

        if self.do_forecast:
            filenames: List[str] = list(filenames_or_file)  # type: ignore[arg-type]
        else:
            filenames = [filenames_or_file]  # type: ignore[list-item]

        for filename in filenames:
            logger.info("Processing GFS file: %s", filename)
            ReadGFS_3DData(filename, area, self.lambert_conformal)

    def run(self) -> None:
        """Download and process GFS data."""
        from erftools.utils.map import create_US_map, write_vtk_map

        logger.info("Starting GFSHindcast run")
        result = self.download()
        _, area = result
        write_vtk_map(*create_US_map(area), "USMap_LambertProj.vtk")
        self.process(result)
        logger.info("Finished GFSHindcast run")
