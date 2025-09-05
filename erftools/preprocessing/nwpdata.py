from typing import Union, Tuple
from datetime import datetime
import pandas as pd

import tqdm
import urllib.request

from erftools.utils.projection import create_lcc_mapping

class NWPDataset(object):
    """Base class for handling from numerical weather prediction modeled data

    This should not be used directly.
    """

    def __init__(self,
                 datetime_in: Union[str, datetime, pd.Timestamp],
                 area: Tuple[float, float, float, float],
                 projection: str = 'lambert',
                 forecast: int = 0,
                 **kwargs):
        """
        Parameters
        ----------
        datetime_in: str or datetime-like
            Analysis datetime (YYYY-MM-DD HH:00)
        area: tuple
            (lat_max, lon_min, lat_min, lon_max)
        projection: str, optional
            Type of map projection -- only Lambert available for now
        forecast: int, optional
            If > 0, then use the historical forecast data product and
            retrieve the requested number of forecast hours
        **kwargs: optional
            Additional dataset-specific parameters
        """
        self.analysis_datetime = pd.to_datetime(datetime_in)
        self.area = area
        self.projection_type = projection
        self.forecast = forecast

        self._validate_inputs()
        self._default_setup()
        self._setup(**kwargs)

    def _validate_inputs(self):
        if self.forecast < 0:
            raise ValueError("Number of forecast hours should be >= 0")

        if len(self.area) != 4:
            raise ValueError("Area should have four values")
        if not all(isinstance(bnd, (int, float)) for bnd in self.area):
            raise TypeError("Area lat/lon bounds must be numeric")
        lat_max, lon_min, lat_min, lon_max = self.area
        if (lat_max <= lat_min) or (lon_max <= lon_min):
            raise ValueError("Expect area to be defined as "
                             "(lat_max, lon_min, lat_min, lon_max)")

    def _default_setup(self):
        # setup forecast times
        self.datetimes = pd.date_range(start=self.analysis_datetime,
                                       freq='1h',
                                       periods=self.forecast+1)

        # setup map projection
        proj = self.projection_type.lower()
        if proj in ['lcc','lambert','lambert conformal conic']:
            self.projection_type = 'Lambert conformal conic'
            self.proj = create_lcc_mapping(self.area)
        else:
            raise NotImplementedError(f'Projection type: {self.projection_type}')

    def _setup(self,**kwargs):
        """Do dataset-specific setup, validation tasks"""
        pass

    def _download_with_progress(url, filename, position):
        def hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pbar.total = total_size
                pbar.update(downloaded - pbar.n)
        with tqdm(total=0,
                  unit='B', unit_scale=True,
                  position=position,
                  desc=os.path.basename(filename)) as pbar:
            urllib.request.urlretrieve(url, filename, hook)

    def download(self):
        """This uses the appropriate API to download grib data"""
        raise NotImplementedError("Subclass needs to define download()")

