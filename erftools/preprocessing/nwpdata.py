import os
from typing import Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import urllib3
# suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from erftools.utils.projection import create_lcc_mapping
from erftools.utils.map import create_US_map, write_vtk_map


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
            raise ValueError('Number of forecast hours should be >= 0')

        if len(self.area) != 4:
            raise ValueError('Area should have four values')
        if not all(isinstance(bnd, (int, float)) for bnd in self.area):
            raise TypeError('Area lat/lon bounds must be numeric')
        lat_max, lon_min, lat_min, lon_max = self.area
        if (lat_max <= lat_min) or (lon_max <= lon_min):
            raise ValueError('Expect area to be defined as '
                             '(lat_max, lon_min, lat_min, lon_max)')

    def _default_setup(self):
        self.urls = []
        self.filenames = []

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

    def _download_with_progress(self, url, filename, chunk_size=8192,
                                position=0):
        # send request with streaming to enable progress bar
        with requests.get(url, stream=True,
                          headers={'User-Agent': 'Mozilla/5.0'},
                          verify=False) as r:
            total_size = int(r.headers.get("Content-Length", 0))

            with tqdm(
                total=total_size,
                unit='B', unit_scale=True,
                position=position,
                desc=os.path.basename(filename)
            ) as pbar, open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

    def _parallel_download(self, urls, filenames, chunk_size=8192,
                           max_workers=4):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, (url, fpath) in enumerate(zip(urls, filenames)):
                if os.path.isfile(fpath):
                    print(f'{fpath} found')
                else:
                    executor.submit(self._download_with_progress,
                                    url, fpath, chunk_size=chunk_size, position=i)

    def download(self, dpath='.', nprocs=1):
        """Download all grib files"""
        if len(self.filenames) == 0:
            raise ValueError('No grib files to download -- invalid inputs?')

        filenames = [os.path.join(dpath, fname) for fname in self.filenames]

        if nprocs==1:
            for url,filename in zip(self.urls, self.filenames):
                fpath = os.path.join(dpath, filename)
                if os.path.isfile(fpath):
                    print(f'{fpath} found')
                else:
                    self._download_with_progress(url, fpath)
        else:
            self._parallel_download(self.urls, filenames, max_workers=nprocs)

    def create_US_map(self, plot=False, output=None):
        """Create a map of the US in projected coordinates

        The default coordinate system is Lambert conformal conic. The
        resulting coordinates may be plotted on screen and/or output
        as an ASCII VTK file; otherwise, the projected coordinates and
        a list of state IDs are returned.
        """
        x_trans, y_trans, id_vec = create_US_map(self.area)
        if output is not None:
            write_vtk_map(x_trans, y_trans, id_vec, output)
        if plot:
            fig,ax = plt.subplots()
            for state_id in np.unique(id_vec):
                sel_state = np.where(id_vec == state_id)[0]
                ax.plot(x_trans[sel_state], y_trans[sel_state], 'k-', lw=1)
            ax.axis('scaled')
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel('$y$ [m]')
            return fig,ax
        elif output is None:
            return x_trans, y_trans, id_vec
