import os
import tqdm
import numpy as np
import pandas as pd

from erftools.preprocessing.nwpdata import NWPDataset

class GFSDataset(NWPDataset):
    """NCAR Global Forecast System (GFS) global analysis data

    Extra Init Parameters
    ---------------------
    product: str or float, optional
        NCAR RDA identifier, e.g., 'd083002' or 'd084001'. This will be
        automatically chosen if not specified
    """

    def _setup(self, **kwargs):
        # get name of RDA data product
        default_product = 'forecast' if self.forecast > 0 else 'final'
        self.product = kwargs.get('product', default_product)

        # get urls and filenames, which differ depending on the
        # analysis product
        self.datetimes, self.urls, self.filenames = construct_urls_filenames(
            self.analysis_datetime,
            self.forecast,
            self.product)


def construct_urls_filenames(datetime, forecast, product):
    rda_prefix = 'https://data-osdf.rda.ucar.edu/ncar/rda'

    if product.lower() in ['forecast', 'd084001', 84.1]:
        # Historical forecast data (0.25 deg x 0.25 deg grids; runs every 6h;
        # forecasts times every 3h up to 240h, every 12h up to 384h)
        assert datetime.hour % 6 == 0, 'Invalid analysis time'

        f_hrs = np.arange(0, min(forecast, 240)+1, 3)
        if forecast > 240:
            extend = np.arange(252, min(forecast, 384)+1, 12)
            f_hrs = np.concatenate((f_hrs, extend))

        filename = datetime.strftime('gfs.0p25.%Y%m%d%H.f{:03d}.grib2')
        url = datetime.strftime(f'{rda_prefix}/d084001/%Y/%Y%m%d/{filename}')

    elif product.lower() in ['fnl', 'final', 'd083003', 83.3]:
        # Final reanalaysis data (0.25 deg x 0.25 deg grids, every 6h)
        assert datetime.hour % 6 == 0, 'Invalid analysis time'

        f_hrs = np.arange(0, min(forecast, 9)+1, 3)

        filename = datetime.strftime('gdas1.fnl0p25.%Y%m%d%H.f{:02d}.grib2')
        url = datetime.strftime(f'{rda_prefix}/d083003/%Y/%Y%m/{filename}')

    datetimes = datetime + pd.to_timedelta(f_hrs, unit='h')
    urls = [url.format(fhr) for fhr in f_hrs]
    filenames = [filename.format(fhr) for fhr in f_hrs]

    return datetimes, urls, filenames
