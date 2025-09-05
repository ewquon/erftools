import os
import tqdm

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

        # get urls and filenames
        urls_fnames = [construct_url_filename(dt, self.product)
                       for dt in self.datetimes]
        self.urls = [url_fname[0] for url_fname in urls_fnames]
        self.filenames = [url_fname[1] for url_fname in urls_fnames]

    def download(self, dpath='.'):
        for url,filename in zip(self.urls, self.filenames):
            fpath = os.path.join(dpath, filename)
            if os.path.isfile(fpath):
                print(f'{fpath} found')
            else:
                self._download_with_progress(url, filename)



def construct_url_filename(datetime, product):
    rda_prefix = 'https://data-osdf.rda.ucar.edu/ncar/rda'

    if product.lower() in ['forecast', 'd084001', 84.1]:
        # Historical forecast data (0.25 deg x 0.25 deg grids, every 3h)
        filename = datetime.strftime('gfs.0p25.%Y%m%d%H.f000.grib2')
        url = datetime.strftime(f'{rda_prefix}/d084001/%Y/%Y%m%d/{filename}')

    elif product.lower() in ['fnl', 'final', 'd083003', 83.3]:
        # Final reanalaysis data (0.25 deg x 0.25 deg grids, every 6h)
        filename = datetime.strftime('gdas1.fnl0p25.%Y%m%d%H.f00.grib2')
        url = datetime.strftime(f'{rda_prefix}/d083003/%Y/%Y%m/{filename}')

    return url, filename
