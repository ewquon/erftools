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
        default_product = 'forecast' if self.forecast > 0 else 'final'
        self.product = kwargs.get('product', default_product)

    def download(self):
        print('doing something')

