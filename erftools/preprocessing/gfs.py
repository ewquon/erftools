import os

import tqdm
import numpy as np
import pandas as pd
from pyproj import Transformer

from erftools.preprocessing.nwpdata import NWPDataset
from erftools.preprocessing.gribdata import GribData
from erftools.constants import CONST_GRAV, R_d, Cp_d
from erftools.utils.microphysics import p_sat


class GFSDataset(NWPDataset):
    """NCAR Global Forecast System (GFS) global analysis data

    Extra Init Parameters
    ---------------------
    product: str or float, optional
        NCAR RDA identifier, e.g., 'd083002' or 'd084001'. This will be
        automatically chosen if not specified
    """

    def _setup(self, **kwargs):
        """Do dataset-specific setup, validation tasks"""
        # get name of RDA data product
        default_product = 'forecast' if self.forecast > 0 else 'final'
        self.product = kwargs.get('product', default_product)

        # get urls and filenames, which differ depending on the
        # analysis product
        self.datetimes, self.urls, self.filenames = construct_urls_filenames(
            self.analysis_datetime,
            self.forecast,
            self.product)

    def read(self, clip=True):
        # Initialize data container with GRIB variable names
        self.grib = GribData(
            gh='Geopotential height',
            temp='Temperature',
            rh='Relative humidity',
            theta='Potential temperature',
            p='Pressure',
            u='U component of wind',
            v='V component of wind',
            w='Geometric vertical velocity',
            qv='Specific humidity',
            qc='Cloud mixing ratio',
            qr='Rain mixing ratio',
            vort='Absolute vorticity')

        # Read fields from the GRIB file and clip data to match simulated area
        print('NOTE: ONLY READ FIRST GRIBFILE FOR NOW') # TODO
        #self.grib.read(self.filenames[0],
        #               filter_level_type=['isobaricInPa','isobaricInhPa'])
        self.grib.read(self.filenames[0])

        # Create projected grid
        self._create_grids(clip=clip)

        # Create fields
        self._calc_nz() # hack
        self._init_arrays()
        for k in np.arange(self.nz-1, -1, -1):
            self._fill_arrays_at_level(k)
        self._set_scalars()

    def _create_grids(self, clip=True):
        """Create projected grid with coordinates (x_grid, y_grid) with
        dimensions nx, ny, nz
        """
        # Extract unique latitude and longitude values
        unique_lats = np.unique(self.grib.lats[:, 0])  # Take the first column for unique latitudes
        unique_lons = np.unique(self.grib.lons[0, :])  # Take the first row for unique longitudes

        print("Min max lat lons are ", unique_lats[0], unique_lats[-1], unique_lons[0], unique_lons[-1])

        nlats = len(unique_lats)
        nlons = len(unique_lons)

        lat_max = self.area[0]
        lon_min = 360.0 + self.area[1]
        lat_min = self.area[2]
        lon_max = 360.0 + self.area[3]

        print("Lat/lon min/max are ", lat_min, lat_max, lon_min, lon_max)

        # Assume regular grid
        lat_resolution = unique_lats[1] - unique_lats[0]
        lon_resolution = unique_lons[1] - unique_lons[0]

        lat_start = int((lat_min - unique_lats[0]) / lat_resolution)
        lat_end   = int((lat_max - unique_lats[0]) / lat_resolution)
        lon_start = int((lon_min - unique_lons[0]) / lon_resolution)
        lon_end   = int((lon_max - unique_lons[0]) / lon_resolution)

        domain_lats = unique_lats[lat_start:lat_end+1]
        domain_lons = unique_lons[lon_start:lon_end+1]

        lon_grid, lat_grid = np.meshgrid(domain_lons, domain_lats)

        transformer = Transformer.from_crs("EPSG:4326", self.proj, always_xy=True)
        self.x_grid, self.y_grid = transformer.transform(lon_grid, lat_grid)

        print("The min max are",(lat_start, lat_end, lon_start, lon_end))

        self.nx = len(domain_lats)
        self.ny = len(domain_lons)
        self.nz = self.grib.u.shape[0]

        print("nx, ny, nz =", self.nx, self.ny, self.nz)

        if clip:
            self.grib.clip(slice(nlats-lat_end-1, nlats-lat_start),
                           slice(lon_start, lon_end+1))

    def _calc_nz(self):
        """TODO: filter by typeOfLevel instead of guessing here"""
        ght = self.grib.gh
        prev_mean = np.mean(ght[0])  # start from the top level
        for k in range(1, ght.shape[0]):
            current_mean = np.mean(ght[k])
            print("Val is", k, current_mean)
            if current_mean >= prev_mean:
                nz_admissible = k
                print(f"Mean starts increasing at index {k}")
                break
            prev_mean = current_mean
        else:
            print("Means are strictly decreasing through all levels.")

        # GFS does not store the data of velocities on the bottom 2 levels
        # Hence doing this. This was identified by using the grb.typeLevel="isobaricInhPA"
        # when extracting the data in the loop above (in addition to using grb.name)
        # and then pringitng out and seeing that the size of ght_3d_hr3 is 33 
        # but uvel_3d_hr3 is 31
        nz = nz_admissible-2

        print("The number of lats, lons, and levels are "
              f"{self.grib.lats.shape[0]}, {self.grib.lats.shape[1]}, {nz}")

        self.nz = nz

    def _fill_arrays_at_level(self, k):
        """Fill arrays for ERF (shape: nx, ny, nz) from GRIB arrays
        (shape: nz, nlat, nlon) at the specified pressure/height level
        """
        self.z_grid[:,:,k] = self.grib.gh[k,:,:]
        print("Avg val is ", k, np.mean(self.z_grid[:,:,k]),  )

        self.uvel_3d[:,:,k] = self.grib.u[k,:,:]
        self.vvel_3d[:,:,k] = self.grib.v[k,:,:]
        try:
            self.wvel_3d[:,:,k] = self.grib.w[k,:,:]
        except IndexError:
            # TODO: check why wvel has fewer levels
            self.wvel_3d[:,:,k] = 0.0
        self.velocity_3d[:,:,k,0] = self.grib.u[k,:,:]
        self.velocity_3d[:,:,k,1] = self.grib.v[k,:,:]
        self.velocity_3d[:,:,k,2] = 0.0

        self.temp_3d[:, :, k] = self.grib.temp[k,:,:]
        self.rh_3d[:, :, k]   = self.grib.rh[k,:,:]

        self.qv_3d[:, :, k]   = self.grib.qv[k,:,:]   if k < len(self.grib.qv)   else 0.0
        self.qc_3d[:, :, k]   = self.grib.qc[k,:,:]   if k < len(self.grib.qc)   else 0.0
        self.qr_3d[:, :, k]   = self.grib.qr[k,:,:]   if k < len(self.grib.qr)   else 0.0
        self.vort_3d[:, :, k] = self.grib.vort[k,:,:] if k < len(self.grib.vort) else 0.0

        # Aliases
        T_lvl = self.temp_3d[:, :, k]
        qv_lvl = self.qv_3d[:, :, k]

        # Integrate moist hydrostatic equation
        # Assuming quantities at surface is same as the first cell
        # TODO: use actual pressure at ground or water surface
        if (k == self.nz-1):
            # TODO: check 1.6 factor
            p_lvl = (
                1000.0
                - 1000.0 / (R_d * T_lvl * (1.0 + 1.6*qv_lvl))
                  * CONST_GRAV * self.z_grid[:,:,k]
            )
        else:
            p_prev = self.pressure_3d[:, :, k+1]
            T_prev = self.temp_3d[:, :, k+1]
            qv_prev = self.qv_3d[:, :, k+1]
            delta_z = self.z_grid[:,:,k] - self.z_grid[:,:,k+1]

            p_lvl = (
                p_prev
                - p_prev / (R_d * T_prev * (1.0 + 1.6*qv_prev))
                * CONST_GRAV * delta_z
            )
        assert np.all(p_lvl > 0)
        self.pressure_3d[:, :, k] = p_lvl

        self.rhod_3d[:,:,k] = (p_lvl * 100.0
                            / (R_d * T_lvl * (1.0 + 1.6*qv_lvl)))

        self.theta_3d[:,:,k] = T_lvl * (1000.0 / p_lvl)**(R_d/Cp_d)

        ps = p_sat(T_lvl) # [hPa]
        self.qsat_3d[:,:,k] = 0.622 * ps / (p_lvl - ps)


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
