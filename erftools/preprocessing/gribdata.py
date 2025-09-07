import numpy as np
import pygrib

class GribData(object):
    """Container for 3D data read from GRIB files"""

    def __init__(self, **varmap):
        """Initialize container given a mapping between internal
        variable name (keys) and the GRIB variable name (values).
        """
        self.pressure_levels = {} # 1D
        self.lats = None # 2D
        self.lons = None # 2D
        self.varmap = varmap
        self.vars = varmap.keys()

        for localvar in self.vars:
            self.pressure_levels[localvar] = []
            setattr(self, localvar, [])

    def read(self, gribfile):
        """Read all variables with defined keys in the varmap"""
        printed_time = False
        with pygrib.open(gribfile) as grbs:
            for grb in grbs:
                if not printed_time:
                    year = grb.year
                    month = grb.month
                    day = grb.day
                    hour = grb.hour
                    minute = grb.minute if hasattr(grb, 'minute') else 0
                    forecast_hour = grb.forecastTime

                    print(f'Date: {year}-{month:02d}-{day:02d}, Time: {hour:02d}:{minute:02d} UTC')
                    date_time_forecast_str = f'{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}_{forecast_hour:03d}'
                    print(f'Datetime string: {date_time_forecast_str}')
                    printed_time = True

                # Retrieve latitude and longitude grids only once
                if (self.lats is None) or (self.lons is None):
                    self.lats, self.lons = grb.latlons()
                
                # Retrieve all variables in the variable mapping
                for localvar, gribvar in self.varmap.items():
                    if gribvar == grb.name:
                        arr = getattr(self, localvar)
                        arr.append(grb.values)
                        self.pressure_levels[localvar].append(grb.level)
                        break

        # NOTE: These levels are not necessarily monotonically increasing, nor finite!
        for varn, lvls in self.pressure_levels.items():
            self.pressure_levels[varn] = np.array(lvls)

        # Stack into a 3D array (level, lat, lon)
        # NOTE: These fields do not necessarily all have the same number of levels!
        for varn in self.vars:
            data = getattr(self, varn)
            setattr(self, varn, np.stack(data, axis=0))

    def clip(self, lat_slice, lon_slice):
        for varn in self.vars:
            arr = getattr(self, varn)
            setattr(self, varn, arr[:, lat_slice, lon_slice])

    def sizes(self):
        for varn in self.vars:
            arr = getattr(self, varn)
            nlevels = arr.shape[0]
            info = f'{varn:8s} : shape={arr.shape}' 
            if varn in self.pressure_levels:
                plevels = self.pressure_levels[varn]
                #info += f' levels={plevels}'
                assert len(plevels) == nlevels
            print(info)
