import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

class AveragedProfiles(object):
    """Process text diagnostic profiles that were written out with:
        ```
        erf.v           = 1
        erf.data_log    = scalars.txt profiles1.txt profiles2.txt profiles3.txt
        erf.profile_int = 100  # output interval
        ```
    These text files include:
    1. Time history of surface quantities (u*, θ*, L)
    2. Time history of mean profiles (ubar, vbar, wbar, thetabar, ...)
    3. Time history of resolved-scale stress profiles (u'u', u'v', u'w', ...)
    4. Time history of subfilter-scale (SFS) stress profiles (tau11, tau12,
       tau13, ...)

    All three horizontal-averaged profile output files need to be specified to
    have complete averaged profile data. Note that AveragedProfiles does not
    process the scalar surface time-history.
    """
    timename = 't' # 'time'
    heightname = 'z' # 'height'

    # output columns -- these must match the output order
    profile1vars = ['u','v','w',
                    'ρ','θ','e',
                    'Kmv','Khv',
                    'qv','qc','qr',
                    'qi','qs','qg']
    profile2vars = ["u'u'", "u'v'", "u'w'",
                    "v'v'", "v'w'", "w'w'",
                    "θ'u'", "θ'v'", "θ'w'", "θ'θ'",
                    "ui'ui'u'", "ui'ui'v'", "ui'ui'w'",
                    "p'u'", "p'v'", "p'w'",
                    "w'qv'", "w'qc'", "w'qr'",
                    "θv'w'"]
    profile3vars = ['τ11','τ12','τ13',
                    'τ22','τ23','τ33',
                    'τθw', "τqvw", "τqcw",'ε']

    # these variables will be assigned the staggered vertical coordinate
    staggeredvars = ["w",
                     "u'w'", "v'w'", "w'w'",
                     "ui'ui'w'",
                     "θ'w'", "θv'w'", "p'w'", "k'w'",
                     "w'qv'", "w'qc'", "w'qr'",
                     "τ13", "τ23",
                     "τθw", "τqvw", "τqcw"]

    def __init__(self, *args, t0=0.0, sampling_interval_s=None, zexact=None,
                 verbose=True):
        """Load diagnostic profile data from 3 datafiles

        Parameters
        ----------
        *args : list or glob string
            Averaged profile datafiles to load
        t0 : float, optional
            With `sampling_interval_s`, used to overwrite the
            timestamps in the text data to address issues with
            insufficient precision
        sampling_interval_s : float, optional
            Overwrite the time dimension coordinate with
            t0 + np.arange(Ntimes)*sampling_interval_s
        zexact : array-like, optional
            List of cell-centered heights in the computational
            domain, used to overwrite the height dimension coordinate
            and address issues with insufficient precision
        """
        self.verbose = verbose
        assert (len(args) == 1) or (len(args) == 3)
        if len(args) == 1:
            if isinstance(args[0], list):
                fpathlist = args[0]
                assert len(fpathlist)<=3, \
                    'Expected list of 3 separate profile datafiles'
            else:
                assert isinstance(args[0], str)
                fpathlist = sorted(glob.glob(args[0]))
                assert len(fpathlist)<=3, \
                    f'Expected to find 3 files, found {len(fpathlist)} {fpathlist}'
        else:
            fpathlist = args
        self._load_profiles(*fpathlist)
        if sampling_interval_s is not None:
            texact = t0 \
                   + (np.arange(self.ds.dims[self.timename])+1) * sampling_interval_s
            self.ds = self.ds.assign_coords({self.timename: texact})
        if zexact is not None:
            self.ds = self.ds.assign_coords({self.heightname: zexact})
        self._process_staggered()

    def _read_text_data(self, fpath, columns):
        df = pd.read_csv(
            fpath, delim_whitespace=True,
            header=None, names=columns,
            dtype=np.float64)
        df = df.set_index([self.timename,self.heightname])
        isdup = df.index.duplicated(keep='last')
        return df.loc[~isdup]

    def _load_profiles(self, mean_fpath, flux_fpath=None, sfs_fpath=None):
        alldata = []
        idxvars = [self.timename, self.heightname]
        assert os.path.isfile(mean_fpath)
        if self.verbose: print('Loading mean profiles from',mean_fpath)
        mean = self._read_text_data(mean_fpath, idxvars+self.profile1vars)
        alldata.append(mean)

        # optional profile data
        if (flux_fpath is not None) and os.path.isfile(flux_fpath):
            if self.verbose: print('Loading resolved flux profiles from',flux_fpath)
            fluxes = self._read_text_data(flux_fpath, idxvars+self.profile2vars)
            alldata.append(fluxes)
        else:
            if self.verbose: print('No resolved stress data available')
        if (sfs_fpath is not None) and os.path.isfile(sfs_fpath):
            if self.verbose: print('Loading SFS stress profiles from',sfs_fpath)
            sfs = self._read_text_data(sfs_fpath, idxvars+self.profile3vars)
            alldata.append(sfs)
        else:
            if self.verbose: print('No SFS data available')

        self.ds = pd.concat(alldata, axis=1).to_xarray()

    def _process_staggered(self):
        topval = self.ds['θ'].isel(t=-1).values[-1]
        if topval > 0:
            # profiles are not on staggered grid
            return
        elif ~np.isfinite(topval):
            # nan???
            print('**NaN found**')
            return
        assert topval == 0
        if self.verbose: print('**Staggered output detected**')
        zstag = self.ds.coords['z'].values
        zcc = 0.5 * (zstag[1:] + zstag[:-1])
        # collect cell-centered and staggered outputs that are available
        stagvars = []
        ccvars = []
        for varn in self.ds.data_vars:
            if varn in self.staggeredvars:
                stagvars.append(varn)
            else:
                ccvars.append(varn)
        # cell-centered: throw out values associated with highest z face, update coordinates
        cc = self.ds[ccvars].isel(z=slice(0,-1))
        cc = cc.assign_coords(z=zcc)
        # average the staggered values to cell centers (destagger)
        cc_destag = 0.5 * ( self.ds[stagvars].isel(z=slice(0,-1)).assign_coords(z=zcc)
                          + self.ds[stagvars].isel(z=slice(1,None)).assign_coords(z=zcc))
        cc_destag = cc_destag.assign_coords(z=zcc)
        cc_destag = cc_destag.rename_vars({var:f'{var}_destag' for var in stagvars})
        # associate staggered outputs with new coordinate
        stag = self.ds[stagvars]
        stag = stag.rename(z='zstag')
        # combine into single dataset
        self.ds = xr.merge([cc,cc_destag,stag])

    def __getitem__(self,key):
        return self.ds[key]

    def calc_ddt(self,*args):
        """Calculate time derivative, based on the given profile output
        interval.
        """
        dt = self.ds.coords[self.timename][1] - self.ds.coords[self.timename][0]
        print('dt=',dt.values)
        varlist = args if len(args) > 0 else self.profile1vars
        for varn in varlist:
            self.ds[f'd{varn}/dt'] = self.ds[varn].diff(self.timename) / dt

    def calc_grad(self,*args):
        """Calculate vertical gradient"""
        dz = self.ds.coords[self.heightname][1] - self.ds.coords[self.heightname][0]
        print('dz=',dz.values)
        allvars = self.profile1vars + self.profile2vars + self.profile3vars
        varlist = args if len(args) > 0 else allvars
        for varn in varlist:
            self.ds[f'd{varn}/dz'] = self.ds[varn].diff(self.heightname) / dz

    def calc_stress(self):
        """Calculate total stresses (note: τ are deviatoric stresses)"""
        trace = self.ds['τ11'] + self.ds['τ22'] + self.ds['τ33']
        assert np.abs(trace).max() < 1e-14
        self.ds['uu_tot'] = self.ds["u'u'"] + self.ds['τ11'] + 2./3.*self.ds['e']
        self.ds['vv_tot'] = self.ds["v'v'"] + self.ds['τ22'] + 2./3.*self.ds['e']
        self.ds['ww_tot'] = self.ds["w'w'"] + self.ds['τ33'] + 2./3.*self.ds['e']
        self.ds['uv_tot'] = self.ds["u'v'"] + self.ds['τ12']
        self.ds['uw_tot'] = self.ds["u'w'"] + self.ds['τ13']
        self.ds['vw_tot'] = self.ds["v'w'"] + self.ds['τ23']
        self.ds['ustar'] = (self.ds['uw_tot']**2 + self.ds['vw_tot']**2)**0.25
        self.ds['hfx'] = self.ds["θ'w'"] + self.ds['τθw']
