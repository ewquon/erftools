import numpy as np
import xarray as xr

from ..constants import CONST_GRAV
from ..utils import destagger

def interp_zlevels(ds,zlevels_stag,ph='PH',phb='PHB',check_ph=True,dtype=float):
    """Vertical interpolation on a WRF-like dataset to staggered zlevels AGL"""
    zinp = (ds[ph] + ds[phb]) / CONST_GRAV
    zinp = zinp.squeeze() # drop Time dim
    zinp_unstag = destagger(zinp)
    zsurf = zinp.isel(bottom_top_stag=0)
    zsurf0 = zsurf.values.ravel()[0]
    assert np.all(zsurf == zsurf0), \
        'Need to implement more sophisiticated interpolation for general terrain'

    zlevels = 0.5 * (zlevels_stag[1:] + zlevels_stag[:-1])

    varlist = list(ds.data_vars)
    stagvars = [varn for varn in varlist if 'bottom_top_stag' in ds[varn].dims]
    unstagvars = [varn for varn in varlist if 'bottom_top' in ds[varn].dims]

    def interpfun(zvals, columndata, zinterp):
        """note: core dim is moved to end
        expected `zvals` dims are (south_north, west_east, bottom_top*)
        expected `columndata` dims are (Time, south_north, west_east, bottom_top*)
        `zinterp` should be a numpy array
        """
        #f = interp1d(zvals, columndata, kind=method, axis=-1,
        #             bounds_error=True, assume_sorted=True)
        #return f(zinterp)
        return np.interp(zinterp, zvals, columndata)

    print('Interpolating staggered vars:', stagvars)
    interp_stag = xr.apply_ufunc(
        interpfun,
        zinp, ds[stagvars],
        input_core_dims=[['bottom_top_stag'], ['bottom_top_stag']],
        output_core_dims=[['INTERP_bottom_top_stag']],
        output_dtypes=[dtype],
        kwargs={'zinterp': zlevels_stag + zsurf0},
        vectorize=True,
    )
    assert np.all(np.isfinite(interp_stag).all()), \
        'Interpolating to staggered z levels failed'

    # interpolated PH, PHB should correspond to the interpolated levels...
    if check_ph:
        print('Checking interpolated geopotential')
        zstag_new = (interp_stag['PH'] + interp_stag['PHB']) / 9.81 - zsurf0
        assert zinp.dims[0] == 'bottom_top_stag'
        zstag_new = zstag_new.squeeze()
        zstag_new = zstag_new.rename(INTERP_bottom_top_stag='bottom_top_stag')
        zstag_new = zstag_new.transpose(*list(zinp.dims)).values
        z_new = 0.5 * (zstag_new[1:,...] + zstag_new[:-1,...])
        relerr = (z_new - zlevels[:,np.newaxis,np.newaxis]) / z_new
        assert np.all(np.abs(relerr) < 1e-6)

    # interpolate unstaggered vars
    print('Interpolating unstaggered vars:', unstagvars)
    interp_unstag = xr.apply_ufunc(
        interpfun,
        zinp_unstag, ds[unstagvars],
        input_core_dims=[['bottom_top'], ['bottom_top']],
        output_core_dims=[['INTERP_bottom_top']],
        output_dtypes=[dtype],
        kwargs={'zinterp': zlevels + zsurf0},
        vectorize=True,
    )
    assert np.all(np.isfinite(interp_unstag).all()), \
        'Interpolating to unstaggered z levels failed'

    # update original dataset
    print(interp_unstag)
    print(interp_stag)
    interp_unstag = interp_unstag.rename(INTERP_bottom_top='bottom_top')
    interp_stag = interp_stag.rename(INTERP_bottom_top_stag='bottom_top_stag')
    for varn,da in interp_unstag.items():
        dims = ds[varn].dims
        ds[varn] = da.transpose(*dims)
    for varn,da in interp_stag.items():
        dims = ds[varn].dims
        ds[varn] = da.transpose(*dims)

    return ds

