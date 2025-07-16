import numpy as np
import xarray as xr

from ..constants import CONST_GRAV
from ..utils import destagger, stagger

def interp_zlevels(ds,zlevels_stag,
                   ph='PH',phb='PHB',
                   xlo=False,xhi=False,ylo=False,yhi=False,
                   check_ph=True,
                   dtype=float):
    """Vertical interpolation on a WRF-like dataset to staggered zlevels AGL"""
    # requested heights
    zlevels_unstag = 0.5 * (zlevels_stag[1:] + zlevels_stag[:-1])

    # get geopotential heights from the dataset
    zinp_stag = (ds[ph] + ds[phb]) / CONST_GRAV
    zinp_stag = zinp_stag.squeeze() # drop Time dim
    assert 'bottom_top_stag' in zinp_stag.dims, \
        f'Unexpected zinp_stag dims: {zinp_stag.dims}'

    zsurf = zinp_stag.isel(bottom_top_stag=0)
    zsurf0 = zsurf.values.ravel()[0]
    assert np.all(zsurf == zsurf0), \
        'Need to implement more sophisiticated interpolation for general terrain'

    # get corresponding geopotential heights for unstaggered or horizontally
    # staggered fields
    zinp_unstag = destagger(zinp_stag, dim='bottom_top_stag')
    zinp_u = stagger(zinp_unstag, dim='west_east')
    if xlo:
        zinp_u = zinp_u.isel(west_east_stag=slice(0,-1))
    elif xhi:
        zinp_u = zinp_u.isel(west_east_stag=slice(1,None))
    zinp_v = stagger(zinp_unstag, dim='south_north')
    if ylo:
        zinp_v = zinp_v.isel(south_north_stag=slice(0,-1))
    elif yhi:
        zinp_v = zinp_v.isel(south_north_stag=slice(1,None))

    # sort vars by dimensions
    fieldtypes = {}
    for varn, da in ds.items():
        dims = tuple(da.dims)
        have_vert = False
        for dim in dims:
            if dim.startswith('bottom_top'):
                have_vert = True
                break
        if not have_vert:
            continue
        if not dims in fieldtypes.keys():
            fieldtypes[dims] = [varn]
        else:
            fieldtypes[dims].append(varn)

    # don't overwrite original data
    ds = ds.copy(deep=True)

    # linearly interpolate each set of variables, which will include vertical
    # dimension bottom_top or bottom_top_stag
    def interpfun(zvals, columndata, zinterp):
        #f = interp1d(zvals, columndata, kind=method, axis=-1,
        #             bounds_error=True, assume_sorted=True)
        #return f(zinterp)
        return np.interp(zinterp, zvals, columndata)

    # now, actually do the interpolation on each group of fields with the same
    # dimensions
    for dims,varlist in fieldtypes.items():
        dimlist = list(dims)
        if 'Time' in dimlist:
            dimlist.remove('Time')
        is_column_func = (len(dimlist)==1 and dimlist[0].startswith('bottom_top'))
        if 'bottom_top_stag' in dims:
            print('Interpolating staggered vars with', dims,':',varlist)
            if is_column_func:
                zinp = zinp_stag.mean(['south_north','west_east'])
            else:
                zinp = zinp_stag
            zout = zlevels_stag + zsurf0
            vert_dim = 'bottom_top_stag'
        else:
            assert 'bottom_top' in dims
            print('Interpolating unstaggered vars with', dims,':',varlist)
            if 'U' in varlist:
                zinp = zinp_u
            elif 'V' in varlist:
                zinp = zinp_v
            elif is_column_func:
                zinp = zinp_unstag.mean(['south_north','west_east'])
            else:
                zinp = zinp_unstag
            zout = zlevels_unstag + zsurf0
            vert_dim = 'bottom_top'
        new_vert_dim = f'NEW_{vert_dim}'
        zout = xr.DataArray(zout, dims=new_vert_dim)

        interp_fields = xr.apply_ufunc(
            interpfun,
            zinp, ds[varlist], zout, # args for interpfun
            input_core_dims=[[vert_dim], [vert_dim], [new_vert_dim]],
            output_core_dims=[[new_vert_dim]],
            output_dtypes=[dtype],
            output_sizes={new_vert_dim: len(zinp)},
            vectorize=True,
            dask='parallelize',
        )
        assert np.all(np.isfinite(interp_fields).all()), 'Interpolation failed'

        dims = list(dims)
        dims[dims.index(vert_dim)] = new_vert_dim
        for varn,da in interp_fields.items():
            ds[varn] = da.transpose(*dims)

    ds = ds.rename(NEW_bottom_top='bottom_top',
                   NEW_bottom_top_stag='bottom_top_stag')

    # interpolated PH, PHB should correspond to the interpolated levels...
    if check_ph:
        print('Checking interpolated geopotential')
        zstag_new = (ds[ph] + ds[phb]) / CONST_GRAV - zsurf0
        zstag_new = zstag_new.squeeze()
        zstag_new = zstag_new.transpose(*list(zinp_stag.dims)).values
        z_new = 0.5 * (zstag_new[1:,...] + zstag_new[:-1,...])
        relerr = (z_new - zlevels_unstag[:,np.newaxis,np.newaxis]) / z_new
        assert np.all(np.abs(relerr) < 1e-6)

    return ds

