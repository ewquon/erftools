import numpy as np
import xarray as xr

def get_stag_dims(ds_cc, dim_to_stagger='bottom_top'):
    """Return dict of dataset dimensions with the specified dimension
    renamed and increased in size by 1
    """
    stag_dims = {}
    for dim,size in ds_cc.sizes.items():
        if dim==dim_to_stagger:
            stag_dims[dim+'_stag'] = size+1
        else:
            stag_dims[dim] = size
    return stag_dims

def get_lo_faces(da,dim='bottom_top_stag'):
    """Get low faces and rename staggered dimension, for destaggering"""
    assert dim.endswith('_stag')
    return da.isel({dim:slice(0,-1)}).rename({dim:dim[:-5]})

def get_hi_faces(da,dim='bottom_top_stag'):
    """Get high faces and rename staggered dimension, for destaggering"""
    assert dim.endswith('_stag')
    return da.isel({dim:slice(1,None)}).rename({dim:dim[:-5]})

def destagger(da,dim='bottom_top_stag'):
    """Interpolate staggered field to cell-centered locations."""
    assert dim in da.dims
    hi = get_hi_faces(da,dim)
    lo = get_lo_faces(da,dim)
    return 0.5*(hi + lo)

def stagger(da,dim='bottom_top_stag',loval=0.0,hival=None):
    """Interpolate cell-centered field to staggered z locations on the
    interior; extrapolate to domain top; and set the surface value.
    """
    assert 'z' in da.dims, 'Expected unstaggered field'
    assert 'zstag' not in da.dims, 'Should not have both z and zstag'

    zbot = da.coords['zstag'].values[0]
    ztop = da.coords['zstag'].values[-1]

    top = 1.5*da.isel(z=-1) - 0.5*da.isel(z=-2)
    top = top.expand_dims({'z':[ztop]})

    interior = xr.DataArray(
        0.5*(  da.isel(z=slice(0,  -1)).values
             + da.isel(z=slice(1,None)).values),
        coords={'t': ds.coords['t'],
                'z': ds.coords['zstag'].values[1:-1]},
        dims=('t','z')
    )

    surface = surfval * xr.ones_like(top)
    surface = surface.assign_coords(z=[zbot])

    da = xr.concat((surface,interior,top), dim='z')
    da = da.transpose('t','z')
    da = da.rename(z='zstag')
    return da
