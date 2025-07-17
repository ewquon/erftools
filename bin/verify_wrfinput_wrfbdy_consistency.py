#!/usr/bin/env python
"""
Perform sanity checks for wrfinput, wrfbdy data. This script will throw an
assertion error if anything is in disagreement.
"""
import sys
import numpy as np
import xarray as xr

from erftools.utils.wrf import get_mass_weighted

def checkfields(fld1,fld2):
    if np.all(fld1 == fld2):
        print(f'  {fld1.name} and {fld2.name} are EQUAL')
    else:
        #assert np.allclose(fld1,fld2), f'{fld1.name} and {fld2.name} DIFFER'
        if np.allclose(fld1,fld2):
            absdiff = np.abs(fld2-fld1)
            maxabs = np.max(absdiff)
            maxrel = np.max(absdiff/np.abs(fld1))
            maxidx = np.unravel_index(absdiff.argmax(), absdiff.shape)
            print(f'  {fld1.name} and {fld2.name} are CLOSE... max diffs:',
                  maxabs.item(), maxrel.item(), 'at', maxidx)
        else:
            print(f'  {fld1.name} and {fld2.name} DIFFER')

if len(sys.argv) <= 2:
    sys.exit(f'USAGE: {sys.argv[0]} wrfinput wrfbdy')
inpfile = sys.argv[1]
bdyfile = sys.argv[2]

inp = xr.open_dataset(inpfile).isel(Time=0)
bdy = xr.open_dataset(bdyfile)

bdywidth = bdy.sizes['bdy_width']

print('Verifying direct correspondence between MU fields')
print(bdy['MU_BXS'].sizes)
print(bdy['MU_BXE'].sizes)
print(bdy['MU_BYS'].sizes)
print(bdy['MU_BYE'].sizes)
for bdyoff in range(bdywidth):
    print('bdy_width=',bdyoff)
    checkfields(inp['MU'].isel(west_east=bdyoff),
                bdy['MU_BXS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(inp['MU'].isel(west_east=-(bdyoff+1)),
                bdy['MU_BXE'].isel(Time=0,bdy_width=bdyoff))
    checkfields(inp['MU'].isel(south_north=bdyoff),
                bdy['MU_BYS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(inp['MU'].isel(south_north=-(bdyoff+1)),
                bdy['MU_BYE'].isel(Time=0,bdy_width=bdyoff))

print('\nVerifying mass weighting in U,V')
print('U_BXS',bdy['U_BXS'].sizes)
print('U_BXE',bdy['U_BXE'].sizes)
print('U_BYS',bdy['U_BYS'].sizes)
print('U_BYE',bdy['U_BYE'].sizes)
print('V_BXS',bdy['V_BXS'].sizes)
print('V_BXE',bdy['V_BXE'].sizes)
print('V_BYS',bdy['V_BYS'].sizes)
print('V_BYE',bdy['V_BYE'].sizes)
for bdyoff in range(bdywidth):
    print('bdy_width=',bdyoff)
    checkfields(get_mass_weighted('U',inp,west_east_stag=bdyoff),
                bdy['U_BXS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('U',inp,west_east_stag=-(bdyoff+1)),
                bdy['U_BXE'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('U',inp,south_north=bdyoff),
                bdy['U_BYS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('U',inp,south_north=-(bdyoff+1)),
                bdy['U_BYE'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('V',inp,west_east=bdyoff),
                bdy['V_BXS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('V',inp,west_east=-(bdyoff+1)),
                bdy['V_BXE'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('V',inp,south_north_stag=bdyoff),
                bdy['V_BYS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('V',inp,south_north_stag=-(bdyoff+1)),
                bdy['V_BYE'].isel(Time=0,bdy_width=bdyoff))

print('\nVerifying mass weighting in T')
print('T_BXS',bdy['T_BXS'].sizes)
print('T_BXE',bdy['T_BXE'].sizes)
print('T_BYS',bdy['T_BYS'].sizes)
print('T_BYE',bdy['T_BYE'].sizes)
T0 = 300.0
inp['T'] += T0
bdy['T_BXS'] += T0
bdy['T_BXE'] += T0
bdy['T_BYS'] += T0
bdy['T_BYE'] += T0
for bdyoff in range(bdywidth):
    print('bdy_width=',bdyoff)
    checkfields(get_mass_weighted('T',inp,west_east=bdyoff),
                bdy['T_BXS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('T',inp,west_east=-(bdyoff+1)),
                bdy['T_BXE'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('T',inp,south_north=bdyoff),
                bdy['T_BYS'].isel(Time=0,bdy_width=bdyoff))
    checkfields(get_mass_weighted('T',inp,south_north=-(bdyoff+1)),
                bdy['T_BYE'].isel(Time=0,bdy_width=bdyoff))

print('\nVerifying mass weighting in scalar fields')
vars_to_check = ['QVAPOR','QCLOUD','QRAIN']
for bdyvarn in vars_to_check:
    print('')
    print(f'{bdyvarn}_BXS',bdy[f'{bdyvarn}_BXS'].sizes)
    print(f'{bdyvarn}_BXE',bdy[f'{bdyvarn}_BXE'].sizes)
    print(f'{bdyvarn}_BYS',bdy[f'{bdyvarn}_BYS'].sizes)
    print(f'{bdyvarn}_BYE',bdy[f'{bdyvarn}_BYE'].sizes)
    for bdyoff in range(bdywidth):
        print('bdy_width=',bdyoff)
        inpvarn = 'THM' if bdyvarn=='T' else bdyvarn
        #print('Checking',inpvarn,bdyvarn)
        checkfields(get_mass_weighted(inpvarn,inp,west_east=bdyoff),
                    bdy[bdyvarn+'_BXS'].isel(Time=0,bdy_width=bdyoff))
        checkfields(get_mass_weighted(inpvarn,inp,west_east=-(bdyoff+1)),
                    bdy[bdyvarn+'_BXE'].isel(Time=0,bdy_width=bdyoff))
        checkfields(get_mass_weighted(inpvarn,inp,south_north=bdyoff),
                    bdy[bdyvarn+'_BYS'].isel(Time=0,bdy_width=bdyoff))
        checkfields(get_mass_weighted(inpvarn,inp,south_north=-(bdyoff+1)),
                    bdy[bdyvarn+'_BYE'].isel(Time=0,bdy_width=bdyoff))
