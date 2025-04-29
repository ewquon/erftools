# ERFtools
A collection of scripts for facilitating the usage of ERF.

## Examples

### Converting a WRF namelist into ERF inputs
```python
from erftools.preprocessing import WRFInputDeck
wrf = WRFInputDeck('namelist.input')
wrf.process_initial_conditions('wrfinput_d01',
                               landuse_table_path='/Users/equon/WRF/run/LANDUSE.TBL',
                               write_hgt='terrain_height.txt',
                               write_z0='roughness_height.txt')
wrf.write_inputfile('inputs')
```

### Postprocessing data logs
Data logs are output with the `erf.data_log` param and can include time histories of surface conditions and planar averaged profiles (e.g., for idealized LES simulations)
```python
from erftools.postprocessing import DataLog
log = DataLog(f'{simdir}/surf_hist.dat',
              f'{simdir}/mean_profiles.dat',
              f'{simdir}/flux_profiles.dat',
              f'{simdir}/sfs_profiles.dat')
log.calc_stress()
log.est_abl_height('max_theta_grad')
print(log.ds) # data are stored in an xarray dataset
```
