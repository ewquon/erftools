import sys
import numpy as np
import contextlib
import platform
from collections.abc import MutableMapping


# https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
@contextlib.contextmanager
def smart_open(fpath=None):
    if (fpath is not None):
        f = open(fpath, 'w')
    else:
        f = sys.stdout
    try:
        yield f
    finally:
        if f is not sys.stdout:
            f.close()


class ERFInputFile(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.pop('verbose',True)
        self.store = dict({
            'amr.refinement_indicators': '',
            'zhi.type': 'SlipWall',
            # retrieved from wrfinput_d01 
            'erf.most.z0': None,
            'erf.most.surf_temp': None,
            'erf.latitude': 90.0,
            'erf.rotational_time_period': 86400.0,
            # estimated quantities
            'erf.z_levels': [],  # can estimate from wrfinput_d01
        })
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __str__(self):
        #return '\n'.join([f'{key} = {str(val)}' for key,val in self.items()])
        s = ''
        for key,val in self.items():
            if isinstance(val, (list,tuple,np.ndarray)):
                val = ' '.join([str(v) for v in val])
            s += f'{key} = {str(val)}\n'
        return s.rstrip()

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        try:
            oldval = self[key]
        except KeyError:
            pass
        else:
            if self.verbose:
                print(f'Overriding existing `{key}` with {value}')
        finally:
            self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)
    
    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def write(self,fpath=None,ideal=False):
        inputs = self.store.copy()

        refinement_boxes = ''
        refinement_indicators = inputs.pop('amr.refinement_indicators')
        boxes = refinement_indicators.split()
        for box in boxes:
            loindices = ' '.join([str(val) for val in inputs.pop(f'amr.{box}.in_box_lo')])
            hiindices = ' '.join([str(val) for val in inputs.pop(f'amr.{box}.in_box_hi')])
            refinement_boxes += f'amr.{box}.in_box_lo = {loindices}\n' 
            refinement_boxes += f'amr.{box}.in_box_hi = {hiindices}\n' 

        with smart_open(fpath) as f:

            f.write('# ------------------  INPUTS TO MAIN PROGRAM  -------------------\n')
            f.write(f"""# generated by https://github.com/erf-model/erftools

stop_time          = {inputs.pop('stop_time')}

""")
            if platform.system() == 'Darwin':
                f.write('amrex.fpe_trap_invalid = 0\n')
            else:
                f.write('amrex.fpe_trap_invalid = 1\n')
            f.write(f"""
fabarray.mfiter_tile_size = 1024 1024 1024

# PROBLEM SIZE & GEOMETRY
amr.n_cell           = {' '.join([str(v) for v in inputs.pop('amr.n_cell')])}
geometry.prob_extent = {' '.join([str(v) for v in inputs.pop('geometry.prob_extent')])}
geometry.is_periodic = {' '.join([str(int(b)) for b in inputs.pop('geometry.is_periodic')])}
""")
            zlevels = inputs.pop('erf.z_levels')
            if len(zlevels) > 0:
                f.write(f"""
#erf.z_levels = {' '.join([str(v) for v in zlevels])}  # TODO: need to implement this input
""")
            max_level = inputs.pop('amr.max_level')
            f.write(f"""
# TIME STEP CONTROL
erf.fixed_dt       = {inputs.pop('erf.fixed_dt')}  # fixed time step depending on grid resolution

# REFINEMENT / REGRIDDING
amr.max_level      = {max_level}  # maximum level number allowed
""")
            if refinement_indicators != '':
                f.write("""
amr.ref_ratio_vect = {' '.join([str(v) for v in inputs.pop('amr.ref_ratio_vect')])}
amr.refinement_indicators = {refinement_indicators}
{refinement_boxes.rstrip()}
""")

            f.write('\n# BOUNDARY CONDITIONS\n')
            BCtypes = [
                f'{xyz}{lohi}.type'
                for xyz in ['x','y','z']
                for lohi in ['lo','hi']]
            BCs_to_write = [name for name in BCtypes
                            if name in inputs.keys()]
            for bcname in BCs_to_write:
                bcdef = inputs.pop(bcname)
                if bcname=='zlo.type':
                    zlo_type = bcdef
                f.write(f'{bcname} = "{bcdef}"\n')

            if zlo_type.upper() == 'MOST':
                assert inputs['erf.most.z0'] is not None
                assert inputs['erf.most.surf_temp'] is not None
                f.write(f"""erf.most.z0 = {inputs.pop('erf.most.z0')}  # TODO: use roughness map
erf.most.surf_temp = {inputs.pop('erf.most.surf_temp')}  # TODO: use surface temperature map
""")

            if ideal:
                f.write("""
# INITIAL CONDITIONS
erf.init_type           = "input_sounding"
erf.init_sounding_ideal = 1
""")
            else:
                bdylist = ' '.join([f'"wrfbdy_d{idom+1:02d}"'
                                    for idom in
                                    range(max_level+1)])
                f.write(f"""
# INITIAL CONDITIONS
erf.init_type    = "real"
erf.nc_init_file = "wrfinput_d01"
erf.nc_bdy_file  = {bdylist}
""")

            f.write(f"""
# PHYSICS OPTIONS
erf.les_type = "{inputs.pop('erf.les_type')}"  # TODO: specify for each level
erf.pbl_type = "{inputs.pop('erf.pbl_type')}"  # TODO: specify for each level
erf.abl_driver_type = "None"
erf.use_gravity = true
erf.use_coriolis = true
erf.latitude = {inputs.pop('erf.latitude')}
erf.rotational_time_period = {inputs.pop('erf.rotational_time_period')}
""")

            molec_diff_type = inputs.pop('erf.molec_diff_type')
            if molec_diff_type.lower() != "none":
                f.write(f"""
erf.molec_diff_type = "{molec_diff_type}"
erf.rho0_trans = {inputs.pop('erf.rho0_trans')}  # i.e., dynamic == kinematic coefficients
erf.dynamicViscosity = {inputs.pop('erf.dynamicViscosity')}  # TODO: specify for each level
erf.alpha_T = {inputs.pop('erf.alpha_T')}  # TODO: specify for each level
erf.alpha_C = {inputs.pop('erf.alpha_C')}  # TODO: specify for each level
""")

            f.write(f"""
# DIAGNOSTICS & VERBOSITY
erf.sum_interval = 1  # timesteps between computing mass
erf.v            = 1  # verbosity in ERF.cpp
amr.v            = 1  # verbosity in Amr.cpp

""")
            if len(inputs.keys()) > 0:
                f.write('# OTHER PARAMS\n')
                for key,val in inputs.items():
                    f.write(f'{key} = {val}\n')

        if self.verbose and (fpath is not None):
            print('Wrote',fpath)
            
