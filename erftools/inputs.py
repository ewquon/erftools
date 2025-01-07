import sys
import contextlib
import numpy as np


def parmparse(prefix, ppdata):
    return {key[len(prefix)+1:]: val
            for key,val in ppdata.items()
            if key.startswith(prefix+'.')}

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


class ERFInputs(object):
    """Input data container with validation and output"""

    def __init__(self,inpfile=None,**ppdata):
        if inpfile:
            ppdata = self.parse_input(inpfile)

        # top level inputs
        self.max_step = int(ppdata.get('max_step',-1))
        self.start_time = float(ppdata.get('start_time',0.))
        self.stop_time = float(ppdata.get('stop_time',1e34))

        # read other inputs
        self.amr = AMRParms(**parmparse('amr',ppdata))
        self.geometry = GeometryParms(**parmparse('geometry',ppdata))
        self.erf = ERFParms(**parmparse('erf',ppdata))

        # after reading geometry...
        self.read_bcs(ppdata)

        # read problem-specific inputs
        self.prob = parmparse('prob',ppdata)

        self.validate()

    def parse_input(self,fpath):
        pp = {}
        with open(fpath,'r') as f:
            for line in f:
                line = line.lstrip()
                try:
                    trailingcomment = line.index('#')
                except ValueError:
                    pass
                else:
                    line = line[:trailingcomment].rstrip()
                if line == '':
                    continue
                split = line.split('=')
                key = split[0].rstrip()
                val = split[1].lstrip()
                vals = val.split()
                if len(vals) == 1:
                    val = vals[0]
                    val = val.strip('"').strip("'")
                    pp[key] = val
                else:
                    pp[key] = vals
        return pp

    def read_bcs(self,ppdata):
        if not self.geometry.is_periodic[0]:
            assert 'xlo.type' in ppdata.keys()
            assert 'xhi.type' in ppdata.keys()
            self.xlo = parmparse('xlo',ppdata)
            self.xhi = parmparse('xhi',ppdata)
        if not self.geometry.is_periodic[1]:
            assert 'ylo.type' in ppdata.keys()
            assert 'yhi.type' in ppdata.keys()
            self.ylo = parmparse('ylo',ppdata)
            self.yhi = parmparse('yhi',ppdata)
        if not self.geometry.is_periodic[2]:
            assert 'zlo.type' in ppdata.keys()
            assert 'zhi.type' in ppdata.keys()
            self.zlo = parmparse('zlo',ppdata)
            self.zhi = parmparse('zhi',ppdata)
            if self.zlo['type'] == 'MOST':
                self.most = parmparse('erf.most',ppdata)

    def validate(self):
        # additional validation that depends on different parmparse types
        if self.erf.use_terrain:
            if self.erf.terrain_z_levels:
                nz = self.amr.n_cell[2]
                assert len(self.erf.terrain_z_levels) == nz+1
