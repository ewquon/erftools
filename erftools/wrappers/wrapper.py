import os
import shutil
import glob
import subprocess
import numpy as np

class Wrapper(object):
    solver = 'EXEC_REL_PATH'

    def __init__(self,builddir=None):
        self.initialized = False
        self.setup_complete = False
        self.rundir = ''
        self._find_build_dir(builddir)
        self._get_ncpus()
        self._get_mpi_environ()

    def _find_build_dir(self,builddir):
        if builddir is None:
            erf_abl_path = shutils.which('erf_abl')
            assert erf_abl_path != '', 'Need to specify ERF builddir'
            # erf_abl_path = /path/to/builddir/Exec/ABL/erf_abl
            self.builddir = os.sep.join(erf_abl_path.split(os.sep)[:-3])
        else:
            assert os.path.isdir(builddir)
            self.builddir = builddir

    def _get_ncpus(self):
        try:
            import multiprocessing as mp
        except ImportError:
            self.ncpus_avail = 1
        else:
            self.ncpus_avail = mp.cpu_count()

    def _get_mpi_environ(self):
        mpirun = shutils.which('mpirun')
        if mpirun == '':
            mpirun = shutils.which('mpiexec')
        if mpirun == '':
            mpirun = shutils.which('srun')
        if mpirun == '':
            print('Note: MPI environment not found')
        self.mpirun = mpirun

    def init(self,z_profile,u_profile,v_profile,th_profile,qv_profile=None,
             p_surf=1e5,th_surf=None,qv_surf=None):
        self.initialized = True

    def setup(self):
        self.setup_complete = True

    def check_inputs(self):
        """This is a template"""

    def run(self,rundir='.rundir',ncpu=1):
        """This is a template"""
        if not self.iniitalized:
            print('Need to call init()')
        if not self.setup_complete:
            print('Need to call setup()')
        self.check_inputs()
        # setup run directory
        self.rundir = rundir
        os.makedir(rundir, exist_ok=True)
        self._create_inputs(rundir)
        # call solver
        ncpu = min(ncpu, self.ncpus_avail)
        if self.mpirun != '':
            cmd = [self.mpirun,'-n',ncpu]
        else:
            cmd = []
        cmd += [self.solver,'inputs']
        with open(os.path.join(rundir,'log.out'),'w') as outfile, \
             open(os.path.join(rundir,'log.err'),'w') as errfile:
            result = subprocess.run(cmd,stdout=outfile,stderr=errfile)

    def cleanup(self,rundir=None,realclean=True):
        if rundir is None:
            rundir = self.rundir
        if os.path.isdir(rundir):
            if realclean:
                shutil.rmtree(rundir)
            else:
                for dpath in glob.glob(os.path.join(rundir,'plt*')):
                    if os.path.isdir(dpath):
                        shutil.rmtree(dpath)
                for dpath in glob.glob(os.path.join(rundir,'chk*')):
                    if os.path.isdir(dpath):
                        shutil.rmtree(dpath)
        else:
            print(f'{rundir} not found?')


class ABLWrapper(Wrapper):
    solver = 'Exec/ABL/erf_abl'

    def __init__(self,builddir=None):
        super().__init__(builddir=builddir)
