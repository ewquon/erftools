import os
import shutil
import glob
import subprocess
import numpy as np

from ..inputs import ERFInputFile
from ..input_sounding import InputSounding

class Wrapper(object):
    """Base class for developing wrappers around ERF solvers

    This includes all essential functionality to setup and run ERF, but
    the derived classes provide an opportunity to do some specialized
    setup and input validation (see "template" below).
    """
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
            erf_abl_path = shutil.which('erf_abl')
            assert erf_abl_path is not None, 'Need to specify ERF builddir'
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
        mpirun = shutil.which('mpirun')
        if mpirun == '':
            mpirun = shutil.which('mpiexec')
        if mpirun == '':
            mpirun = shutil.which('srun')
        if mpirun == '':
            print('Note: MPI environment not found')
        self.mpirun = mpirun

    def init(self,z_profile,u_profile,v_profile,th_profile,qv_profile=None,
             p_surf=1e5,th_surf=None,qv_surf=None):
        assert len(z_profile) == len(u_profile) == len(v_profile) == len(th_profile)
        if qv_profile is None:
            qv_profile = np.zeros_like(z_profile)
        else:
            assert len(qv_profile) == len(z_profile)
        if th_surf is None:
            th_surf = th_profile[0]
        if qv_surf is None:
            qv_surf = qv_profile[0]
        self.input_sounding = InputSounding(
            p_surf=p_surf,
            th_surf=th_surf,
            qv_surf=qv_surf,
            z_profile=z_profile,
            th_profile=th_profile,
            qv_profile=qv_profile,
            u_profile=u_profile,
            v_profile=v_profile)
        self.initialized = True

    def setup(self,
              n_cell=[0,0,0],
              prob_extent=[0.0,0.0,0.0],
              periodic=[True,True,False],
              max_level=0, # level index, not the number of levels
              zlo_type="MOST", MOST_z0=0.1, MOST_surf_temp=None,
              les_type="None",
              pbl_type="None",
              molec_diff_type="None",
              **kwargs):
        """
        This is a template
        """
        assert np.all([n>0 for n in n_cell]), 'Need to specify number of cells'
        assert np.all([L>0 for L in prob_extent]), 'Need to specify problem extent'
        if MOST_surf_temp is None:
            MOST_surf_temp = self.input_sounding.th_surf
        sim_params = {
            'amr.n_cell': n_cell,
            'geometry.prob_extent': prob_extent,
            'geometry.is_periodic': periodic,
            'amr.max_level': max_level,
            'zlo.type': zlo_type,
            'erf.most.z0': MOST_z0,
            'erf.most.surf_temp': MOST_surf_temp,
            'erf.les_type': les_type,
            'erf.pbl_type': pbl_type,
            'erf.molec_diff_type': molec_diff_type,
        }
        for key,val in kwargs.items():
            sim_params[key] = val
        self.inputs = ERFInputFile(sim_params, verbose=False)

        self.inputs['zhi.theta_grad'] = \
                (self.input_sounding.th[-1] - self.input_sounding.th[-2]) \
              / (self.input_sounding.z[ -1] - self.input_sounding.z[ -2])

        self.setup_complete = True

    def check_inputs(self):
        """
        This is a template
        """
        # do some input validation...
        if 'erf.restart' in self.inputs.keys():
            chkpt = os.path.join(self.rundir, self.inputs['erf.restart'])
            assert os.path.isdir(chkpt), f'Checkpoint {chkpt} not found'

    def create_input_files(self):
        input_sounding_file = os.path.join(self.rundir,'input_sounding')
        self.input_sounding.write(input_sounding_file, overwrite=True)
        input_file = os.path.join(self.rundir,'inputs')
        self.inputs.write(input_file,ideal=True)

    def run(self, stop_time, dt, restart=None,
            plot_int=-1, check_int=-1,
            postproc=True,
            rundir='.rundir', ncpu=1):
        assert self.initialized, 'Need to call init()'
        assert self.setup_complete, 'Need to call setup()'
        self.rundir = rundir
        self.inputs['stop_time'] = stop_time
        self.inputs['erf.fixed_dt'] = dt
        self.inputs['erf.plot_int'] = plot_int
        self.inputs['erf.check_int'] = check_int
        if restart is None:
            self.inputs.pop('erf.restart',None)
        else:
            self.inputs['erf.restart'] = restart
        # setup run directory
        os.makedirs(rundir, exist_ok=True)
        self.check_inputs()
        self.create_input_files()
        # call solver
        ncpu = min(ncpu, self.ncpus_avail)
        if self.mpirun != '':
            cmd = [self.mpirun,'-n',str(ncpu)]
        else:
            cmd = []
        cmd += [self.solver, 'inputs']
        with open(os.path.join(rundir,'log.out'),'w') as outfile, \
             open(os.path.join(rundir,'log.err'),'w') as errfile:
            result = subprocess.run(cmd, cwd=rundir,
                                    stdout=outfile,
                                    stderr=errfile)
        if result.returncode != 0:
            print('Return code:',result.returncode)
        elif postproc:
            self.post()

        return result

    def post(self):
        """
        This is a template
        """
        self.ds = None

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
        elif rundir != '':
            print(f'{rundir} not found?')


class ABLWrapper(Wrapper):
    solver = 'Exec/ABL/erf_abl'

    def __init__(self,n_cell,prob_extent,builddir=None,**kwargs):
        super().__init__(builddir=builddir,**kwargs)
        self.solver = os.path.join(self.builddir,self.solver)
        self.sim_params = {
            'n_cell': n_cell,
            'prob_extent': prob_extent,
        }
        for key,val in kwargs.items():
            self.sim_params[key] = val
