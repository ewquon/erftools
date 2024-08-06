import numpy as np
from scipy.signal import find_peaks

from .wrapper import ABLWrapper
from ..postprocessing import AveragedProfiles

class SCM(ABLWrapper):
    """Basic single-column model driver"""

    def __init__(self,
                 nz,
                 ztop,
                 Tsim=None, # default for run()
                 dt=None, # default for run()
                 abl_geo_wind=[0,0],
                 pbl_type='MYNN2.5',
                 latitude=90.0,
                 rotational_time_period=86400.0,
                 builddir=None,
                 **kwargs):
        n_cell = [2,2,nz] # minimum blocking factor
        self.Tsim = Tsim
        self.dt = dt
        prob_extent = [n_cell[0]*3000., n_cell[1]*3000., ztop]
        super().__init__(n_cell,prob_extent,builddir=builddir,**kwargs)

        self.ds = None # solution profiles

        self.sim_params['pbl_type'] = pbl_type
        self.sim_params['erf.fixed_mri_dt_ratio'] = 4

        # Coriolis
        self.f_c = 4*np.pi / rotational_time_period * np.sin(np.radians(latitude))
        self.sim_params['erf.latitude'] = latitude
        self.sim_params['erf.rotational_time_period'] = rotational_time_period
        self.sim_params['erf.coriolis_3d'] = 0

        self.abl_geo_wind = np.array(abl_geo_wind[:2])
        self.sim_params['erf.abl_driver_type'] = 'GeostrophicWind'
        self.sim_params['erf.abl_geo_wind'] = f'{self.abl_geo_wind[0]:g} {self.abl_geo_wind[1]:g}'

        # Outputs
        self.sim_params['erf.data_log'] = 'surf.dat mean.dat'
        self.sim_params['erf.profile_int'] = 60

    def init_soln(self,
                  z_profile,  # [m]
                  th_profile, # potential temperature [K]
                  qv_profile=None, # water vapor mixing ratio [kg/kg]
                  u_profile=None,
                  v_profile=None,
                 ):
        """Initialize input_sounding profile. Must specify temperature profile.
        The default wind profile is constant velocity equal to the geostrophic
        wind. Air is dry by default.
        """
        if u_profile is None:
            u_profile = self.abl_geo_wind[0] * np.ones_like(z_profile)
        if v_profile is None:
            v_profile = self.abl_geo_wind[1] * np.ones_like(z_profile)
        super(SCM,self).init_soln(z_profile,
                                  u_profile,
                                  v_profile,
                                  th_profile,
                                  qv_profile)

    def setup(self, MOST_z0=0.1, MOST_surf_temp=None, MOST_surf_temp_hist=None,
              **kwargs):
        super(SCM,self).setup(MOST_z0=MOST_z0,
                              MOST_surf_temp=MOST_surf_temp,
                              MOST_surf_temp_hist=MOST_surf_temp_hist,
                              **kwargs)

    def post(self):
        avg = AveragedProfiles(f'{self.rundir}/mean.dat', verbose=False)
        self.ds = avg.ds


class GeostrophicWindEstimator(SCM):
    """This will estimate the geostrophic wind that gives a specified
    wind speed at a reference height.
    """

    def __init__(self,
                 nz,
                 ztop,
                 init_z_profile,
                 init_th_profile,
                 abl_geo_wind=None,
                 target_height=85.0,
                 target_wind_speed=8.0,
                 target_wind_direction=270.0,
                 **kwargs):
        self.target_height = target_height # [m]
        self.target_wind_speed = target_wind_speed # [m/s]
        self.target_wind_direction = target_wind_direction # [deg]
        self.target_angle_rad = np.radians(270.0 - target_wind_direction)

        # save initial conditions -- u,v are set to the geostrophic wind
        self.init_z_profile = init_z_profile
        self.init_th_profile = init_th_profile

        # Geostrophic forcing
        if abl_geo_wind is None:
            self.abl_geo_wind = self.target_wind_speed * \
                    np.array([np.cos(self.target_angle_rad),
                              np.sin(self.target_angle_rad)])
        else:
            self.abl_geo_wind = np.array(abl_geo_wind)

        super().__init__(nz, ztop, abl_geo_wind=self.abl_geo_wind, **kwargs)

    def setup(self, MOST_z0=0.1, MOST_surf_temp=None, MOST_surf_temp_hist=None,
              **kwargs):
        # we update the input params with the correct geo wind
        self.sim_params['erf.abl_geo_wind'] = f'{self.abl_geo_wind[0]} {self.abl_geo_wind[1]}'
        super().setup(MOST_z0=MOST_z0,
                      MOST_surf_temp=MOST_surf_temp,
                      MOST_surf_temp_hist=MOST_surf_temp_hist,
                      **kwargs)

    def estimate_asymptotic_wind_at_height(self, zref, verbose=True,
                                           plot=False):
        uref = self.ds['u'].sel(z=zref)
        umax = uref.isel(t=find_peaks(uref)[0])
        assert len(umax) >= 2, 'Need longer u timeseries'

        vref = self.ds['v'].sel(z=zref)
        vmax = vref.isel(t=find_peaks(vref)[0])
        assert len(vmax) >= 2, 'Need longer v timeseries'

        if len(umax.t) == 2:
            T_d = np.diff(umax.t).item()
        else:
            T_d = np.diff(umax.t).max()
        ω_d = 2*np.pi / T_d
        assert ω_d <= self.f_c # underdamped

        # ω_d = sqrt(1 - ζ**2)f_c
        ζ = np.sqrt(1 - (ω_d/self.f_c)**2)

        # calculate asymptotic conditions
        # - at first peak (t1): u1 = A0 + A
        # - at second peak (t2): u2 = A0 + A*np.exp(-ζω(t2-t1))
        # u2 - u1 = A*(np.exp(-ζω(t2-t1)) - 1)

        Upert = umax.diff('t').item() / (np.exp(-ζ*self.f_c*T_d) - 1)
        U0 = umax.isel(t=0).item() - Upert

        Vpert = vmax.diff('t').item() / (np.exp(-ζ*self.f_c*T_d) - 1)
        V0 = vmax.isel(t=0).item() - Vpert

        if verbose:
            print(f'  estimated U,V = {U0:g}, {V0:g} m/s at {zref:g} m'
                  f' (ω_d={ω_d:g} rad/s, ζ={ζ:g})')

        if plot:
            import matplotlib.pyplot as plt
            t0_u = umax.t[0].values
            t0_v = vmax.t[0].item()

            fig,axs = plt.subplots(nrows=2,sharex=True)
            axs[0].plot(uref.t, uref)
            axs[0].plot(umax.t, umax, 'r+')
            axs[0].plot(uref.t, U0 + Upert*np.exp(-ζ*ω_d*(uref.t-t0_u)), 'k', lw=0.5)
            axs[0].axhline(U0,c='g',ls='--')
            axs[1].plot(vref.t, vref)
            axs[1].plot(vmax.t, vmax, 'r+')
            axs[1].plot(vref.t, V0 + Vpert*np.exp(-ζ*ω_d*(uref.t-t0_v)), 'k', lw=0.5)
            axs[1].axhline(V0,c='g',ls='--')
            for ax in axs:
                ax.grid(alpha=0.2)

        return U0,V0

    def opt(self,Tsim=None,dt=None,maxiter=10,tol=1e-4,**run_kwargs):
        """Repeatedly run simulations, adjusting erf.abl_geo_wind, until
        the target wind speed matches the simulated wind speed magnitude
        (within wind-speed tolerance `tol`) at the target height.
        Lastly, rotate the geostrophic wind vector to obtain the target
        wind direction at the target height.
        """
        nstep = 0
        U0_hist = []
        V0_hist = []
        abl_geo_wind_hist = []
        if Tsim is None:
            Tsim = self.Tsim
            assert Tsim is not None, 'Need to specify Tsim during init or call to opt()'
        if dt is None:
            dt = self.dt
            assert dt is not None, 'Need to specify dt during init or call to opt()'
        check_int = int(Tsim/dt)

        self.cleanup(realclean=False)

        print('[ STEP',nstep,']')
        self.init_soln(self.init_z_profile, self.init_th_profile)
        self.setup()
        result = self.run(Tsim, dt=dt, check_int=check_int, **run_kwargs)
        assert result.returncode == 0, f'Check {self.rundir}/log.err'
        self.ds = AveragedProfiles(f'{self.rundir}/mean.dat')
        U0, V0 = self.estimate_asymptotic_wind_at_height(self.target_height)
        Umag_sim = (U0**2 + V0**2)**0.5
        err = np.abs(Umag_sim - self.target_wind_speed)

        print('  simulated |U| =',Umag_sim,' error =',err)
        U0_hist.append(U0)
        V0_hist.append(V0)
        abl_geo_wind_hist.append(self.abl_geo_wind.copy())

        while (err > tol) and (nstep < maxiter):
            nstep += 1
            if len(abl_geo_wind_hist) == 1:
                mag_corr = self.target_wind_speed / Umag_sim
                self.abl_geo_wind *= mag_corr
                print('  scaled abl_geo_wind to',self.abl_geo_wind)
            else:
                # extrapolate
                Umag1 = np.sqrt(U0_hist[-1]**2 + V0_hist[-1]**2)
                Umag0 = np.sqrt(U0_hist[-2]**2 + V0_hist[-2]**2)
                Ug_vec1 = abl_geo_wind_hist[-1]
                Ug_vec0 = abl_geo_wind_hist[-2]
                self.abl_geo_wind = Ug_vec1 + \
                        (Ug_vec1 - Ug_vec0)/(Umag1 - Umag0) * (self.target_wind_speed - Umag1)
                print('  extrapolated abl_geo_wind to',self.abl_geo_wind)

            print(f'[ STEP {nstep} ] output written to {self.rundir}/log.out')
            self.init_soln(self.init_z_profile, self.init_th_profile)
            self.setup()
            result = self.run(Tsim, dt=dt, check_int=check_int, **run_kwargs)
            assert result.returncode == 0
            self.ds = AveragedProfiles(f'{self.rundir}/mean.dat')
            U0, V0 = self.estimate_asymptotic_wind_at_height(self.target_height)
            Umag_sim = (U0**2 + V0**2)**0.5
            err = np.abs(Umag_sim - self.target_wind_speed)

            print('  simulated |U| =',Umag_sim,' error =',err)
            U0_hist.append(U0)
            V0_hist.append(V0)
            abl_geo_wind_hist.append(self.abl_geo_wind.copy())

        print('Rotating geostrophic wind vector to get target wind direction')
        wdir_sim = 180. + np.degrees(np.arctan2(U0_hist[-1], V0_hist[-1]))
        wdir_corr = self.target_wind_direction - wdir_sim  # this is still following meteorlogical convention, so >0 ==> clockwise
        ang = np.radians(wdir_corr)
        Rmat = np.array([[ np.cos(ang),np.sin(ang)],
                         [-np.sin(ang),np.cos(ang)]])
        self.abl_geo_wind = Rmat.dot(self.abl_geo_wind)
        print(self.abl_geo_wind)

        # get sim ready if we want to run again with final setup
        self.init_soln(self.init_z_profile, self.init_th_profile)
        self.setup()

        return U0_hist, V0_hist, abl_geo_wind_hist
