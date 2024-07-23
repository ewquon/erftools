import numpy as np
from scipy.signal import find_peaks

from .wrapper import ABLWrapper
from ..postprocessing import AveragedProfiles

class GeostrophicWindEstimator(ABLWrapper):
    """This will estimate the geostrophic wind that gives a specified
    wind speed at a reference height
    """

    def __init__(self,
                 nz,
                 zmax,
                 init_z_profile,
                 init_th_profile,
                 Tsim=None,
                 dt=None,
                 pbl_type='MYNN2.5',
                 target_height=85.0,
                 target_wind_speed=8.0,
                 target_wind_direction=270.0,
                 latitude=90.0,
                 rotation_time_period=86400.0,
                 builddir=None,
                 **kwargs):
        n_cell = [4,4,nz]
        dz = np.round(zmax/nz)
        self.Tsim = Tsim
        self.dt = dt
        prob_extent = dz * np.array(n_cell)
        super().__init__(n_cell,prob_extent,builddir=builddir,**kwargs)

        self.init_z_profile = init_z_profile # [m]
        self.init_th_profile = init_th_profile # potential temperature [K]
        self.target_height = target_height # [m]
        self.target_wind_speed = target_wind_speed # [m/s]
        self.target_wind_direction = target_wind_direction # [deg]
        self.target_angle_rad = np.radians(270.0 - target_wind_direction)

        self.ds = None # solution profiles

        self.sim_params['pbl_type'] = pbl_type
        self.sim_params['erf.fixed_mri_dt_ratio'] = 4

        # Coriolis
        self.f_c = 4*np.pi / rotation_time_period * np.sin(np.radians(latitude))
        self.sim_params['erf.latitude'] = latitude
        self.sim_params['erf.rotation_time_period'] = rotation_time_period

        # Geostrophic forcing
        self.abl_geo_wind = self.target_wind_speed * \
                np.array([np.cos(self.target_angle_rad),
                          np.sin(self.target_angle_rad)])
        self.sim_params['erf.abl_driver_type'] = 'GeostrophicWind'
        self.sim_params['erf.abl_geo_wind'] = f'{self.abl_geo_wind[0]:g} {self.abl_geo_wind[1]:g}'

        # Outputs
        self.sim_params['erf.data_log'] = 'surf.dat mean.dat'
        self.sim_params['erf.profile_int'] = 60

    def estimate_asymptotic_wind_at_height(self, zref, verbose=True,
                                           plot=False):
        uref = self.df['u'].sel(z=zref)
        umax = uref.isel(t=find_peaks(uref)[0])
        assert len(umax) >= 2, 'Need longer u timeseries'

        vref = self.df['v'].sel(z=zref)
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

    def opt(self,Tsim=None,dt=None,maxiter=10,tol=1e-3,**run_kwargs):
        nstep = 0
        if Tsim is None:
            Tsim = self.Tsim
            assert Tsim is not None, 'Need to specify Tsim during init or call to opt()'
        if dt is None:
            dt = self.dt
            assert dt is not None, 'Need to specify dt during init or call to opt()'
        check_int = int(Tsim/dt)

        self.cleanup(realclean=False)

        print('[ STEP',nstep,']')
        self.init(self.init_z_profile,
                  self.abl_geo_wind[0] * np.ones_like(self.init_z_profile),
                  self.abl_geo_wind[1] * np.ones_like(self.init_z_profile),
                  self.init_th_profile)
        self.setup(**self.sim_params)
        result = self.run(Tsim, dt=dt, check_int=check_int, **run_kwargs)
        assert result.returncode == 0
        self.df = AveragedProfiles(f'{self.rundir}/mean.dat')
        U0, V0 = self.estimate_asymptotic_wind_at_height(self.target_height)
        Umag_sim = (U0**2 + V0**2)**0.5
        err = np.abs(Umag_sim - self.target_wind_speed)
        print('  simulated |U| =',Umag_sim,' error =',err)

        while (err > tol) and (nstep < maxiter):
            nstep += 1
            mag_corr = self.target_wind_speed / Umag_sim
            print('  scaling abl_geo_wind by',mag_corr)
            self.abl_geo_wind *= mag_corr

            print(f'[ STEP {nstep}] output written to {self.rundir}/log.out')
            self.init(self.init_z_profile,
                      self.abl_geo_wind[0] * np.ones_like(self.init_z_profile),
                      self.abl_geo_wind[1] * np.ones_like(self.init_z_profile),
                      self.init_th_profile)
            self.sim_params['erf.abl_geo_wind'] = f'{self.abl_geo_wind[0]:g} {self.abl_geo_wind[1]:g}'
            self.setup(**self.sim_params)
            result = self.run(Tsim, dt=dt, check_int=check_int, **run_kwargs)
            assert result.returncode == 0
            self.df = AveragedProfiles(f'{self.rundir}/mean.dat')
            U0, V0 = self.estimate_asymptotic_wind_at_height(self.target_height)
            Umag_sim = (U0**2 + V0**2)**0.5
            err = np.abs(Umag_sim - self.target_wind_speed)
            print('  simulated |U| =',Umag_sim,' error =',err)
