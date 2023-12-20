from setup_power_spectra_jit import setup_power_BCMP
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from jax import vmap
from jax_cosmo import Cosmology
from functools import partial
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
import time
import jax_cosmo.background as bkgrd
from jax_cosmo import power 
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import z2a
import numpy as onp
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
import jax_cosmo.background as bkgrd

class get_power_BCMP:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                num_points_trapz_int=64,
                setup_power_BCMP_obj=None,
                verbose_time=False,doyclonly=False, kcut=None
            ):    

        if verbose_time:
            t0 = time.time()

        self.cosmo_params = sim_params_dict['cosmo']
        self.zbin=[0.3,0.45]

        self.cosmo_jax = Cosmology(
            Omega_c=self.cosmo_params['Om0'] - self.cosmo_params['Ob0'],
            Omega_b=self.cosmo_params['Ob0'],
            h=self.cosmo_params['H0'] / 100.,
            sigma8=self.cosmo_params['sigma8'],
            n_s=self.cosmo_params['ns'],
            Omega_k=0.,
            w0=self.cosmo_params['w0'],
            wa=0.
            )

        if verbose_time:
            ti = time.time()
        if setup_power_BCMP_obj is None:
            setup_power_BCMP_obj = setup_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time)
        if verbose_time:
            print('Time for setup_power_BCMP: ', time.time() - ti)
            ti = time.time()

        self.calc_nfw_only = analysis_dict['calc_nfw_only']
        self.r_array = setup_power_BCMP_obj.r_array
        self.M_array = setup_power_BCMP_obj.M_array
        self.z_array = setup_power_BCMP_obj.z_array
        self.kPk_array = setup_power_BCMP_obj.kPk_array
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.conc_array = setup_power_BCMP_obj.conc_array
        self.nM, self.nz, self.nc = len(self.M_array), len(self.z_array), len(self.conc_array)

        zmin, zmax, nz = halo_params_dict['zmin'], halo_params_dict['zmax'], halo_params_dict['nz']
        Mmin, Mmax, nM = halo_params_dict['Mmin'], halo_params_dict['Mmax'], halo_params_dict['nM']
        self.chi_array = setup_power_BCMP_obj.chi_array
        self.dchi_dz_array = bkgrd.dchioverda(self.cosmo_jax, self.scale_fac_a_array)*(self.scale_fac_a_array**2)

        self.logc_array = jnp.log(self.conc_array)
        sig_logc = setup_power_BCMP_obj.sig_logc_z_array
        sig_logc_mat = jnp.tile(sig_logc[:, None, None], (1, self.nM, self.nc))
        conc_mat = jnp.tile(self.conc_array[None, None, :], (self.nz, self.nM, 1))
        cmean_mat = jnp.tile(setup_power_BCMP_obj.conc_Mz_mat[:,:,None], (1, 1, self.nc))
        self.p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_mat)/ sig_logc_mat)**2) * (1.0/(sig_logc_mat * jnp.sqrt(2*jnp.pi)))
        if verbose_time:
            print('Time for computing p_logc_Mz: ', time.time() - ti)
            ti = time.time()

        self.hmf_Mz_mat = setup_power_BCMP_obj.hmf_Mz_mat
        self.ycl= setup_power_BCMP_obj.ycl
        self.ycl_integrated = vmap(self.get_ycl_int, (0))(jnp.arange(1))[0]
        self.Mcl = setup_power_BCMP_obj.Mcl
        self.Mcl500_integrated = vmap(self.get_Mcl_int, (0))(jnp.arange(1))[0]
        if doyclonly:
            return




#(const.c.value * 1e-3) / bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)
        self.uyl_mat = jnp.moveaxis(setup_power_BCMP_obj.uyl_mat, 1, 3)
        self.byl_mat = setup_power_BCMP_obj.byl_mat
        self.ukappal_dmb_prefac_mat = jnp.moveaxis(setup_power_BCMP_obj.ukappal_dmb_prefac_mat, 1, 3)
        self.bkl_dmb_mat = setup_power_BCMP_obj.bkl_dmb_mat        
        if self.calc_nfw_only:
            self.ukappal_nfw_prefac_mat = jnp.moveaxis(setup_power_BCMP_obj.ukappal_nfw_prefac_mat, 1, 3)        
            self.bkl_nfw_mat = setup_power_BCMP_obj.bkl_nfw_mat
            
        self.Pklin_lz_mat = setup_power_BCMP_obj.Pklin_lz_mat
        self.Pknonlin_lz_mat = setup_power_BCMP_obj.Pknonlin_lz_mat
        self.ell_array = setup_power_BCMP_obj.ell_array
        self.bias_Mz_mat = setup_power_BCMP_obj.bias_Mz_mat
        self.bh = setup_power_BCMP_obj.bh
        self.nell = len(self.ell_array)

        
        nz_info_dict = analysis_dict['nz_info_dict']
        self.nbins = nz_info_dict['nbins']
        self.z_array_nz = jnp.array(nz_info_dict['z_array'])
        self.zmax = self.z_array_nz[-1]
        pzs_inp_mat = np.zeros((self.nbins, len(self.z_array_nz)))
        for jb in range(self.nbins):
            pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
        self.pzs_inp_mat = jnp.array(pzs_inp_mat)

        self.za = jnp.linspace(0.01, self.zmax, 10000) 
        self.chia = bkgrd.radial_comoving_distance(self.cosmo_jax, 1.0/(1.0+self.za))

        if verbose_time:
            ti = time.time()
        vmap_func1 = vmap(self.weak_lensing_kernel, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Wk_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nz)).T
        if verbose_time:
            print('Time for computing Wk_mat: ', time.time() - ti)
            ti = time.time()

        if analysis_dict['do_sheary']:
            vmap_func1 = vmap(self.get_Cl_kappa_y_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_y_1h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_y_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_y_2h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_y_2h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_y_2h_mat: ', time.time() - ti)
                ti = time.time()


        if analysis_dict['do_shearh']:
            vmap_func1 = vmap(self.get_Cl_kappa_M_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_M_1h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_m_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_M_2h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_M_2h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_m_2h_mat: ', time.time() - ti)
                ti = time.time()
            
            vmap_func1 = vmap(self.get_Cl_kappa_M_nfw_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_M_1h_nfw_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_m_1h_nfw_mat: ', time.time() - ti)
                ti = time.time()


        if analysis_dict['do_shear2pt']:
            vmap_func1 = vmap(self.get_Cl_kappa_kappa_1h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_1h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_kappa_2h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_2h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_2h_mat: ', time.time() - ti)
                # print('Total time for computing all Cls: ', time.time() - t0)
                ti = time.time()                
        if analysis_dict.get('do_yy', False):
            self.Cl_y_y_1h_mat = vmap(self.get_Cl_y_y_1h)(jnp.arange(self.nell))
            if verbose_time:
                print('Time for computing Cl_y_y_1h_mat: ', time.time() - ti)
                ti = time.time()

            self.Cl_y_y_2h_mat = vmap(self.get_Cl_y_y_2h)(jnp.arange(self.nell))
            if verbose_time:
                print('Time for computing Cl_y_y_2h_mat: ', time.time() - ti)
                ti = time.time()

        if analysis_dict['do_shear2pt']:
            vmap_func1 = vmap(self.get_Cl_kappa_kappa_1h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_1h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_kappa_2h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_2h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_2h_mat: ', time.time() - ti)
                # print('Total time for computing all Cls: ', time.time() - t0)
                ti = time.time()                

            if self.calc_nfw_only:
                vmap_func1 = vmap(self.get_Cl_kappa_kappa_nfw_1h, (0, None, None))
                vmap_func2 = vmap(vmap_func1, (None, 0, None))
                vmap_func3 = vmap(vmap_func2, (None, None, 0))
                self.Cl_kappa_kappa_nfw_1h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
                if verbose_time:
                    print('Time for computing Cl_kappa_kappa_nfw_1h_mat: ', time.time() - ti)
                    # print('Total time for computing all Cls: ', time.time() - t0)                
                    ti = time.time()

                vmap_func1 = vmap(self.get_Cl_kappa_kappa_nfw_2h, (0, None, None))
                vmap_func2 = vmap(vmap_func1, (None, 0, None))
                vmap_func3 = vmap(vmap_func2, (None, None, 0))
                self.Cl_kappa_kappa_nfw_2h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
                if verbose_time:
                    print('Time for computing Cl_kappa_kappa_2h_mat: ', time.time() - ti)
                    print('Total time for computing all Cls: ', time.time() - t0)
                    # ti = time.time()                



            vmap_func1 = vmap(self.get_Cl_kappa_kappa_halofit, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None, ))
            vmap_func3 = vmap(vmap_func2, (None, None, 0, ))
            self.Cl_kappa_kappa_halofit_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            self.Cl_kappa_kappa_halofit_mat = self.Cl_kappa_kappa_halofit_mat.at[jnp.where(jnp.isnan(self.Cl_kappa_kappa_halofit_mat))].set(0)
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_halofit_mat: ', time.time() - ti)
                # print('Total time for computing all Cls: ', time.time() - t0)
                ti = time.time()                


    @partial(jit, static_argnums=(0))
    def get_ycl_int(self, dummy):
        fx_intc = jnp.trapz(jnp.log(self.ycl)*self.p_logc_Mz, x=self.logc_array)/jnp.trapz(self.p_logc_Mz, x=self.logc_array)
        yall=[]
        for i in range(self.nM):
            nc = self.hmf_Mz_mat[:,i]
            #nc = nc.at[jnp.where(self.z_array<self.zbin[0])].set(0)
            #nc  = nc.at[jnp.where(self.z_array>self.zbin[1])].set(0)
            fx = fx_intc[:,i]* (self.chi_array ** 2) * self.dchi_dz_array*nc
            
            @vmap
            def integrand_norm(z_prime):
                return jnp.interp(z_prime, self.z_array, (self.chi_array ** 2) * self.dchi_dz_array*nc)
            norm = simps(integrand_norm, self.z_array[0], self.z_array[-1], 128)
            
            def integrand(z_prime):
                return jnp.interp(z_prime, self.z_array, fx)
            fx_intz = jnp.exp(simps(integrand, self.z_array[0], self.z_array[-1], 128)/norm)
            yall.append(fx_intz)
        return jnp.array(yall)


    def get_Mcl_int(self, dummy):
        fx_intc = jnp.trapz(jnp.log(self.Mcl)*self.p_logc_Mz, x=self.logc_array)/jnp.trapz(self.p_logc_Mz, x=self.logc_array)
        yall=[]
        for i in range(self.nM):
            nc = self.hmf_Mz_mat[:,i]
            #nc = nc.at[jnp.where(self.z_array<self.zbin[0])].set(0)
            #nc  = nc.at[jnp.where(self.z_array>self.zbin[1])].set(0)
            fx = fx_intc[:,i]* (self.chi_array ** 2) * self.dchi_dz_array*nc
            
            @vmap
            def integrand_norm(z_prime):
                return jnp.interp(z_prime, self.z_array, (self.chi_array ** 2) * self.dchi_dz_array*nc)
            norm = simps(integrand_norm, self.z_array[0], self.z_array[-1], 128)
            
            def integrand(z_prime):
                return jnp.interp(z_prime, self.z_array, fx)
            fx_intz = jnp.exp(simps(integrand, self.z_array[0], self.z_array[-1], 128)/norm)
            yall.append(fx_intz)
        return jnp.array(yall)







        
    @partial(jit, static_argnums=(0,))
    def weak_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array[jz]
        chi = self.chi_array[jz]

        @vmap
        def integrand(z_prime):
            #chi_prime = jnp.exp(jnp.interp(z_prime, self.z_array, jnp.log(self.chi_array)))
            #chi_prime = radial_comoving_distance(self.cosmo_jax, 1.0/(1.0+z_prime))
            chi_prime = jnp.exp(jnp.interp(z_prime, self.za, jnp.log(self.chia)))
            dndz = (jnp.interp(z_prime, self.z_array_nz, self.pzs_inp_mat[jb, :]))
            return dndz * jnp.clip(chi_prime - chi, 0) / jnp.clip(chi_prime, 1)
        @vmap
        def integrand_norm(z_prime):
            dndz = (jnp.interp(z_prime, self.z_array_nz, self.pzs_inp_mat[jb, :]))
            return dndz

        H0 = 100.0
        c = const.c.value * 1e-3
        constant_factor = 3.0 * H0**2 * self.cosmo_jax.Omega_m / (2.0 * (c**2))
        return constant_factor * radial_kernel

    @partial(jit, static_argnums=(0,1))
    def weak_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array[jz]
        return jnp.squeeze(self.get_weak_lensing_kernel_z(jb, z))



    @partial(jit, static_argnums=(0,))
    def nla_kernel(self, pzs, bias, z, ell):
        """
        Computes the NLA IA kernel
        """
        # stack the dndz of all redshift bins
        dndz = jnp.stack([pz(z) for pz in pzs], axis=0)
        b = bias(self.cosmo_jax, z)
        radial_kernel = dndz * b * bkgrd.H(self.cosmo_jax, z2a(z))
        # Apply common A_IA normalization to the kernel
        # Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
        radial_kernel *= (
            -(5e-14 * const.rhocrit) * self.cosmo_jax.Omega_m / bkgrd.growth_factor(self.cosmo_jax, z2a(z))
        )
        # Constant factor
        constant_factor = 1.0
        # Ell dependent factor
        ell_factor = jnp.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2
        return constant_factor * ell_factor * radial_kernel

    @partial(jit, static_argnums=(0,))
    def get_Cl_y_y_1h(self, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        uyl_jl = self.uyl_mat[jl, ...]        
        fx = uyl_jl * uyl_jl * self.p_logc_Mz
        fx_intc = jnp.trapz(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz
    
    @partial(jit, static_argnums=(0,))
    def get_Cl_y_y_2h(self, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        byl_jl = self.byl_mat[jl]
        
        fx = byl_jl * byl_jl * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_y_1h(self, jb, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb = self.Wk_mat[jb,:]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        uyl_jl = self.uyl_mat[jl, ...]
        ukl_jl = self.ukappal_dmb_prefac_mat[jl, ...]
        
        fx = uyl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jnp.trapz(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk  * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_y_2h(self, jb, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb = self.Wk_mat[jb]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        bkl_jl = self.bkl_dmb_mat[jl]
        byl_jl = self.byl_mat[jl]
        
        fx = byl_jl * bkl_jl * prefac_for_uk  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_1h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)

        ukl_jl = self.ukappal_dmb_prefac_mat[jl]       
        
        fx = ukl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jnp.trapz(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * prefac_for_uk2 * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_2h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        bkl_jl = self.bkl_dmb_mat[jl]
        
        fx = (bkl_jl**2) * prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz

    def nonlinearpowercut(self, rcut=None):
        powerspectra = np.zeros((len(self.z_array), len(self.kPk_array)))
        for i in range(len(self.z_array)):
            pin = power.nonlinear_matter_power(self.cosmo_jax, self.kPk_array, 1/(1+self.z_array[i]))
            from mcfit import P2xi, xi2P
            r, xi = P2xi(self.kPk_array)(pin)
            xicut = xi
            xicut[r>rcut[1]]=0
            xicut[r<rcut[0]]=0
            kcut,pcut = xi2P(r)(xicut)
            powerspectra[i] = pcut
        powerspectra[~np.isfinite(powerspectra)] =0
        return powerspectra
    #@partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_halofit(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        

        #fx = ((self.ell_array[jl]+2)*(self.ell_array[jl]+1)*(self.ell_array[jl])*(self.ell_array[jl]-1)/((self.ell_array[jl] + 0.5) ** 4))*prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pknonlin_lz_mat[jl]
        rcut = self.kcut
        if rcut is not None:
            from jax.scipy.interpolate import RegularGridInterpolator
            powerin = self.nonlinearpowercut(rcut)
            self.powerin = powerin
            interp = RegularGridInterpolator((self.z_array, self.kPk_array), powerin, method='linear')
        @vmap 
        def integrand(z_prime):
            chi = jnp.interp(z_prime, self.z_array, self.chi_array)
            dchi_dz = jnp.interp(z_prime, self.z_array, self.dchi_dz_array)
            k = (self.ell_array[jl] + 0.5) / jnp.clip(chi, 1.0)
            Wk_jb1 = self.get_weak_lensing_kernel_z(jb1, z_prime)
            prefac_for_uk1 = Wk_jb1/jnp.clip(chi**2, 1.0)
            Wk_jb2 = self.get_weak_lensing_kernel_z(jb2, z_prime)
            prefac_for_uk2 = Wk_jb2/jnp.clip(chi**2, 1.0)
            if rcut is not None:
                p = interp((z_prime, k))
            else:
                p = power.nonlinear_matter_power(self.cosmo_jax, k, 1/(1+z_prime))
            fx = ((self.ell_array[jl]+2)*(self.ell_array[jl]+1)*(self.ell_array[jl])*(self.ell_array[jl]-1)/((self.ell_array[jl] + 0.5) ** 4))*prefac_for_uk1 * prefac_for_uk2  * (chi** 2) * dchi_dz * p 
            return fx
        fx_intz = simps(integrand, self.z_array[0], self.z_array[-1], 128)

        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_nfw_1h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)

        ukl_jl = self.ukappal_nfw_prefac_mat[jl]       
        
        fx = ukl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jnp.trapz(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * prefac_for_uk2 * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_nfw_2h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        bkl_jl = self.bkl_nfw_mat[jl]
        
        fx = (bkl_jl**2) * prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jnp.trapz(fx, x=self.z_array)
        return fx_intz

    #@partial(jit, static_argnums=(0,))
    def get_Cl_kappa_M_1h(self, jb1, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        M=8E14
        Wk_jb1 = self.Wk_mat[jb1, :]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)

        ukl_jl = self.ukappal_dmb_prefac_mat[jl, :,:,:]       
        ukl_jlgiveM = ukl_jl[:, -2, :]#jnp.interp([jnp.log(M)], jnp.log(self.M_array), ukl_jl)
        delta_c = 1/self.hmf_Mz_mat[:,-2]
        delta_c = delta_c.at[jnp.where(self.z_array<self.zbin[0])].set(0)
        delta_c  = delta_c.at[jnp.where(self.z_array>self.zbin[1])].set(0)
        #p_logc_Mz_givenM = jnp.interp([jnp.log(M)], jnp.log(self.M_array), self.p_logc_Mz_givenM)
        p_logc_Mz_givenM = self.p_logc_Mz[:,-2,:]
        fx = 1 * ukl_jlgiveM * p_logc_Mz_givenM
        fx_intc = jnp.trapz(fx, x=self.logc_array)*delta_c
        fx = fx_intc * self.hmf_Mz_mat[:,-2]
        fx_intM = fx #= jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * (self.chi_array ** 2) * self.dchi_dz_array

        @vmap
        def integrand_norm(z_prime):
            return jnp.interp(z_prime, self.z_array, (self.chi_array ** 2) * self.dchi_dz_array)
        norm = simps(integrand_norm, self.zbin[0], self.zbin[1], 128)
        def integrand(z_prime):
            return jnp.interp(z_prime, self.z_array, fx)
        fx_intz = simps(integrand, self.z_array[0], self.z_array[-1], 128)/norm

        return fx_intz
    #@partial(jit, static_argnums=(0,))
    def get_Cl_kappa_M_nfw_1h(self, jb1, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        M=8E14
        Wk_jb1 = self.Wk_mat[jb1, :]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)

        ukl_jl = self.ukappal_nfw_prefac_mat[jl, :,:,:]       
        ukl_jlgiveM = ukl_jl[:, -2, :]#jnp.interp([jnp.log(M)], jnp.log(self.M_array), ukl_jl)
        delta_c = 1/self.hmf_Mz_mat[:,-2]
        delta_c = delta_c.at[jnp.where(self.z_array<self.zbin[0])].set(0)
        delta_c  = delta_c.at[jnp.where(self.z_array>self.zbin[1])].set(0)
        #p_logc_Mz_givenM = jnp.interp([jnp.log(M)], jnp.log(self.M_array), self.p_logc_Mz_givenM)
        p_logc_Mz_givenM = self.p_logc_Mz[:,-2,:]
        fx = 1 * ukl_jlgiveM * p_logc_Mz_givenM
        fx_intc = jnp.trapz(fx, x=self.logc_array)*delta_c
        fx = fx_intc * self.hmf_Mz_mat[:,-2]
        fx_intM = fx #= jnp.trapz(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * (self.chi_array ** 2) * self.dchi_dz_array
        #norm = jnp.trapz(((self.chi_array ** 2) * self.dchi_dz_array)[(self.z_array>self.zbin[0])&(self.z_array<self.zbin[1])], x=self.z_array[(self.z_array>self.zbin[0])&(self.z_array<self.zbin[1])])
        #fx_intz = jnp.trapz(fx, x=self.z_array)/norm
        @vmap
        def integrand_norm(z_prime):
            return jnp.interp(z_prime, self.z_array, (self.chi_array ** 2) * self.dchi_dz_array)
        norm = simps(integrand_norm, self.zbin[0], self.zbin[1], 128)

        def integrand(z_prime):
            return jnp.interp(z_prime, self.z_array, fx)
        fx_intz = simps(integrand, self.z_array[0], self.z_array[-1], 128)/norm


        return fx_intz



    #@partial(jit, static_argnums=(0,))
    def get_Cl_kappa_M_2h(self, jb1, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        bh_jl = self.bh[:, -2]
        bh_jl = bh_jl.at[jnp.where(self.z_array<self.zbin[0])].set(0)
        bh_jl = bh_jl.at[jnp.where(self.z_array>self.zbin[1])].set(0)

        
        fx = bh_jl*prefac_for_uk1 * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        #norm = jnp.trapz(((self.chi_array ** 2) * self.dchi_dz_array)[(self.z_array>self.zbin[0])&(self.z_array<self.zbin[1])], x=self.z_array[(self.z_array>self.zbin[0])&(self.z_array<self.zbin[1])])
        #fx_intz = jnp.trapz(fx, x=self.z_array)/norm
        @vmap
        def integrand_norm(z_prime):
            return jnp.interp(z_prime, self.z_array, (self.chi_array ** 2) * self.dchi_dz_array)
        norm = simps(integrand_norm, self.zbin[0], self.zbin[1], 128)

        def integrand(z_prime):
            return jnp.interp(z_prime, self.z_array, fx)
        fx_intz = simps(integrand, self.z_array[0], self.z_array[-1], 128)/norm


        return fx_intz

