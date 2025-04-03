"""
PyTorch implementation of Stochastic Coupled Ekman-Stokes Model (SCESM).

Long Li, March 9, 2025.
ODYSSEY Project-Team, INRIA, Rennes, France.
"""
import torch
import numpy as np

def qsat(t, p):
    """
    Compute saturation vapor pressure es[mb] given t[°C] and p[mb].
    (Buck, 1981: J.Appl.Meteor., 20, 1527-1532).
    """
    es = 6.1121 * torch.exp(17.502*t/(240.97 + t))
    es *= (1.0007 + p*3.46e-6)
    return es

def qsea(sst, pa):
    """
    Compute saturation specific humidity [g/kg] at sea surface
    given sst[°C] and pa[mb].
    """
    eo = 0.98 * qsat(sst, pa) # 98% reduction at sea surface
    qo = 622*eo/(pa - 0.378*eo)
    return qo

def qair(ast, pa, arh):
    """
    Compute specific humidity [g/kg] given ast[°C], pa[mb] and arh[%].
    """
    arh /= 100 # frational rh
    ea = arh * qsat(ast,pa) # partial pressure [mb]
    qa = 621.97*ea/(pa - 0.378*ea)
    return qa

def psiu(zeta, **arr_kwargs):
    """
    Compute velocity structure function given zeta = z/L.
    (Beljaars et Holtslag, 1991, Grachev et al., 2000)
    """
    # Unstable case
    f = zeta**2 / (1 + zeta**2)
    X = (1 - 15*zeta).clamp(min=1e-8)**(1/4) # ensure no negative roots
    Y = (1 - 10.15*zeta).clamp(min=1e-8)**(1/3)
    psi_unstable = (1 - f) * (2*torch.log((1 + X)/2) + torch.log((1 + X**2)/2) - 2*torch.atan(X) + torch.pi/2) \
            + f * (1.5*torch.log((1 + Y + Y**2)/3) - 3**0.5 * torch.atan((1 + 2*Y)/3**0.5) + torch.pi/3**0.5)
    
    # Stable case
    c = torch.minimum(torch.tensor(50, **arr_kwargs), 0.35*zeta)
    psi_stable = -(1 + zeta + 2/3*(zeta - 14.28) * torch.exp(-c) + 8.525)
    
    return torch.where(zeta<0, psi_unstable, psi_stable) 

def psit(zeta, **arr_kwargs):
    """
    Compute temperature structure function given zeta = z/L.
    (Beljaars et Holtslag, 1991, Grachev et al., 2000)
    """
    # Unstable case
    f = zeta**2 / (1 + zeta**2)
    X = (1 - 15*zeta).clamp(min=1e-8)**(1/2)
    Y = (1 - 34.15*zeta).clamp(min=1e-8)**(1/3) 
    psi_unstable = (1 - f) * (2*torch.log((1 + X)/2)) + f * (1.5*torch.log((1 + Y + Y**2)/3) \
            - 3**0.5 * torch.atan((1 + 2*Y) / 3**0.5) + torch.pi / 3**0.5) 
    
    # Stable case
    c = torch.minimum(torch.tensor(50, **arr_kwargs), 0.35*zeta)
    psi_stable = -((1 + 2/3*zeta)**(3/2) + (2/3) * (zeta - 14.28) * torch.exp(-c) + 8.525)
    
    return torch.where(zeta<0, psi_unstable, psi_stable)

def cheby_grid(nx, **arr_kwargs):
    """
    Chebyshev nodes, differentiation matrix and quadrature coefficients on [1,-1].
    """
    # Chebyshev-Lobatto points
    theta = torch.linspace(0, torch.pi, nx, **arr_kwargs).reshape(-1,1)
    x = torch.cos(theta)
    
    # Diff. matrix
    c = torch.ones_like(x)
    c[[0,-1]] *= 2.
    c *= (-1)**torch.arange(nx, **arr_kwargs).reshape(-1,1)
    X = x.tile(1,nx)
    dX = X - X.T
    D = c @ (1./c).T / (dX + torch.eye(nx, **arr_kwargs))
    D -= torch.diag(D.sum(dim=-1)) 
    
    # Quadrature coef. (weight for integration) 
    W = torch.zeros_like(x)
    v = torch.ones((nx-2,1), **arr_kwargs)
    n = nx - 1
    if n%2 == 0:
        W[0] = 1/(n*n-1)
        W[-1] = 1/(n*n-1)
        for k in range(1,n//2):
            v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
        v = v - torch.cos(n*theta[1:-1])/(n*n-1)
    else:
        W[0] = 1/(n*n)
        W[-1] = 1/(n*n)
        for k in range(1,(n-1)//2+1):
            v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
    W[1:-1] = 2*v/n
    return x, D, W


class SCESM:
    """
    Stochastic Coupled Ekman-Stokes Model (SCESM).
    """
    
    def __init__(self, param):
        
        # Data types
        self.farr_kwargs = {'dtype': torch.float64, 'device': param['device']}
        self.carr_kwargs = {'dtype': torch.complex128, 'device': param['device']}

        # Input parameters
        for key in param:
            val = param[key]
            if (type(val) == float):
                val = torch.tensor(val, **self.farr_kwargs)
            if (type(val) == complex):
                val = torch.tensor(val, **self.carr_kwargs)
            setattr(self, key, val) 
        pa = param.get('pa', 1015.) # surface air pressure [mb] (default = 1015mb)
        self.zi = param.get('zi', 600.) # plantenary boundary layer (PBL) height [m] (default = 600m)
        
        # Derived parameters
        self.sqrt_dt = self.dt**0.5
        self.rrho = self.rhoa/self.rhoo    
        self.rrho_sqrt = self.rrho**0.5
        qo = qsea(self.sst, pa)/1000 # surface water specific humidity (kg/kg)
        qa = qair(self.ast, pa, self.arh)/1000 # air specific humidity (kg/kg) 
        self.qdiff = qa - qo # relative specific humidity (kg/kg)
        self.ta = self.ast + 273.16 # air potential temp. (K)
        self.tdiff = self.ast - self.sst + 0.0098*self.delta # relative potential temp. (K)  
        
        # Initializations
        self.set_grid()
        self.ua = torch.zeros(self.ne,self.nza,1, **self.carr_kwargs) # atmos. Ekman (ageostrophic) velocity (m/s)
        self.uo = torch.zeros(self.ne,self.nzo,1, **self.carr_kwargs) # ocean
        self.us = self.Us * torch.exp(2*self.ks*self.zo) * torch.ones(self.ne,1,1, **self.carr_kwargs) # Stokes drift (m/s)
        if self.wave_age:
            omega = (self.g*self.ks)**(1/2) # deep water dispersion (1/s)
            #self.Hs = 4*(self.Us/(omega*self.ks))**(1/2) # significant wave height (m)
            self.Hs = 2*(2*self.Us/(omega*self.ks))**(1/2)
            self.Cp = omega / self.ks # phase speed (m/s) 
        if self.rand_wave:
            self.us *= torch.exp(1j*torch.deg2rad(self.theta_std) * torch.randn(self.ne,1,1, **self.farr_kwargs))

    def set_grid(self):
        """
        Chebyshev nodes, differentiation matrix and quadrature coefficients.
        """
        # Atmos grid
        z, D, W = cheby_grid(self.nza, **self.farr_kwargs) # defined on [1,-1]
        a, b = 0.5*(self.delta - self.Ha), 0.5*(self.Ha + self.delta)
        self.za = a*z + b  # transform onto [delta,Ha]
        self.Da, self.Wa = D/a, abs(a)*W
        self.za.unsqueeze_(0) # for ensemble operations
        self.Da = (self.Da.unsqueeze(0)).type(torch.complex128) # for complex operations
        self.Wa = (self.Wa.T.unsqueeze(0)).type(torch.complex128)
        
        # Ocean grid
        z, D, W = cheby_grid(self.nzo, **self.farr_kwargs) # defined on [1,-1]
        a, b = 0.5*(self.delto - self.Ho), 0.5*(self.Ho + self.delto)
        self.zo = a*z + b  # transform onto [delto,Ho]
        self.Do, self.Wo = D/a, a*W
        self.zo.unsqueeze_(0) # for ensemble operations
        self.Do = (self.Do.unsqueeze(0)).type(torch.complex128) # for complex operations
        self.Wo = (self.Wo.T.unsqueeze(0)).type(torch.complex128)
    
    def neutral_flux(self, check_iter=None):
        """
        Solving the bulk flux formulation under neutral condition 
        (without stratification) by iterative fixed-point method.
        """        
        za = self.za[:,0].squeeze()
        udiff = self.ua[:,0:1] + self.uga - self.uo[:,0:1] - self.ugo        
        U = abs(udiff) # relative speed (m/s)
        
        # Initial guess
        zou = 1e-4 # surface roughness length (m) 
        u10 = U * np.log(10/zou) / torch.log(za/zou) # 10m wind speed (m/s)
        self.ustar = 0.035*u10 
        
        # First update upon bulk formula
        zou = 0.011 * self.ustar**2 / self.g + 0.11 * self.nua_m / self.ustar
        self.Cd = (self.kappa / torch.log(za/zou))**2 
        self.ustar = self.kappa * U / torch.log(za/zou)
        
        # Fixed-point iterations
        for it in range(10):
            ustar = self.ustar.clone() # previous store for convergence
            if self.wave_age:
                zou = 3.35*self.Hs * (self.ustar/self.Cp)**3.4 # wave age effects on sea roughness 
            else:
                zou = 0.011 * self.ustar**2 / self.g # wind-dependent roughness  
            zou += 0.11 * self.nua_m / self.ustar # add friction effect
            self.Cd = (self.kappa / torch.log(za/zou))**2
            self.ustar = self.Cd**(1/2) * U
            # Check convergence
            if (abs(self.ustar - ustar) / ustar).mean() < 1e-6:
                break
        
        if check_iter:
            print(f'Bulk iterations = {it}') 
        self.tau = self.ustar**2 / U * udiff # wind stress (m^2/s^2) 

    def coare_flux(self, check_iter=None):
        """
        Solving the COARE algorithm for bulk flux formulation with the 
        stability condition by iterative fixed-point method.
        """ 
        za = self.za[:,0:1]
        udiff = self.ua[:,0:1] + self.uga - self.uo[:,0:1] - self.ugo
        udabs = abs(udiff) # relative speed (m/s)
        
        # Initial guess
        zou = 1e-4 # momentum roughness length (m) 
        ugust = 0.5 # wind gust factor (m/s) 
        U = (udabs**2 + ugust**2)**(1/2) # add wind gust (m/s)
        u10 = U * np.log(10/zou) / torch.log(za/zou) # 10m wind speed (m/s)
        self.ustar = 0.035*u10 # friction velocity (m/s)
        
        # First update upon bulk formula
        alpha_ch = 0.011 # Charnock parameter
        zou = alpha_ch*self.ustar**2/self.g + 0.11*self.nua_m/self.ustar
        Cd10 = (self.kappa / torch.log(10/zou))**2 # 10m neutral drag coef.
        Ct10 = 0.00115 / Cd10**(1/2) # neutral temp. transfer coef.
        zot = 10/torch.exp(self.kappa/Ct10) # temp. and moisture roughness length [m]
        Cd = (self.kappa / torch.log(za/zou))**2 # momentum transfer coef.
        Ct = self.kappa / torch.log(za/zot) # temp. transfer coef.
        Cc = self.kappa * Ct / Cd 
        Riuc = -za / (self.zi*0.004*self.beta**3) # constant Richardson number
        Riu = self.g*za/self.ta * (self.tdiff + 0.61*self.qdiff*self.ta) / U**2 # bulk Richardson number
        zeta = torch.where(Riu>0, Cc*Riu/(1+Riu/Riuc), Cc*Riu*(1+3*Riu/Cc)) # MOST stability param.
        self.ustar = self.kappa * U / (torch.log(za/zou) - psiu(zeta, **self.farr_kwargs)) # including stability condition
        log_psi = torch.log(za/zot) - psit(zeta, **self.farr_kwargs)
        self.tstar = self.kappa * self.tdiff / log_psi
        self.qstar = self.kappa * self.qdiff / log_psi   
        alpha_ch = 0.011 + 0.007 * torch.clamp((U-10)/8, min=0, max=1) # (Johnson et al. 1998)
        
        # Fixed-point iterations
        for it in range(10): 
            # Update roughness length
            if self.wave_age:
                #zou = 3.35*self.Hs * (self.ustar/self.Cp)**3.4 # Donelan (1990) 
                if self.sea_state:
                    zou = 0.091*self.Hs * (self.ustar/self.Cp)**2
                else:
                    zou = 0.114*(self.ustar/self.Cp)**0.622  * self.ustar**2/self.g                
            else:
                zou = alpha_ch*self.ustar**2/self.g
            zou += 0.11*self.nua_m/self.ustar
            zot = torch.clamp(5.5e-5*(zou*self.ustar/self.nua_m)**(-0.6), max=1.15e-4)
            # Update neutral transfer sub-coef.
            cdn_sqrt = self.kappa / torch.log(za/zou)
            chn_sqrt = self.kappa / torch.log(za/zot)
            # Update stability param. and transfert sub-coef.
            zeta = self.g*self.kappa*za / self.ustar**2 * (self.tstar/self.ta + 0.61*self.qstar)
            cd_sqrt = cdn_sqrt / (1 - cdn_sqrt/self.kappa*psiu(zeta, **self.farr_kwargs))
            ch_sqrt = chn_sqrt / (1 - chn_sqrt/self.kappa*psit(zeta, **self.farr_kwargs))
            # Update frcition scales
            ustar = self.ustar.clone()
            tstar = self.tstar.clone()
            qstar = self.qstar.clone()
            self.ustar = cd_sqrt * U
            self.tstar = ch_sqrt * self.tdiff
            self.qstar = ch_sqrt * self.qdiff
            # Check convergence
            uerr = abs(self.ustar - ustar) / abs(ustar)
            terr = abs(self.tstar - tstar) / abs(tstar)
            qerr = abs(self.qstar - qstar) / abs(qstar)
            if max(uerr.mean(), terr.mean(), qerr.mean()) < 1e-6:
                break
            else:
                # Update wind gust and Charnock param.
                Bf = -self.g*self.ustar * (self.tstar/self.ta + 0.61*self.qstar)
                ugust = torch.where(Bf>0, self.beta*(Bf*self.zi)**(1/3), 0.2)
                U = (udabs**2 + ugust**2)**(1/2)
                alpha_ch = 0.011 + 0.007 * torch.clamp((U-10)/8, min=0, max=1)
        
        if check_iter:
            print(f'Bulk iterations = {it}')
        self.tau = self.ustar**2 * udiff / U # wind stress
        self.Cd = cd_sqrt**2 # momentum transfer coef.        

    def kpp_viscosity(self):
        """
        Compute the eddy viscosity coef. based on the K-Profile Parameterization.
        """
        # Atmos KPP viscosity
        self.ha = self.ca * self.ustar / abs(self.f) # boundary layer depth (m)
        zeta = self.za / self.ha # normalized height
        mask = torch.where(abs(self.za) <= abs(self.ha), 1., 0.)
        self.nua = self.nua_m + self.kappa*self.ustar*abs(self.za)*(1-zeta)**2 * mask

        # Ocean KPP viscosity
        self.ho = self.co * self.rrho_sqrt * self.ustar / abs(self.f) 
        zeta = self.zo / self.ho
        mask = torch.where(abs(self.zo) <= abs(self.ho), 1., 0.)
        self.nuo = self.kappa*self.rrho_sqrt*self.ustar*abs(self.zo)*(1-zeta)**2 * mask
        """
        if self.Us != 0:
            # Effective viscosity due to surface wave (McWilliams and Sullivan, 2000)
            self.La = torch.sqrt(self.rrho_sqrt*self.ustar / self.Us) # Langmuir number 
            self.nuo *= torch.sqrt(1 + 0.08*self.La**(-4))
        """
        self.nuo += self.nuo_m

    def rand_atmos_rhs(self):
        """
        Compute the atmos. RHS terms in random model.
        """
        # Generate random field from viscosity
        dB = self.sqrt_dt * torch.randn(self.ua.shape, **self.farr_kwargs)
        sigma_dB_z = torch.sqrt(2*(self.nua - self.nua_m)) * dB # only vertical component (m)

        # Advection of ua by sigma_dB
        du = -(sigma_dB_z * self.Da) @ self.ua        
        return du

    def rand_ocean_rhs(self):
        """
        Compute the ocean RHS terms in random model.
        """
        # Generate random fields from viscosity
        dB = self.sqrt_dt * torch.randn(self.uo.shape, **self.farr_kwargs)
        sigma = torch.sqrt(2*(self.nuo - self.nuo_m))
        sigma_dB_z = sigma * dB # vertical component (m)
        sigma_dB_h = 4*self.ks*self.us*torch.nan_to_num(1./sigma, posinf=0., neginf=0.)*dB # horizontal

        # Advection noise and Coriolis noise
        du = -(sigma_dB_z * self.Do) @ self.uo - 1j*self.f * sigma_dB_h

        # Additional random wave mixing
        if self.rand_ocean == 'full':
            du += self.Do @ (self.nuo*self.Do) @ self.us*self.dt - (sigma_dB_z*self.Do) @ self.us
        return du
    
    def step(self):
        """
        Time stepping with an implicit scheme.
        """
        # Compute air-sea fluxes
        if self.bulk_coare:
            self.coare_flux()
        else:
            self.neutral_flux()

        # Compute eddy viscosity
        self.kpp_viscosity()
        
        # Atmos LHS operator
        I = torch.eye(self.nza, **self.farr_kwargs)
        nuD = self.nua * self.Da
        L = I - self.dt*(self.Da @ nuD - 1j*self.f*I)
        L[:,0,:] = nuD[:,0,:]  # surface boundary
       
        # Atmos stepping
        rhs = self.ua.clone()
        if self.rand_atmos:
            rhs += self.rand_atmos_rhs() 
        rhs[:,0,:] = self.tau[:,0,:]
        self.ua[:,:-1] = torch.linalg.solve(L[:,:-1,:-1], rhs[:,:-1])

        # Ocean LHS operator
        I = torch.eye(self.nzo, **self.farr_kwargs)
        nuD = self.nuo * self.Do
        L = I - self.dt*(self.Do @ nuD - 1j*self.f*I)
        L[:,0,:] = nuD[:,0,:]  # surface boundary
        
        # Ocean stepping
        rhs = self.uo - 1j*self.f*self.dt*I @ self.us # add Coriolis Stokes force 
        if self.rand_ocean:
            rhs += self.rand_ocean_rhs()
        rhs[:,0,:] = self.rrho * self.tau[:,0,:]
        self.uo[:,:-1] = torch.linalg.solve(L[:,:-1,:-1], rhs[:,:-1])
       

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    outdir = '/srv/storage/ithaca@storage2.rennes.grid5000.fr/lli/ekman/coare_new/rcm_rs_wa_ss'
    param = {
            'rand_atmos': True, # random atmos model
            'rand_ocean': True, # random ocean model (True, False, 'full')
            'rand_wave': True, # random wave model
            'ne': 500, # ensemble size
            'nza': 1000, # atmos grid size
            'nzo': 300, # ocean grid size
            'dt': 300., # time step (s)
            'delta': 10., # atmos surface level (m)
            'Ha': 1000., # lower bound of free-atmos (m)
            'delto': -1., # ocean surface level (m)
            'Ho': -100., # lower bound of free-ocean (m)
            'f': 8.36e-5, # Coriolis parameter (s^-1)
            'g': 9.81, # gravity (m/s^2)
            'rhoo': 1.0e3, # water density (kg/m^3)
            'rhoa': 1.0, # air density (kg/m^3)
            'kappa': 0.4, # von Karman constant
            'ca': 0.2, # atmos boundary depth constant
            'co': -0.7, # ocean boundary depth constant
            'nua_m': 1.5e-5, # atmos molecular viscosity (m^2/s)
            'nuo_m': 1e-6, # ocean molecular viscosity (m^2/s)
            'uga': 9.+0.j, # geostrophic wind speed (m/s)
            'ugo': 0.+0.j, # geostrophic current speed (m/s)
            'Us': 0.068, # Stokes magnitude (m/s), set to zero if no Stokes
            'ks': 0.105, # Wavenumber of Stokes drift (1/m)
            'theta_std': 5., # standard deviation of Stokes angular direction (deg)
            'wave_age': True, # roughness length parameterization due to wave age
            'sea_state': True, # sea-state dependent param.            
            'bulk_coare': True, # COARE bulk algorithm for air-sea turbulent fluxes 
            'ast': 26.5, # air surface temperature (°C)
            'sst': 28., # sea surface temperature (°C)
            'arh': 0., # air relative humidity (%)
            'beta': 1.2, # wind gust constant
            'device': 'cuda', #if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
    }

    run = SCESM(param)

    # Control param.
    t = 0.
    dt = param['dt']
    n_steps = int(20*24*3600/dt) + 1
    freq_checknan = int(3600/dt)
    freq_log = int(3600/dt)
    freq_plot = 0*int(6*3600/dt)
    freq_save = int(3600/dt) # per hour
    n_steps_save = 1

    data = np.load('LOTUS3_data.npz')
    z_ref = data['z'].astype('float64')[:-2]
    um_ref = data['um'].astype('float64')[:-2]
    vm_ref = data['vm'].astype('float64')[:-2]
    nobs = len(um_ref) + len(vm_ref)
    data.close()

    # Init. logout
    if freq_log > 0:
        log_str = '*****************************************\n' \
                  '  Stochastic Ekman-Stokes Coupled Model  \n' \
                  '*****************************************\n\n'
        log_str += 'Input parmeters\n' \
                   '---------------\n' \
                   f'{param}\n\n' \
                   'Output log\n' \
                   '----------\n' \
                   'Time, Surf. wind speed (mean,std), Atmos. transp. amp. (mean,std), Atmos. BL depth (mean,std), '\
                   'Surf. current speed (mean,std), Ocean transp. amp. (mean,std), Ocean BL depth (mean,std)'
        print(log_str)

    # Init. outputs
    if freq_save > 0:
        import os
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, 'param.pth')
        torch.save(param, filename)
        filename = os.path.join(outdir, f'operators.npz')
        np.savez(filename, 
                za=run.za.squeeze().cpu().numpy().astype('float32'),
                zo=run.zo.squeeze().cpu().numpy().astype('float32'),
                Da=run.Da.real.squeeze().cpu().numpy().astype('float32'),
                Do=run.Do.real.squeeze().cpu().numpy().astype('float32'),
                Wa=run.Wa.real.squeeze().cpu().numpy().astype('float32'),
                Wo=run.Wo.real.squeeze().cpu().numpy().astype('float32')
                )

    # Init. figures
    if freq_plot > 0:
        za = run.za.squeeze().cpu().numpy()
        zo = (run.zo.squeeze().cpu().numpy())[::-1]
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1,4,figsize=(12,4),constrained_layout=True)
        plt.pause(0.1)

    # Time-steppings    
    for n in range(1, n_steps+1):
        run.step()
        t += dt

        if n % freq_checknan == 0: 
            if torch.isnan(abs(run.ua)).any(): 
                raise ValueError(f'Stopping, NAN number of `ua` at iteration {n}.')
            if torch.isnan(abs(run.uo)).any(): 
                raise ValueError(f'Stopping, NAN number of `uo` at iteration {n}.')
        
        if freq_plot > 0 and n % freq_plot == 0: 
            [ax[i].clear() for i in range(4)]
            
            Ua = run.ua + run.uga
            ua = Ua.real.squeeze(-1).cpu().numpy()
            va = Ua.imag.squeeze(-1).cpu().numpy() 
            ua_mean, ua_std = np.mean(ua,axis=0), np.std(ua,axis=0) 
            va_mean, va_std = np.mean(va,axis=0), np.std(va,axis=0) 
            
            Uo = run.uo + run.ugo
            uo = Uo.real.squeeze(-1).cpu().numpy()
            vo = Uo.imag.squeeze(-1).cpu().numpy()
            uo_mean, uo_std = np.mean(uo,axis=0)[::-1], np.std(uo,axis=0)[::-1] 
            vo_mean, vo_std = np.mean(vo,axis=0)[::-1], np.std(vo,axis=0)[::-1] 

            ax[0].plot(ua_mean, za)
            ax[0].fill_betweenx(za, ua_mean-ua_std, ua_mean+ua_std, alpha=0.15)
            ax[0].set(ylabel=r'$z$ (m)', xlabel='$u_a$ (m/s)')
            
            ax[1].plot(va_mean, za)
            ax[1].fill_betweenx(za, va_mean-va_std, va_mean+va_std, alpha=0.15)
            ax[1].set(ylabel=r'$z$ (m)', xlabel='$v_a$ (m/s)')
            
            ax[2].plot(um_ref[::-1], z_ref[::-1], '+', c='k')
            ax[2].plot(uo_mean, zo)
            ax[2].fill_betweenx(zo, uo_mean-uo_std, uo_mean+uo_std, alpha=0.15)
            ax[2].set(ylabel=r'$z$ (m)', xlabel='$u_o$ (m/s)')
            
            ax[3].plot(vm_ref[::-1], z_ref[::-1], '+', c='k')
            ax[3].plot(vo_mean, zo)
            ax[3].fill_betweenx(zo, vo_mean-vo_std, vo_mean+vo_std, alpha=0.15)
            ax[3].set(ylabel=r'$z$ (m)', xlabel='$v_o$ (m/s)')
            
            [ax[i].grid() for i in range(4)]
            plt.suptitle(f'Time: {int(t//86400):03d} days {int(t%86400//3600):02d} hours')
            plt.yticks(rotation=90)
            plt.pause(0.5)

        if freq_log > 0 and n % freq_log == 0:
            ua = abs(run.ua[:,0]).squeeze().cpu().numpy() # surface wind amp.
            Ta = abs(run.Wa @ run.ua).squeeze().cpu().numpy() # Ekman transport amp. (m^2/s)
            ha = run.ha.squeeze().cpu().numpy()
            
            uo = abs(run.uo[:,0]).squeeze().cpu().numpy() # surface current amp.
            To = abs(run.Wo @ run.uo).squeeze().cpu().numpy() 
            ho = run.ho.squeeze().cpu().numpy()

            #print(f'Langmuir number = {(run.La).squeeze().cpu().numpy().mean()}')
            #print(f'wind stress = {(run.rhoa*abs(run.tau)).squeeze().cpu().numpy().mean()} (N/m^2)')
            
            log_str = f't={int(t//86400):03d}d{int(t%86400//3600):02d}h, ' \
                      f'ua0=({np.mean(ua):.3f},{np.std(ua):.3f})m/s, ' \
                      f'Ta=({np.mean(Ta):.3f},{np.std(Ta):.3f})m^2/s, ' \
                      f'ha=({np.mean(ha):.2f},{np.std(ha):.2f})m, ' \
                      f'uo0=({np.mean(uo):.3f},{np.std(uo):.3f})m/s, ' \
                      f'To=({np.mean(To):.3f},{np.std(To):.3f})m^2/s, ' \
                      f'ho=({np.mean(ho):.2f},{np.std(ho):.2f})m'
            print(log_str)
            
        if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
            filename = os.path.join(outdir, f't_{n_steps_save}.npz')
            np.savez(filename, t=t/86400, 
                    ua=run.ua.real.squeeze().cpu().numpy().astype('float32'),
                    va=run.ua.imag.squeeze().cpu().numpy().astype('float32'),
                    uo=run.uo.real.squeeze().cpu().numpy().astype('float32'),
                    vo=run.uo.imag.squeeze().cpu().numpy().astype('float32'),
                    ha=run.ha.squeeze().cpu().numpy().astype('float32'),
                    ho=run.ho.squeeze().cpu().numpy().astype('float32'), 
                    Cd=run.Cd.squeeze().cpu().numpy().astype('float32'),
                    ustar=run.ustar.squeeze().cpu().numpy().astype('float32'),
                    taux=run.tau.real.squeeze().cpu().numpy().astype('float32'),
                    tauy=run.tau.imag.squeeze().cpu().numpy().astype('float32')
                    )
            n_steps_save += 1
            if n % (10*freq_save) == 0:
                print(f'Data saved to {filename}')

    if freq_log > 0:
        log_str = '\n*********************************************\n' \
                    '                      END                    \n' \
                    '*********************************************'
        print(log_str)
