import os
import sys
import glob
import torch
import numpy as np
from netCDF4 import Dataset
from scipy.stats import skew,kurtosis,circmean,circstd
from scipy.stats import wasserstein_distance as wdist
from properscoring import crps_ensemble

def init_netcdf(nc, ftype='f4'):
    # Axes
    nc.createVariable('t',ftype,('t',))
    nc.createVariable('za',ftype,('za',))
    nc.createVariable('zo',ftype,('zo',))
    # Ensemble scores
    nc.createVariable('dist_mse_var',ftype,('t',))
    nc.createVariable('mean_crps',ftype,('t',))
    nc.createVariable('mean_wdist',ftype,('t',))
    # Stats of (ua,va)
    nc.createVariable('ua_mean',ftype,('t','za',))
    nc.createVariable('ua_std',ftype,('t','za',))
    nc.createVariable('ua_skew',ftype,('t','za',))
    nc.createVariable('ua_kurt',ftype,('t','za',))
    nc.createVariable('va_mean',ftype,('t','za',))
    nc.createVariable('va_std',ftype,('t','za',))
    nc.createVariable('va_skew',ftype,('t','za',))
    nc.createVariable('va_kurt',ftype,('t','za',))
    nc.createVariable('uva_corr',ftype,('t','za',))
    nc.createVariable('uva_arg_mean',ftype,('t','za',))
    nc.createVariable('uva_arg_std',ftype,('t','za',))
    # Stats of (uo,vo)
    nc.createVariable('uo_mean',ftype,('t','zo',))
    nc.createVariable('uo_std',ftype,('t','zo',))
    nc.createVariable('uo_skew',ftype,('t','zo',))
    nc.createVariable('uo_kurt',ftype,('t','zo',))
    nc.createVariable('vo_mean',ftype,('t','zo',))
    nc.createVariable('vo_std',ftype,('t','zo',))
    nc.createVariable('vo_skew',ftype,('t','zo',))
    nc.createVariable('vo_kurt',ftype,('t','zo',))
    nc.createVariable('uvo_corr',ftype,('t','zo',))
    nc.createVariable('uvo_arg_mean',ftype,('t','zo',))
    nc.createVariable('uvo_arg_std',ftype,('t','zo',))
    # Stats of atmos energy and work
    nc.createVariable('mkea',ftype,('t','za',))
    nc.createVariable('ekea',ftype,('t','za',))
    nc.createVariable('mkea_int',ftype,('t',))
    nc.createVariable('ekea_int',ftype,('t',))
    nc.createVariable('mwwa',ftype,('t',))
    nc.createVariable('ewwa',ftype,('t',))
    # Stats of ocean energy and work
    nc.createVariable('mkeo',ftype,('t','zo',))
    nc.createVariable('ekeo',ftype,('t','zo',))
    nc.createVariable('mkeo_int',ftype,('t',))
    nc.createVariable('ekeo_int',ftype,('t',))
    nc.createVariable('mwwo',ftype,('t',))
    nc.createVariable('ewwo',ftype,('t',))
    # Stats of (taux,tauy)
    nc.createVariable('taux_mean',ftype,('t',))
    nc.createVariable('taux_std',ftype,('t',))
    nc.createVariable('taux_skew',ftype,('t',))
    nc.createVariable('taux_kurt',ftype,('t',))
    nc.createVariable('tauy_mean',ftype,('t',))
    nc.createVariable('tauy_std',ftype,('t',))
    nc.createVariable('tauy_skew',ftype,('t',))
    nc.createVariable('tauy_kurt',ftype,('t',))
    nc.createVariable('tauxy_corr',ftype,('t',))
    nc.createVariable('tauxy_arg_mean',ftype,('t',))
    nc.createVariable('tauxy_arg_std',ftype,('t',))
    # Stats of ha
    nc.createVariable('ha_mean',ftype,('t',))
    nc.createVariable('ha_std',ftype,('t',))
    nc.createVariable('ha_skew',ftype,('t',))
    nc.createVariable('ha_kurt',ftype,('t',))
    # Stats of ho
    nc.createVariable('ho_mean',ftype,('t',))
    nc.createVariable('ho_std',ftype,('t',))
    nc.createVariable('ho_skew',ftype,('t',))
    nc.createVariable('ho_kurt',ftype,('t',))
    # Stats of Cd
    nc.createVariable('Cd_mean',ftype,('t',))
    nc.createVariable('Cd_std',ftype,('t',))
    nc.createVariable('Cd_skew',ftype,('t',))
    nc.createVariable('Cd_kurt',ftype,('t',))
    # Stats of ustar
    nc.createVariable('ustar_mean',ftype,('t',))
    nc.createVariable('ustar_std',ftype,('t',))
    nc.createVariable('ustar_skew',ftype,('t',))
    nc.createVariable('ustar_kurt',ftype,('t',))
    # Stats of (Txa,Tya)
    nc.createVariable('Txa_mean',ftype,('t',))
    nc.createVariable('Txa_std',ftype,('t',))
    nc.createVariable('Txa_skew',ftype,('t',))
    nc.createVariable('Txa_kurt',ftype,('t',))
    nc.createVariable('Tya_mean',ftype,('t',))
    nc.createVariable('Tya_std',ftype,('t',))
    nc.createVariable('Tya_skew',ftype,('t',))
    nc.createVariable('Tya_kurt',ftype,('t',))
    nc.createVariable('Txya_corr',ftype,('t',))
    nc.createVariable('Txya_arg_mean',ftype,('t',))
    nc.createVariable('Txya_arg_std',ftype,('t',))
    # Stats of (Txo,Tyo)
    nc.createVariable('Txo_mean',ftype,('t',))
    nc.createVariable('Txo_std',ftype,('t',))
    nc.createVariable('Txo_skew',ftype,('t',))
    nc.createVariable('Txo_kurt',ftype,('t',))
    nc.createVariable('Tyo_mean',ftype,('t',))
    nc.createVariable('Tyo_std',ftype,('t',))
    nc.createVariable('Tyo_skew',ftype,('t',))
    nc.createVariable('Tyo_kurt',ftype,('t',))
    nc.createVariable('Txyo_corr',ftype,('t',))
    nc.createVariable('Txyo_arg_mean',ftype,('t',))
    nc.createVariable('Txyo_arg_std',ftype,('t',))
    # Add attributes
    nc.variables['t'].long_name = 'Time axis'
    nc.variables['t'].units = 'days'
    nc.variables['za'].long_name = 'Atmos vertical axis'
    nc.variables['za'].units = 'm'
    nc.variables['zo'].long_name = 'Ocean vertical axis'
    nc.variables['zo'].units = 'm'
    nc.variables['ua_mean'].long_name = 'Mean of atmos zonal velocity'
    nc.variables['ua_mean'].units = 'm/s'
    nc.variables['ua_std'].long_name = 'Standard deviation of atmos zonal velocity'
    nc.variables['ua_std'].units = 'm/s'
    nc.variables['ua_skew'].long_name = 'Skewness of atmos zonal velocity'
    nc.variables['ua_kurt'].long_name = 'Kurtosis of atmos zonal velocity'
    nc.variables['va_mean'].long_name = 'Mean of atmos meridional velocity'
    nc.variables['va_mean'].units = 'm/s'
    nc.variables['va_std'].long_name = 'Standard deviation of atmos meridional velocity'
    nc.variables['va_std'].units = 'm/s'
    nc.variables['va_skew'].long_name = 'Skewness of atmos meridional velocity'
    nc.variables['va_kurt'].long_name = 'Kurtosis of atmos meridional velocity'
    nc.variables['uva_corr'].long_name = 'Correlation coefficient of atmos velocity'
    nc.variables['uva_arg_mean'].long_name = 'Mean of wind velocity angle'
    nc.variables['uva_arg_mean'].units = 'rad'
    nc.variables['uva_arg_std'].long_name = 'Standard deviation of wind velocity angle'
    nc.variables['uva_arg_std'].units = 'rad'
    nc.variables['uo_mean'].long_name = 'Mean of ocean downwind velocity'
    nc.variables['uo_mean'].units = 'm/s'
    nc.variables['uo_std'].long_name = 'Standard deviation of ocean downwind velocity'
    nc.variables['uo_std'].units = 'm/s'
    nc.variables['uo_skew'].long_name = 'Skewness of ocean downwind velocity'
    nc.variables['uo_kurt'].long_name = 'Kurtosis of ocean downwind velocity'
    nc.variables['vo_mean'].long_name = 'Mean of ocean crosswind velocity'
    nc.variables['vo_mean'].units = 'm/s'
    nc.variables['vo_std'].long_name = 'Standard deviation of ocean crosswind velocity'
    nc.variables['vo_std'].units = 'm/s'
    nc.variables['vo_skew'].long_name = 'Skewness of ocean crosswind velocity'
    nc.variables['vo_kurt'].long_name = 'Kurtosis of ocean crosswind velocity'
    nc.variables['uvo_corr'].long_name = 'Correlation coefficient of ocean velocity'
    nc.variables['uvo_arg_mean'].long_name = 'Mean of current velocity angle'
    nc.variables['uvo_arg_mean'].units = 'rad'
    nc.variables['uvo_arg_std'].long_name = 'Standard deviation of current velocity angle'
    nc.variables['uvo_arg_std'].units = 'rad'
    nc.variables['mkea'].long_name = 'Atmos mean kinetic energy density'
    nc.variables['mkea'].units = 'm^2/s^2'
    nc.variables['ekea'].long_name = 'Atmos eddy kinetic energy density'
    nc.variables['ekea'].units = 'm^2/s^2'
    nc.variables['mkea_int'].long_name = 'Atmos integrated mean kinetic energy'
    nc.variables['mkea_int'].units = 'J/m^2'
    nc.variables['ekea_int'].long_name = 'Atmos integrated eddy kinetic energy'
    nc.variables['ekea_int'].units = 'J/m^2'
    nc.variables['mwwa'].long_name = 'Atmos mean wind work'
    nc.variables['mwwa'].units = 'W/m^2'
    nc.variables['ewwa'].long_name = 'Atmos eddy wind work'
    nc.variables['ewwa'].units = 'W/m^2'
    nc.variables['mkeo'].long_name = 'Ocean mean kinetic energy density'
    nc.variables['mkeo'].units = 'm^2/s^2'
    nc.variables['ekeo'].long_name = 'Ocean eddy kinetic energy density'
    nc.variables['ekeo'].units = 'm^2/s^2'
    nc.variables['mkeo_int'].long_name = 'Ocean integrated mean kinetic energy'
    nc.variables['mkeo_int'].units = 'J/m^2'
    nc.variables['ekeo_int'].long_name = 'Ocean integrated eddy kinetic energy'
    nc.variables['ekeo_int'].units = 'J/m^2'
    nc.variables['mwwo'].long_name = 'Ocean mean wind work'
    nc.variables['mwwo'].units = 'W/m^2'
    nc.variables['ewwo'].long_name = 'Ocean eddy wind work'
    nc.variables['ewwo'].units = 'W/m^2'
    nc.variables['taux_mean'].long_name = 'Mean of zonal wind stress'
    nc.variables['taux_mean'].units = 'Pa'
    nc.variables['taux_std'].long_name = 'Standard deviation of zonal wind stress'
    nc.variables['taux_std'].units = 'Pa'
    nc.variables['taux_skew'].long_name = 'Skewness of zonal wind stress'
    nc.variables['taux_kurt'].long_name = 'Kurtosis of zonal wind stress'
    nc.variables['tauy_mean'].long_name = 'Mean of meridional wind stress'
    nc.variables['tauy_mean'].units = 'Pa'
    nc.variables['tauy_std'].long_name = 'Standard deviation of meridional wind stress'
    nc.variables['tauy_std'].units = 'Pa'
    nc.variables['tauy_skew'].long_name = 'Skewness of meridional wind stress'
    nc.variables['tauy_kurt'].long_name = 'Kurtosis of meridional wind stress'
    nc.variables['tauxy_corr'].long_name = 'Correlation coefficient of wind stress'
    nc.variables['tauxy_arg_mean'].long_name = 'Mean of wind stress angle'
    nc.variables['tauxy_arg_mean'].units = 'rad'
    nc.variables['tauxy_arg_std'].long_name = 'Standard deviation of wind stress angle'
    nc.variables['tauxy_arg_std'].units = 'rad'
    nc.variables['ha_mean'].long_name = 'Mean of atmos boundary layer depth'
    nc.variables['ha_mean'].units = 'm'
    nc.variables['ha_std'].long_name = 'Standard deviation of atmos boundary layer depth'
    nc.variables['ha_std'].units = 'm'
    nc.variables['ha_skew'].long_name = 'Skewness of atmos boundary layer depth'
    nc.variables['ha_kurt'].long_name = 'Kurtosis of atmos boundary layer depth'
    nc.variables['ho_mean'].long_name = 'Mean of ocean boundary layer depth'
    nc.variables['ho_mean'].units = 'm'
    nc.variables['ho_std'].long_name = 'Standard deviation of ocean boundary layer depth'
    nc.variables['ho_std'].units = 'm'
    nc.variables['ho_skew'].long_name = 'Skewness of ocean boundary layer depth'
    nc.variables['ho_kurt'].long_name = 'Kurtosis of ocean boundary layer depth'
    nc.variables['Cd_mean'].long_name = 'Mean of air-sea momentum transfer coef'
    nc.variables['Cd_std'].long_name = 'Standard deviation of air-sea momentum transfer coef'
    nc.variables['Cd_skew'].long_name = 'Skewness of air-sea momentum transfer coef'
    nc.variables['Cd_kurt'].long_name = 'Kurtosis of air-sea momentum transfer coef'
    nc.variables['ustar_mean'].long_name = 'Mean of air friction velocity scale'
    nc.variables['ustar_mean'].units = 'm/s'
    nc.variables['ustar_std'].long_name = 'Standard deviation of air friction velocity scale'
    nc.variables['ustar_std'].units = 'm/s'
    nc.variables['ustar_skew'].long_name = 'Skewness of air friction velocity scale'
    nc.variables['ustar_kurt'].long_name = 'Kurtosis of air friction velocity scale'
    nc.variables['Txa_mean'].long_name = 'Mean of atmos zonal transport'
    nc.variables['Txa_mean'].units = 'm^2/s'
    nc.variables['Txa_std'].long_name = 'Standard deviation of atmos zonal transport'
    nc.variables['Txa_std'].units = 'm^2/s'
    nc.variables['Txa_skew'].long_name = 'Skewness of atmos zonal transport'
    nc.variables['Txa_kurt'].long_name = 'Kurtosis of atmos zonal transport'
    nc.variables['Tya_mean'].long_name = 'Mean of atmos meridional transport'
    nc.variables['Tya_mean'].units = 'm^2/s'
    nc.variables['Tya_std'].long_name = 'Standard deviation of atmos meridional transport'
    nc.variables['Tya_std'].units = 'm^2/s'
    nc.variables['Tya_skew'].long_name = 'Skewness of atmos meridional transport'
    nc.variables['Tya_kurt'].long_name = 'Kurtosis of atmos meridional transport'
    nc.variables['Txya_corr'].long_name = 'Correlation coefficient of atmos transport'
    nc.variables['Txya_arg_mean'].long_name = 'Mean of atmos transport angle'
    nc.variables['Txya_arg_mean'].units = 'rad'
    nc.variables['Txya_arg_std'].long_name = 'Standard deviation of atmos transport angle'
    nc.variables['Txya_arg_std'].units = 'rad'
    nc.variables['Txo_mean'].long_name = 'Mean of ocean downwind transport'
    nc.variables['Txo_mean'].units = 'm^2/s'
    nc.variables['Txo_std'].long_name = 'Standard deviation of ocean downwind transport'
    nc.variables['Txo_std'].units = 'm^2/s'
    nc.variables['Txo_skew'].long_name = 'Skewness of ocean downwind transport'
    nc.variables['Txo_kurt'].long_name = 'Kurtosis of ocean downwind transport'
    nc.variables['Tyo_mean'].long_name = 'Mean of ocean crosswind transport'
    nc.variables['Tyo_mean'].units = 'm^2/s'
    nc.variables['Tyo_std'].long_name = 'Standard deviation of ocean crosswind transport'
    nc.variables['Tyo_std'].units = 'm^2/s'
    nc.variables['Tyo_skew'].long_name = 'Skewness of ocean crosswind transport'
    nc.variables['Tyo_kurt'].long_name = 'Kurtosis of ocean crosswind transport'
    nc.variables['Txyo_corr'].long_name = 'Correlation coefficient of ocean transport'
    nc.variables['Txyo_arg_mean'].long_name = 'Mean of ocean transport angle'
    nc.variables['Txyo_arg_mean'].units = 'rad'
    nc.variables['Txyo_arg_std'].long_name = 'Standard deviation of ocean transport angle'
    nc.variables['Txyo_arg_std'].units = 'rad'
    nc.variables['dist_mse_var'].long_name = 'Distance between MSE and VAR'
    nc.variables['mean_crps'].long_name = 'Mean of continuous ranked proper score'
    nc.variables['mean_wdist'].long_name = 'Mean of Wasserstein distance'
    
####################################################################################

# Set param.
dirm = '/srv/storage/ithaca@storage2.rennes.grid5000.fr/lli/ekman/coare_new'
dirs = os.listdir(dirm)
dtype = 'float64' # 'float32' or 'float64'
n_samples = 1000 # for Wasserstein distance

# Read observations
data = np.load('LOTUS3_rot_data.npz')
z_obs = data['z'][:-2].astype(dtype) # obs. depth (m)
uvm_obs = np.concatenate((data['um'][:-2].astype(dtype), data['vm'][:-2].astype(dtype))) # mean downwind and crosswind current (m/s)
uvs_obs = np.concatenate(( data['ue'][:-2].astype(dtype) / 2.0*53**(1/2), \
                           data['ve'][:-2].astype(dtype) / 1.7*53**(1/2) )) # from CI to std (53 is the effective dof)
data.close()

ftype = 'f8' if dtype == 'float64' else 'f4'
for s in dirs:
    datdir = os.path.join(dirm, s) 
    print(f'Read data from {datdir}')
    nt = len(glob.glob(os.path.join(datdir,'t_*.npz')))

    # Read param
    param = torch.load(os.path.join(datdir,'param.pth'))
    rhoa, rhoo = param['rhoa'], param['rhoo'] # air and water densities (kg/m^3)
    uga, vga = param['uga'].real, param['uga'].imag # geostrophic wind (m/s)
    ugo, vgo = param['ugo'].real, param['ugo'].imag # geo. current (m/s)
    
    # Read operators
    data = np.load(os.path.join(datdir,'operators.npz'))
    Wa, Wo = data['Wa'].astype(dtype), data['Wo'].astype(dtype) # integration weights (m)
    za, zo = data['za'].astype(dtype), data['zo'].astype(dtype) # vertical axes (m)
    data.close()

    # Init. output
    file1 = os.path.join(datdir,'stats_rot.nc')
    if os.path.exists(file1):
        os.remove(file1)
    nc = Dataset(file1, 'w', format='NETCDF4')
    nc.createDimension('t', nt)
    nc.createDimension('za', len(za))
    nc.createDimension('zo', len(zo))
    init_netcdf(nc, ftype)
    nc.variables['za'][:] = za
    nc.variables['zo'][:] = zo    
    idz = [abs(zo - val).argmin() for val in z_obs] # obs. indices

    # Time iterations
    for n in range(nt):
        
        # Read input data
        print(f'file: t_{n+1}.npz')
        data = np.load(os.path.join(datdir,f't_{n+1}.npz'))
        nc.variables['t'][n] = float(data['t']) # time (days)
        ua, va = uga + data['ua'].astype(dtype), vga + data['va'].astype(dtype) # total wind (m/s)
        uo, vo = ugo + data['uo'].astype(dtype), vgo + data['vo'].astype(dtype) # total current (m/s)
        ha, ho = data['ha'].astype(dtype), data['ho'].astype(dtype) # boundary layers depth (m)
        Cd, ustar = data['Cd'].astype(dtype), data['ustar'].astype(dtype) # drag coef. (nondim) and friction velocity (m/s)
        taux, tauy = rhoa * data['taux'].astype(dtype), rhoa * data['tauy'].astype(dtype) # wind stress (Pa)
        data.close()

        # Rotate into wind stress coordinates
        theta = np.arctan2(tauy, taux)[:,None] # wind stress angle (rad)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        uo_rot = cos_theta * uo + sin_theta * vo  # downwind component (m/s)
        vo_rot = sin_theta * uo - cos_theta * vo  # crosswind component (m/s)

        # Atmos lower-order stats
        Tx = (Wa*ua).sum(axis=-1) # zonal (eastward) transport (m^2/s)
        Ty = (Wa*va).sum(axis=-1) # meridional (northward) transport (m^2/s)
        nc.variables['ua_mean'][n] = ua.mean(axis=0)
        nc.variables['va_mean'][n] = va.mean(axis=0)
        nc.variables['ha_mean'][n] = ha.mean() 
        nc.variables['Cd_mean'][n] = Cd.mean() 
        nc.variables['ustar_mean'][n] = ustar.mean() 
        nc.variables['taux_mean'][n] = taux.mean()
        nc.variables['tauy_mean'][n] = tauy.mean()
        nc.variables['Txa_mean'][n] = Tx.mean() 
        nc.variables['Tya_mean'][n] = Ty.mean() 
        nc.variables['ua_std'][n] = ua.std(axis=0)
        nc.variables['va_std'][n] = va.std(axis=0)
        nc.variables['ha_std'][n] = ha.std()
        nc.variables['Cd_std'][n] = Cd.std()
        nc.variables['ustar_std'][n] = ustar.std()
        nc.variables['taux_std'][n] = taux.std()
        nc.variables['tauy_std'][n] = tauy.std()
        nc.variables['Txa_std'][n] = Tx.std() 
        nc.variables['Tya_std'][n] = Ty.std() 
        nc.variables['uva_corr'][n] = ( (ua - nc.variables['ua_mean'][n].data) \
                                       *(va - nc.variables['va_mean'][n].data) ).mean(axis=0) \
                                    / ( nc.variables['ua_std'][n].data * nc.variables['va_std'][n].data )
        nc.variables['tauxy_corr'][n] = ( (taux - nc.variables['taux_mean'][n].data) \
                                        * (tauy - nc.variables['tauy_mean'][n].data) ).mean() \
                                      / ( nc.variables['taux_std'][n].data * nc.variables['tauy_std'][n].data )
        nc.variables['Txya_corr'][n] = ( (Tx - nc.variables['Txa_mean'][n].data) \
                                       * (Ty - nc.variables['Tya_mean'][n].data) ).mean() \
                                     / ( nc.variables['Txa_std'][n].data * nc.variables['Tya_std'][n].data )

        # Atmos higher-order stats
        nc.variables['ua_skew'][n] = skew(ua, axis=0, bias=False)   
        nc.variables['va_skew'][n] = skew(va, axis=0, bias=False) 
        nc.variables['ha_skew'][n] = skew(ha, bias=False)   
        nc.variables['Cd_skew'][n] = skew(Cd, bias=False)   
        nc.variables['ustar_skew'][n] = skew(ustar, bias=False)   
        nc.variables['taux_skew'][n] = skew(taux, bias=False)   
        nc.variables['tauy_skew'][n] = skew(tauy, bias=False)   
        nc.variables['Txa_skew'][n] = skew(Tx, bias=False)   
        nc.variables['Tya_skew'][n] = skew(Ty, bias=False)   
        nc.variables['ua_kurt'][n] = kurtosis(ua, axis=0, bias=False) 
        nc.variables['va_kurt'][n] = kurtosis(va, axis=0, bias=False) 
        nc.variables['ha_kurt'][n] = kurtosis(ha, bias=False) 
        nc.variables['Cd_kurt'][n] = kurtosis(Cd, bias=False) 
        nc.variables['ustar_kurt'][n] = kurtosis(ustar, bias=False) 
        nc.variables['taux_kurt'][n] = kurtosis(taux, bias=False) 
        nc.variables['tauy_kurt'][n] = kurtosis(tauy, bias=False) 
        nc.variables['Txa_kurt'][n] = kurtosis(Tx, bias=False)   
        nc.variables['Tya_kurt'][n] = kurtosis(Ty, bias=False)   

        # Atmos angular stats.
        theta = np.arctan2(tauy, taux)
        nc.variables['tauxy_arg_mean'][n] = circmean(theta)
        nc.variables['tauxy_arg_std'][n] = circstd(theta)
        theta = np.arctan2(Ty, Tx)
        nc.variables['Txya_arg_mean'][n] = circmean(theta)
        nc.variables['Txya_arg_std'][n] = circstd(theta)
        theta = np.arctan2(va, ua)
        nc.variables['uva_arg_mean'][n] = circmean(theta, axis=0)
        nc.variables['uva_arg_std'][n] = circstd(theta, axis=0)

        # Atmos energy and work
        nc.variables['mkea'][n] = ((nc.variables['ua_mean'][n].data)**2 + (nc.variables['va_mean'][n].data)**2)/2
        nc.variables['ekea'][n] = ((nc.variables['ua_std'][n].data)**2 + (nc.variables['va_std'][n].data)**2)/2 
        nc.variables['mkea_int'][n] = rhoa * (Wa * nc.variables['mkea'][n].data).sum()
        nc.variables['ekea_int'][n] = rhoa * (Wa * nc.variables['ekea'][n].data).sum()
        nc.variables['mwwa'][n] = nc.variables['ua_mean'][n,0].data * nc.variables['taux_mean'][n].data \
                                + nc.variables['va_mean'][n,0].data * nc.variables['tauy_mean'][n].data
        nc.variables['ewwa'][n] = ( (ua[:,0] - nc.variables['ua_mean'][n,0].data) * \
                                    (taux - nc.variables['taux_mean'][n].data) + \
                                    (va[:,0] - nc.variables['va_mean'][n,0].data) * \
                                    (tauy - nc.variables['tauy_mean'][n].data) ).mean()

        # Ocean lower-order stats
        Tx = (Wo*uo_rot).sum(axis=-1)
        Ty = (Wo*vo_rot).sum(axis=-1)
        nc.variables['uo_mean'][n] = uo_rot.mean(axis=0) 
        nc.variables['vo_mean'][n] = vo_rot.mean(axis=0)
        nc.variables['ho_mean'][n] = ho.mean() 
        nc.variables['Txo_mean'][n] = Tx.mean() 
        nc.variables['Tyo_mean'][n] = Ty.mean() 
        nc.variables['uo_std'][n] = uo_rot.std(axis=0)
        nc.variables['vo_std'][n] = vo_rot.std(axis=0)
        nc.variables['ho_std'][n] = ho.std()
        nc.variables['Txo_std'][n] = Tx.std() 
        nc.variables['Tyo_std'][n] = Ty.std() 
        nc.variables['uvo_corr'][n] = ( (uo_rot - nc.variables['uo_mean'][n].data) \
                                      * (vo_rot - nc.variables['vo_mean'][n].data) ).mean(axis=0) \
                                    / ( nc.variables['uo_std'][n].data * nc.variables['vo_std'][n].data )
        nc.variables['Txyo_corr'][n] = ( (Tx - nc.variables['Txo_mean'][n].data) \
                                       * (Ty - nc.variables['Tyo_mean'][n].data) ).mean() \
                                     / ( nc.variables['Txo_std'][n].data * nc.variables['Tyo_std'][n].data )

        # Ocean higher-order stats
        nc.variables['uo_skew'][n] = skew(uo_rot, axis=0, bias=False)   
        nc.variables['vo_skew'][n] = skew(vo_rot, axis=0, bias=False) 
        nc.variables['ho_skew'][n] = skew(ho, bias=False)   
        nc.variables['Txo_skew'][n] = skew(Tx, bias=False)   
        nc.variables['Tyo_skew'][n] = skew(Ty, bias=False)   
        nc.variables['uo_kurt'][n] = kurtosis(uo_rot, axis=0, bias=False) 
        nc.variables['vo_kurt'][n] = kurtosis(vo_rot, axis=0, bias=False) 
        nc.variables['ho_kurt'][n] = kurtosis(ho, bias=False) 
        nc.variables['Txo_kurt'][n] = kurtosis(Tx, bias=False)   
        nc.variables['Tyo_kurt'][n] = kurtosis(Ty, bias=False)   

        # Ocean angular stats.
        theta = np.arctan2(Ty, Tx)
        nc.variables['Txyo_arg_mean'][n] = circmean(theta)
        nc.variables['Txyo_arg_std'][n] = circstd(theta)
        theta = np.arctan2(vo_rot, uo_rot)
        nc.variables['uvo_arg_mean'][n] = circmean(theta, axis=0)
        nc.variables['uvo_arg_std'][n] = circstd(theta, axis=0)
        
        # Ocean energy and work
        nc.variables['mkeo'][n] = ((nc.variables['uo_mean'][n].data)**2 + (nc.variables['vo_mean'][n].data)**2)/2
        nc.variables['ekeo'][n] = ((nc.variables['uo_std'][n].data)**2 + (nc.variables['vo_std'][n].data)**2)/2 
        nc.variables['mkeo_int'][n] = rhoo * (Wo * nc.variables['mkeo'][n].data).sum()
        nc.variables['ekeo_int'][n] = rhoo * (Wo * nc.variables['ekeo'][n].data).sum()
        nc.variables['mwwo'][n] = nc.variables['uo_mean'][n,0].data * nc.variables['taux_mean'][n].data \
                                + nc.variables['vo_mean'][n,0].data * nc.variables['tauy_mean'][n].data
        
        nc.variables['ewwo'][n] = ( (uo_rot[:,0] - nc.variables['uo_mean'][n,0].data) * \
                                    (taux - nc.variables['taux_mean'][n].data) + \
                                    (vo_rot[:,0] - nc.variables['vo_mean'][n,0].data) * \
                                    (tauy - nc.variables['tauy_mean'][n].data) ).mean()
        
        # Ensemble forecast scores
        uv = np.concatenate((uo_rot[:,idz], vo_rot[:,idz]), axis=-1) 
        mse = ((uv.mean(axis=0) - uvm_obs)**2).mean() 
        var = (uv.var(axis=0)).mean() 
        nc.variables['dist_mse_var'][n] = abs(mse - (uv.shape[0]+1)/uv.shape[0] * var)
        #nc.variables['mean_crps'][n] = crps_ensemble(uvm_obs, uv.T).mean()
        
        # Wasserstein distance
        samples = np.random.normal(uvm_obs, uvs_obs, (n_samples, len(uvm_obs)))
        wd = np.array([wdist(uv[:,k], samples[:,k]) for k in range(len(uvm_obs))])
        nc.variables['mean_wdist'][n] = wd.mean()
        
        # CRPS
        crps_samples = np.array([crps_ensemble(sample, uv.T).mean() for sample in samples])
        nc.variables['mean_crps'][n] = crps_samples.mean()
    
    # Close output
    nc.close()
    print(f'NetCDF file saved in {datdir} \n')
