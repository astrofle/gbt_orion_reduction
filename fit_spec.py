#!/usr/bin/env python

import sys

import numpy as np
import pylab as plt

from lmfit import Model

from crrlpy import crrls

if __name__ == '__main__':
    
    inspec = sys.argv[1] #'stacks/AGBT02A_028_02_S4.5_orion_spec_proc_stack_C154a.ascii'
    output = sys.argv[2] #'plots/AGBT02A_028_02_S4.5_orion_stack_C154a.pdf'
    
    data = np.ma.masked_invalid(np.loadtxt(inspec))
    data[:,0].mask = data[:,1].mask
    data[:,0] = data[:,0]*1e-3 #+ 4.
    
    model_g0 = Model(crrls.gaussian, prefix='g0_')
    model_g1 = Model(crrls.gaussian, prefix='g1_') # gaussian(x, sigma, center, amplitude):
    
    params_g0 = model_g0.make_params()
    params_g1 = model_g1.make_params()
    
    params_g0['g0_center'].set(value=5., vary=True, max=30., min=-10.)
    params_g0['g0_amplitude'].set(value=-0.5, vary=True)#, min=1e-8)
    params_g0['g0_sigma'].set(value=1., vary=True, min=0.1, max=20.)
    
    params_g1['g1_center'].set(value=-5.8, vary=True, max=15., min=-15.)
    params_g1['g1_amplitude'].set(value=4., vary=True, min=1e-8)
    params_g1['g1_sigma'].set(value=5.0, vary=True, min=0., max=20.)
    
    fit_g0 = model_g0.fit(data[:,1].compressed(), x=data[:,0].compressed(), params=params_g0)
    
    vel_hrrl = crrls.freq2vel(crrls.n2f(199, 'RRL_HIalpha'), 
                              crrls.vel2freq(crrls.n2f(199, 'RRL_CIalpha'), 
                                             data[:,0]*1e3))*1e-3
    vel_hrrl = np.ma.masked_where(data[:,0].mask, vel_hrrl)
    
    fit_g1 = model_g1.fit(data[:,1].compressed(), x=vel_hrrl.compressed(), params=params_g1)
    
    print(fit_g0.fit_report())
    print(fit_g1.fit_report())
    
    model_fit = fit_g0.best_fit + fit_g1.best_fit
    residuals = data[:,1].compressed() - model_fit
    
    fig = plt.figure(frameon=False, figsize=(6,5))
    #fig.suptitle(r'Orion P band spectra of C199$\alpha$')
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    
    ax.step(data[:,0], data[:,1], 'r-', drawstyle='steps-pre', lw=1)
    ax.plot(data[:,0].compressed(), model_fit, 'k-', lw=1)
    
    off_y = - 0.0015
    ax.plot(data[:,0].compressed(), residuals + off_y, 'k:')
    ax.fill_between(data[:,0].compressed(), off_y - 3.*residuals.std(), off_y + 3*residuals.std(), color='k', alpha=0.3)
    #ax.step(data1[:,0]*1e-3, data1[:,1]/data1[:,1].std(), 'b-', drawstyle='steps', lw=1, where='pre', label='LL')
    
    #ax.legend(loc=0)
    
    ax.set_xlabel('Velocity (km s$^{-1}$)')
    ax.set_ylabel('$T_{A}$ (arbitrary units)')
    
    ax.minorticks_on()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    
    plt.savefig(output, 
                bbox_inches='tight', 
                pad_inches=0.06)
    plt.close()