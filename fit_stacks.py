#!/usr/bin/env python

import sys

import numpy as np
import pylab as plt

from lmfit import Model
from lmfit.models import PolynomialModel
from astropy.convolution import convolve
from astropy.convolution import Gaussian1DKernel

from crrlpy import crrls, utils

def plot0(data, fit_values, n):
    """
    overlay bandpass on data
    """
    
    plt.suptitle(n)
    #plt.plot(data[:,0].data, data[:,1].data, 'k-', drawstyle='steps')
    plt.plot(data[:,0], data[:,1], 'r-', drawstyle='steps')
    plt.plot(data[:,0].data, fit_values, 'g--', lw=2.)
    #plt.plot(data[:,0], fit1_values, 'g--', lw=2.)
    plt.show()
    plt.close()
    
def plot1(data, fit_values, n):
    """
    corrected data
    """
    
    plt.suptitle(n)
    plt.plot(data[:,0], data[:,1] - fit_values, 'k-', drawstyle='steps', lw=2.)
    #plt.plot(data[:,0].data, data[:,1].data - fit1_values, 'r-', drawstyle='steps', lw=2.)
    plt.show()
    plt.close()

def fit_bandpass(data, degree):
    """
    """
    
    #mask = (data[:,0].mask) | (data[:,1].mask)
    #data[:,0].mask = np.copy(mask)
    #data[:,1].mask = np.copy(mask)
    
    if degree <= 7:
        mod = PolynomialModel(degree)
        params = mod.make_params()
        params = mod.guess(data[:,1].compressed(), x=data[~data[:,1].mask,0])
        fit = mod.fit(data[:,1].compressed(), x=data[~data[:,1].mask,0], params=params)
        fit_values = fit.eval(x=data[:,0])
        fit_values_mskd = fit.best_fit
        
    else:
        fit = np.polyfit(data[:,0], data[:,1], degree)
        poly = np.poly1d(fit)
        fit_values = poly(data[:,0])
        fit_values_mskd = poly(data[~data[:,1].mask,0])
        
    return fit_values, fit_values_mskd

def print_fit(qn, atom, fit):
    
    if atom == 'He':
        dv = dv_herrl
    elif atom == 'H':
        dv = dv_hrrl
    else:
        dv = 0
    
    print('{0}{1}a: {2:.3f} {3:.3f}   {4:.1f} {5:.1f}  {6:.1f} {7:.1f}'.format(atom, qn, 
                                                       fit.params['g{0}_amplitude'.format(atom)].value, 
                                                       fit.params['g{0}_amplitude'.format(atom)].stderr,
                                                       fit.params['g{0}_center'.format(atom)].value - dv, 
                                                       fit.params['g{0}_center'.format(atom)].stderr,
                                                       crrls.sigma2fwhm(fit.params['g{0}_sigma'.format(atom)].value), 
                                                       crrls.sigma2fwhm(fit.params['g{0}_sigma'.format(atom)].stderr),))

stacks_dir = 'stacks'
output_dir = 'stack_results'

stacks = {'AGBT02A_028_02_P0_orion_spec_proc_stack_C279a.ascii':      {'gC':{'v0':5., 'dv':1., 'a':-0.02},
                                                                       'gH':{'v0':143.5, 'dv':1., 'a':0.02},
                                                                       },
          'AGBT02A_028_02_P1..2_orion_spec_proc_stack_C265a.ascii':   {},
          'AGBT02A_028_02_S0..2_orion_spec_proc_stack_C130a.ascii':   {'gC' :{'v0':8., 'dv':1., 'a':0.5},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_02_S3..12_orion_spec_proc_stack_C151a.ascii':  {'gC' :{'v0':8., 'dv':1., 'a':0.5},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_04_S0..2_orion_spec_proc_stack_C145a.ascii':   {'gC' :{'v0':8., 'dv':1., 'a':0.5},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_04_S3..5_orion_spec_proc_stack_C137a.ascii':   {'gC' :{'v0':8., 'dv':1., 'a':0.5},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       'fit_poly': {'degree':7, 'v0':20.0, 'dv':30., 'v0_hrrl':-6.9, 'dv_hrrl':50.},
                                                                       },
          'AGBT02A_028_04_S6..8_orion_spec_proc_stack_C155a.ascii':   {'gC' :{'v0':6., 'dv':1., 'a':0.4},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_05_L0..4_orion_spec_proc_stack_C164a.ascii':   {'gC' :{'v0':6., 'dv':1., 'a':0.4},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_05_L6..9_orion_spec_proc_stack_C156a.ascii':   {'gC' :{'v0':6., 'dv':1., 'a':0.4},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          'AGBT02A_028_05_L10..16_orion_spec_proc_stack_C174a.ascii': {'gC' :{'v0':6., 'dv':1., 'a':0.4},
                                                                       'gHe':{'v0':12., 'dv':1., 'a':0.1},
                                                                       'gH' :{'v0':143.5, 'dv':5., 'a':5.},
                                                                       },
          }

if __name__ == '__main__':
    
    #for stack in ['AGBT02A_028_04_S3..5_orion_spec_proc_stack_C137a.ascii']:
    for stack in stacks.keys():
    
        stack_in = '{0}/{1}'.format(stacks_dir, stack)
        qn = int(stack.split('_')[-1].split('.')[0][1:-1])
        plot_out = '{0}/{1}'.format(output_dir, '{0}.pdf'.format('.'.join(stack.split('.')[:-1])))
        stack_out = '{0}/{1}'.format(output_dir, '{0}.ascii'.format('.'.join(stack.split('.')[:-1])))
        
        #print('{0:.2f}'.format(np.rad2deg(3e8/crrls.n2f(qn, 'RRL_HIalpha')[0]*1e-6/100.)*60.))
        
        #print('Will fit {0} and output a plot in {1} .'.format(stack_in, plot_out))
        
        data = np.ma.masked_invalid(np.loadtxt(stack_in))
        
        #data[:,0].mask = data[:,1].mask
        data[:,0] = data[:,0]*1e-3      # km s-1
        print(utils.get_min_sep(data[:,0]), 1./utils.get_min_sep(data[:,0]))
        
        if utils.get_min_sep(data[:,0]) < 0.5:
            
            # Create kernel
            g = Gaussian1DKernel(stddev=1./utils.get_min_sep(data[:,0]))

            # Convolve data
            data_0 = data[:,0] #convolve(data[:,0], g)
            data_1 = convolve(data[:,1], g)
            data_2 = convolve(data[:,2], g)
        
            data = np.ma.masked_invalid(np.array([data_0, data_1, data_2]).T)
        
        # HRRL velocity shift
        dv_hrrl = crrls.df2dv(crrls.n2f(qn, 'RRL_CIalpha'), 
                              crrls.n2f(qn, 'RRL_CIalpha') - crrls.n2f(qn, 'RRL_HIalpha'))[0]*1e-3
        
        # HeRRL velocity shift
        dv_herrl = crrls.df2dv(crrls.n2f(qn, 'RRL_CIalpha'), 
                               crrls.n2f(qn, 'RRL_CIalpha') - crrls.n2f(qn, 'RRL_HeIalpha'))[0]*1e-3
        
        try: 
            stacks[stack]['fit_poly']
            #print('Removing a polynomial')
            # Determine blanking regions
            ch0_crrl = utils.best_match_indx(stacks[stack]['fit_poly']['v0'] + stacks[stack]['fit_poly']['dv'], data[:,0])
            chf_crrl = utils.best_match_indx(stacks[stack]['fit_poly']['v0'] - stacks[stack]['fit_poly']['dv'], data[:,0])
            ch0_crrl,chf_crrl = np.sort([ch0_crrl,chf_crrl])
            ch0_hrrl = utils.best_match_indx(stacks[stack]['fit_poly']['v0_hrrl'] + stacks[stack]['fit_poly']['dv_hrrl'] + dv_hrrl, data[:,0])
            chf_hrrl = utils.best_match_indx(stacks[stack]['fit_poly']['v0_hrrl'] - stacks[stack]['fit_poly']['dv_hrrl'] + dv_hrrl, data[:,0])
            ch0_hrrl,chf_hrrl = np.sort([ch0_hrrl,chf_hrrl])
            
            # Fit parameters
            degree = stacks[stack]['fit_poly']['degree']
            
            # Mask the line regions
            data[ch0_crrl:chf_crrl,1].mask = True
            data[ch0_hrrl:chf_hrrl,1].mask = True
            
            fit_values, fit_values_mskd = fit_bandpass(data, degree)
            
            #plot0(data, fit_values, qn)
            
            data[ch0_crrl:chf_crrl,1].mask = False
            data[ch0_hrrl:chf_hrrl,1].mask = False
            
            #plot1(data, fit_values, qn)
            
            data[:,1] = data[:,1] - fit_values
            
            
        except KeyError:
            pass
    
        if 'gC' in stacks[stack].keys() and 'gHe' not in stacks[stack].keys() and 'gH' not in stacks[stack].keys():
            
            model = Model(crrls.gaussian, prefix='gC_')
            
            params = model.make_params()
            
            params['gC_center'].set(value=stacks[stack]['gC']['v0'], vary=True, max=30., min=-10.)
            params['gC_amplitude'].set(value=stacks[stack]['gC']['a'], vary=True)
            params['gC_sigma'].set(value=stacks[stack]['gC']['dv'], vary=True, min=0.1, max=20.)
            
            fit = model.fit(data[:,1].compressed(), x=data[~data[:,1].mask,0], params=params)
            
            print_fit(qn, 'C', fit)
            
        elif 'gC' in stacks[stack].keys() and 'gHe' in stacks[stack].keys() and 'gH' not in stacks[stack].keys():
            
            model = Model(crrls.gaussian, prefix='gC_') + Model(crrls.gaussian, prefix='gHe_')
            
            params = model.make_params()
            
            params['gC_center'].set(value=stacks[stack]['gC']['v0'], vary=True, max=30., min=-10.)
            params['gC_amplitude'].set(value=stacks[stack]['gC']['a'], vary=True)
            params['gC_sigma'].set(value=stacks[stack]['gC']['dv'], vary=True, min=0.1, max=20.)
            
            params['gHe_center'].set(value=stacks[stack]['gHe']['v0'], vary=True, max=30., min=-10.)
            params['gHe_amplitude'].set(value=stacks[stack]['gHe']['a'], vary=True)
            params['gHe_sigma'].set(value=stacks[stack]['gHe']['dv'], vary=True, min=0.1, max=20.)
            
            fit = model.fit(data[:,1].compressed(), x=data[~data[:,1].mask,0], params=params)
            
            print_fit(qn, 'C', fit)
            print_fit(qn, 'He', fit)
        
        elif 'gC' in stacks[stack].keys() and 'gHe' not in stacks[stack].keys() and 'gH' in stacks[stack].keys():
            
            model = Model(crrls.gaussian, prefix='gC_') + Model(crrls.gaussian, prefix='gH_')
            
            params = model.make_params()
            
            params['gC_center'].set(value=stacks[stack]['gC']['v0'], vary=True, max=30., min=-10.)
            params['gC_amplitude'].set(value=stacks[stack]['gC']['a'], vary=True)
            params['gC_sigma'].set(value=stacks[stack]['gC']['dv'], vary=True, min=0.1, max=20.)
            
            params['gH_center'].set(value=stacks[stack]['gH']['v0'], vary=True, max=250., min=100.)
            params['gH_amplitude'].set(value=stacks[stack]['gH']['a'], vary=True, min=1e-8)
            params['gH_sigma'].set(value=stacks[stack]['gH']['dv'], vary=True, min=0.1, max=20.)
            
            fit = model.fit(data[:,1].compressed(), x=data[~data[:,1].mask,0], params=params)
            
            print_fit(qn, 'C', fit)
            print_fit(qn, 'H', fit)
        
        elif 'gC' in stacks[stack].keys() and 'gHe' in stacks[stack].keys() and 'gH' in stacks[stack].keys():
            
            model = Model(crrls.gaussian, prefix='gC_') + \
                    Model(crrls.gaussian, prefix='gHe_') + \
                    Model(crrls.gaussian, prefix='gH_')
            
            params = model.make_params()
            
            params['gC_center'].set(value=stacks[stack]['gC']['v0'], vary=True, max=30., min=-10.)
            params['gC_amplitude'].set(value=stacks[stack]['gC']['a'], vary=True)
            params['gC_sigma'].set(value=stacks[stack]['gC']['dv'], vary=True, min=0.1, max=20.)
            
            params['gHe_center'].set(value=stacks[stack]['gHe']['v0'], vary=True, max=30., min=-10.)
            params['gHe_amplitude'].set(value=stacks[stack]['gHe']['a'], vary=True)
            params['gHe_sigma'].set(value=stacks[stack]['gHe']['dv'], vary=True, min=0.1, max=20.)
            
            params['gH_center'].set(value=stacks[stack]['gH']['v0'], vary=True, max=250., min=100.)
            params['gH_amplitude'].set(value=stacks[stack]['gH']['a'], vary=True, min=1e-8)
            params['gH_sigma'].set(value=stacks[stack]['gH']['dv'], vary=True, min=0.1, max=20.)
            
            fit = model.fit(data[:,1].compressed(), x=data[~data[:,1].mask,0], params=params)
            
            print_fit(qn, 'C', fit)
            print_fit(qn, 'He', fit)
            print_fit(qn, 'H', fit)
        
        else:
            
            continue

        residuals = data[:,1] - fit.eval(x=data[:,0])
        
        fig = plt.figure(frameon=False, figsize=(6,5))
        #fig.suptitle(r'Orion P band spectra of C199$\alpha$')
        ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
        
        ax.step(data[:,0], data[:,1], 'r-', drawstyle='steps-pre', lw=1)
        ax.plot(data[:,0], fit.eval(x=data[:,0]), 'k-', lw=1)
        
        off_y = - 0.0015
        ax.plot(data[:,0], residuals + off_y, 'k:')
        ax.fill_between(data[:,0], [off_y - 3.*residuals.std()]*len(data), [off_y + 3*residuals.std()]*len(data), color='k', alpha=0.3)
        #ax.step(data1[:,0]*1e-3, data1[:,1]/data1[:,1].std(), 'b-', drawstyle='steps', lw=1, where='pre', label='LL')
        
        #ax.legend(loc=0)
        
        ax.set_xlabel('Velocity (km s$^{-1}$)')
        ax.set_ylabel('$T_{A}$ (arbitrary units)')
        
        ax.minorticks_on()
        ax.tick_params('both', direction='in', which='both',
                        bottom=True, top=True, left=True, right=True)
        
        plt.savefig(plot_out, 
                    bbox_inches='tight', 
                    pad_inches=0.06)
        plt.close()
        
        np.savetxt(stack_out, np.c_[data, fit.eval(x=data[:,0]), residuals])