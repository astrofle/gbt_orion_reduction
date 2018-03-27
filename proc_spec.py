#!/usr/bin/env python

import sys
import numpy as np
import pylab as plt

from lmfit.models import PolynomialModel

from astropy.stats import sigma_clip

from crrlpy import crrls, utils

def plot0(data, fit_values, fit1_values, n):
    """
    """
    
    plt.suptitle(n)
    plt.plot(data[:,0].data, data[:,1].data, 'k-', drawstyle='steps')
    plt.plot(data[:,0].compressed(), data[:,1].compressed(), 'r-', drawstyle='steps')
    plt.plot(data[:,0].data, fit_values, 'b-')
    plt.plot(data[:,0], fit1_values, 'g--', lw=2.)
    plt.show()
    plt.close()
    
def plot1(data, fit_values, fit1_values, n):
    """
    """
    
    plt.suptitle(n)
    plt.plot(data[:,0].data, data[:,1].data - fit_values, 'b-', drawstyle='steps', lw=2.)
    plt.plot(data[:,0].data, data[:,1].data - fit1_values, 'r-', drawstyle='steps', lw=2.)
    plt.show()
    plt.close()
    
def fit_bandpass0(data, degree):
    """
    """
    
    mask = (data[:,0].mask) | (data[:,1].mask)
    data[:,0].mask = np.copy(mask)
    data[:,1].mask = np.copy(mask)
    
    if degree <= 7:
        mod = PolynomialModel(degree)
        params = mod.make_params()
        params = mod.guess(data[:,1].compressed(), x=data[:,0].compressed())
        fit = mod.fit(data[:,1].compressed(), x=data[:,0].compressed(), params=params)
        fit_values = fit.eval(x=data[:,0].data)
        fit_values_mskd = fit.best_fit
        
    else:
        fit = np.polyfit(data[:,0].compressed(), data[:,1].compressed(), degree)
        poly = np.poly1d(fit)
        fit_values = poly(data[:,0].data)
        fit_values_mskd = poly(data[:,0])
        
    return fit_values, fit_values_mskd

def fit_bandpass1(data, degree):
    """
    """
    
    mask = (data[:,0].mask) | (data[:,1].mask)
    data[:,0].mask = np.copy(mask)
    data[:,1].mask = np.copy(mask)
    
    if degree <= 7:
        mod = PolynomialModel(degree)
        params = mod.make_params()
        params = mod.guess(data[:,1].compressed(), x=data[:,0].compressed())
        fit = mod.fit(data[:,1].compressed(), x=data[:,0].compressed(), params=params)
        fit_values = fit.eval(x=data[:,0].data)
        fit_values_mskd = fit.best_fit
        
    else:
        fit = np.polyfit(data[:,0], data[:,1], degree)
        poly = np.poly1d(fit)
        fit_values = poly(data[:,0].data)
        fit_values_mskd = poly(data[:,0])
        
    return fit_values, fit_values_mskd

def main0(basename, spw, pol, sig, n_list, fit_dict, plot=0):
    
    rms = {}
    
    for n in n_list:
        
        print(n)
        
        filein = 'specs/{0}_orion_spec_spw{1}_pol{2}_sig{3}_n{4:.0f}.ascii'.format(basename, spw, pol, sig, n)
        
        dv_rrls = crrls.df2dv(crrls.n2f(n, 'RRL_CIalpha'), 
                              crrls.n2f(n, 'RRL_CIalpha') - crrls.n2f(n, 'RRL_HIalpha'))*1e-3
        
        data = np.ma.masked_invalid(np.loadtxt(filein))
        
        # Line blanking parameters
        v0 = fit_dict[n]['v0']
        dv = fit_dict[n]['dv']
        v0_hrrl = fit_dict[n]['v0_hrrl']
        dv_hrrl = fit_dict[n]['dv_hrrl']
        
        # Determine blanking regions
        ch0_crrl = utils.best_match_indx(v0 + dv, data[:,0]*1e-3)
        chf_crrl = utils.best_match_indx(v0 - dv, data[:,0]*1e-3)
        ch0_crrl,chf_crrl = np.sort([ch0_crrl,chf_crrl])
        ch0_hrrl = utils.best_match_indx(v0_hrrl + dv_hrrl + dv_rrls, data[:,0]*1e-3)
        chf_hrrl = utils.best_match_indx(v0_hrrl - dv_hrrl + dv_rrls, data[:,0]*1e-3)
        ch0_hrrl,chf_hrrl = np.sort([ch0_hrrl,chf_hrrl])
        
        # Fit parameters
        degree0 = fit_dict[n]['degree0']
        degree1 = fit_dict[n]['degree1']
        
        fit_values, fit_values_mskd = fit_bandpass0(data, degree0)
            
        # Mask the line regions
        data[ch0_crrl:chf_crrl].mask = True
        data[ch0_hrrl:chf_hrrl].mask = True
        
        # Fit the baseline on the masked data
        fit1_values, fit1_values_mskd = fit_bandpass1(data, degree1)
        
        if plot:
            plot0(data, fit_values, fit1_values, n)  # Plot data with bandpass solution
            plot1(data, fit_values, fit1_values, n)  # plot bandpass corrected data
            
        data[ch0_crrl:chf_crrl].mask = False
        data[ch0_hrrl:chf_hrrl].mask = False
        
        spec_bpcorr = np.ma.masked_invalid(data[:,1].data - fit1_values)
        
        # Mask outliers
        spec_bpcorr[ch0_crrl:chf_crrl].mask = True
        spec_bpcorr[ch0_hrrl:chf_hrrl].mask = True
        spec_bpcorr_mskd = np.ma.copy(sigma_clip(spec_bpcorr))
        spec_bpcorr_mskd[ch0_crrl:chf_crrl].mask = False
        spec_bpcorr_mskd[ch0_hrrl:chf_hrrl].mask = False
        
        # Apply outliers mask to original data
        data[:,0].mask = np.copy(spec_bpcorr_mskd.mask)
        data[:,1].mask = np.copy(spec_bpcorr_mskd.mask)
        data = np.ma.masked_invalid(data)
        
        # Re-fit the bandpass
        fit_values, fit_values_mskd = fit_bandpass0(data, degree0)
            
        # Mask the line regions
        data[ch0_crrl:chf_crrl].mask = True
        data[ch0_hrrl:chf_hrrl].mask = True
        
        # Fit the baseline on the masked data
        fit1_values, fit1_values_mskd = fit_bandpass1(data, degree1)
        
        if plot:
            plot0(data, fit_values, fit1_values, n) # Plot data with bandpass solution
            plot1(data, fit_values, fit1_values, n) # plot bandpass corrected data
        
        # Compute rms on line masked bandpass corrected data
        this_rms = np.ma.std(data[:,1] - fit1_values)
        
        # Remove the line mask
        data[ch0_crrl:chf_crrl].mask = False
        data[ch0_hrrl:chf_hrrl].mask = False
        data.fill_value = np.nan
        
        spec_bpcorr = np.ma.masked_invalid(data[:,1].filled() - fit1_values)
        spec_bpcorr.fill_value = np.nan
        
        output = '{0}_proc.ascii'.format('.'.join(filein.split('.')[:-1]))
        np.savetxt(output, np.c_[data[:,0], spec_bpcorr.filled(), fit1_values])
        
        # Save rms for this spectrum
        rms[output] = this_rms
    
    # Save rms values in a text file for later use
    rms_file = 'logs/{0}_orion_spec_spw{1}_pol{2}_sig{3}_rms.log'.format(basename, spw, pol, sig)
    with open(rms_file, 'w') as log:
        for key in rms.keys():
            log.write('{0}  {1}  {2}\n'.format(key, np.power(rms[key], -2.), rms[key]))
        
        
processes = {'AGBT02A_028_01_P0' : main0, # Bad
             'AGBT02A_028_01_C0' : main0,
             'AGBT02A_028_01_C1' : main0,
             'AGBT02A_028_01_C2' : main0,
             'AGBT02A_028_01_C3' : main0,
             'AGBT02A_028_01_L0' : main0,
             'AGBT02A_028_01_L1' : main0,
             'AGBT02A_028_01_L2' : main0,
             'AGBT02A_028_02_P0' : main0, # Good, absorption
             'AGBT02A_028_02_P1' : main0, # Not great
             'AGBT02A_028_02_P2' : main0,
             'AGBT02A_028_02_S0' : main0,
             'AGBT02A_028_02_S1' : main0,
             'AGBT02A_028_02_S2' : main0,
             'AGBT02A_028_02_S3' : main0,
             'AGBT02A_028_02_S4' : main0,
             'AGBT02A_028_02_S5' : main0,
             'AGBT02A_028_02_S6' : main0,
             'AGBT02A_028_02_S7' : main0,
             'AGBT02A_028_02_S8' : main0,
             'AGBT02A_028_02_S9' : main0,
             'AGBT02A_028_02_S10': main0,
             'AGBT02A_028_02_S11': main0,
             'AGBT02A_028_02_S12': main0,
             'AGBT02A_028_04_S0' : main0,
             'AGBT02A_028_04_S1' : main0,
             'AGBT02A_028_04_S2' : main0,
             'AGBT02A_028_04_S3' : main0,
             'AGBT02A_028_04_S4' : main0,
             'AGBT02A_028_04_S5' : main0,
             'AGBT02A_028_04_S6' : main0,
             'AGBT02A_028_04_S7' : main0,
             'AGBT02A_028_04_S8' : main0,
             'AGBT02A_028_04_S9' : main0,
             'AGBT02A_028_05_L0' : main0,
             'AGBT02A_028_05_L1' : main0,
             'AGBT02A_028_05_L2' : main0,
             'AGBT02A_028_05_L3' : main0,
             'AGBT02A_028_05_L4' : main0,
             'AGBT02A_028_05_L5' : main0,
             'AGBT02A_028_05_L6' : main0,
             'AGBT02A_028_05_L7' : main0,
             'AGBT02A_028_05_L8' : main0,
             'AGBT02A_028_05_L9' : main0,
             'AGBT02A_028_05_L10': main0,
             'AGBT02A_028_05_L11': main0,
             'AGBT02A_028_05_L12': main0,
             'AGBT02A_028_05_L13': main0,
             'AGBT02A_028_05_L14': main0,
             'AGBT02A_028_05_L15': main0,
             'AGBT02A_028_05_L16': main0,
             'AGBT02A_028_05_P0' : main0, # Bad
             'AGBT02A_028_06_P0' : main0, # Not great
             'AGBT02A_028_06_P1' : main0, # Not great
             'AGBT02A_028_06_P2' : main0, # Not great, no detection
             'AGBT02A_028_06_P3' : main0, # Bad
             'AGBT02A_028_06_P4' : main0, # Not great, no detection
             'AGBT02A_028_06_P5' : main0, # Not great, no detection
             }

proc_params = {'AGBT02A_028_01_P0': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{196:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    197:{'degree0':9,
                                                         'degree1':9,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    198:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                         },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{196:{'degree0':11,
                                                         'degree1':11,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    197:{'degree0':9,
                                                         'degree1':9,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    198:{'degree0':11,
                                                         'degree1':11,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                     '2':{'spw':1,
                                          'pol':0,
                                          'n_list':{193:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    194:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    195:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                         },
                                     '3':{'spw':1,
                                          'pol':1,
                                          'n_list':{193:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    194:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    195:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         }
                                     },
               'AGBT02A_028_01_C1': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{113:{'degree0':25,
                                                         'degree1':25,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{113:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                     '2':{'spw':1,
                                          'pol':0,
                                          'n_list':{111:{'degree0':5,
                                                         'degree1':11,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '3':{'spw':1,
                                          'pol':1,
                                          'n_list':{111:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                     },
               'AGBT02A_028_01_L0': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{156:{'degree0':9, 'degree1':9, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':7, 'degree1':7, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_02_S0': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S1': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S2': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{#133:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{#133:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{#133:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{#133:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S3': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S4': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S5': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S6': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{153:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S7': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S8': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S9': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':1, 'pol':0, 'sig':1, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':1, 'pol':1, 'sig':1, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '4':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '5':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{149:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '6':{'spw':1, 'pol':0, 'sig':0, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '7':{'spw':1, 'pol':1, 'sig':0, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_02_S10': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                           'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '1':{'spw':0, 'pol':1, 'sig':1, 
                                           'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '2':{'spw':1, 'pol':0, 'sig':1, 
                                           'n_list':{#148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '3':{'spw':1, 'pol':1, 'sig':1, 
                                           'n_list':{#148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '4':{'spw':0, 'pol':0, 'sig':0, 
                                           'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '5':{'spw':0, 'pol':1, 'sig':0, 
                                           'n_list':{148:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '6':{'spw':1, 'pol':0, 'sig':0, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '7':{'spw':1, 'pol':1, 'sig':0, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      },
               'AGBT02A_028_02_S11': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                           'n_list':{}
                                           },
                                      '1':{'spw':0, 'pol':1, 'sig':1, 
                                           'n_list':{}
                                           },
                                      '2':{'spw':1, 'pol':0, 'sig':1, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '3':{'spw':1, 'pol':1, 'sig':1, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '4':{'spw':0, 'pol':0, 'sig':0, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '5':{'spw':0, 'pol':1, 'sig':0, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '6':{'spw':1, 'pol':0, 'sig':0, 
                                           'n_list':{}
                                           },
                                      '7':{'spw':1, 'pol':1, 'sig':0, 
                                           'n_list':{}
                                           },
                                      },
               'AGBT02A_028_02_S12': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '1':{'spw':0, 'pol':1, 'sig':1, 
                                           'n_list':{147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '2':{'spw':1, 'pol':0, 'sig':1, 
                                           'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '3':{'spw':1, 'pol':1, 'sig':1, 
                                           'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '4':{'spw':0, 'pol':0, 'sig':0, 
                                           'n_list':{}
                                           },
                                      '5':{'spw':0, 'pol':1, 'sig':0, 
                                           'n_list':{}
                                           },
                                      '6':{'spw':1, 'pol':0, 'sig':0, 
                                           'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      '7':{'spw':1, 'pol':1, 'sig':0, 
                                           'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                     }
                                           },
                                      },
               'AGBT02A_028_02_P0': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{274:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    275:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    276:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    277:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    278:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    279:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    280:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    281:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    282:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    283:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    284:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    285:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{274:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    275:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    276:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    277:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    278:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    279:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    280:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    281:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    282:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    283:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    284:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    285:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{273:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    274:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    275:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    276:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    277:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    278:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    279:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    280:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    281:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    282:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    283:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    284:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    285:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{273:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    274:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    275:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    276:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    277:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    278:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    279:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    280:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    281:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    282:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    283:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    284:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    285:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_02_P1': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{263:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    264:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #265:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    266:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #267:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    268:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #269:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.}, Noise spike too close to HRRL
                                                    270:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    271:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    272:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    273:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{263:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    264:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #265:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    266:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #267:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    268:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #269:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    270:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    271:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    272:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    273:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{263:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    264:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #265:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    266:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #267:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    268:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    269:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    270:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    271:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    272:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{263:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    264:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #265:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    266:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #267:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    268:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    269:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    270:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    271:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    272:{'degree0':7, 'degree1':7, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_02_P2': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#254:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #255:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    256:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #257:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    258:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    259:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    260:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    261:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    262:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{#254:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #255:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    256:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #257:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    258:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    259:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    260:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    261:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    262:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#254:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #255:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    256:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #257:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    258:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    259:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    260:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    261:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    262:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{#254:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #255:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    256:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #257:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    258:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    259:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    260:{'degree0':9, 'degree1':9, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    261:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    262:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S0': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S1': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{146:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                    },
               'AGBT02A_028_04_S2': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{#141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{#141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':10., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S3': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #139:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{#138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #139:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    #141:{'degree0':11, 'degree1':11, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{#138:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':8.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S4': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{136:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    137:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    138:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S5': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{133:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{133:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{133:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{133:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    134:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    135:{'degree0':5, 'degree1':5, 'v0':10.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':35.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S6': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S7': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{152:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    153:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S8': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_04_S9': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{138:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{138:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{138:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{138:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    139:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    140:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #141:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    142:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    143:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    144:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    145:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    146:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    147:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    148:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    149:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    150:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    151:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_P0': {'0':{'spw':0,
                                          'pol':0,
                                          'sig':1,
                                          'n_list':{241:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    242:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    243:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    244:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    245:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    246:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    247:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':12.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'sig':1,
                                          'n_list':{241:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     242:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     243:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     244:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     245:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     246:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.},
                                                     247:{'degree0':5,
                                                          'degree1':5,
                                                          'v0':12.0,
                                                          'dv':10.,
                                                          'v0_hrrl':-8.9,
                                                          'dv_hrrl':35.}
                                                   }
                                         },
                                    },
               'AGBT02A_028_05_L0': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{166:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    167:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{166:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    167:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{166:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{166:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L1': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{165:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{165:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{165:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{165:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L2': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{163:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    164:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{163:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    164:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{163:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    164:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{163:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    164:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L3': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#162:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{#162:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#162:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{#162:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L4': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    161:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    161:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    161:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{160:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    161:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L5': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{#159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{#159:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L6': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{158:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L7': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    157:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{156:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L8': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{155:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L9': {'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{154:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L10':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{168:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{168:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{167:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    168:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{167:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    168:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L11':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{169:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    170:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{169:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    170:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{169:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    170:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{169:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    170:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L12':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{171:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    172:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{171:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    172:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{171:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    172:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{171:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    172:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L13':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{173:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    174:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{173:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    174:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{173:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    174:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{173:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    174:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L14':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{175:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    176:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{175:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    176:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{175:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    176:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{175:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    176:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L15':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{177:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    178:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{177:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    178:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{177:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    178:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{177:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    178:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_05_L16':{'0':{'spw':0, 'pol':0, 'sig':1, 
                                          'n_list':{#179:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #180:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '1':{'spw':0, 'pol':1, 'sig':1, 
                                          'n_list':{179:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    180:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '2':{'spw':0, 'pol':0, 'sig':0, 
                                          'n_list':{#179:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    #180:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     '3':{'spw':0, 'pol':1, 'sig':0, 
                                          'n_list':{179:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    180:{'degree0':5, 'degree1':5, 'v0':12.0, 'dv':15., 'v0_hrrl':-8.9, 'dv_hrrl':38.},
                                                    }
                                          },
                                     },
               'AGBT02A_028_06_P2': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{223:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    224:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    225:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    226:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    227:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{223:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    224:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    225:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    226:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    227:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                    },
               'AGBT02A_028_06_P3': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{217:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    218:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    219:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    220:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    221:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{217:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    218:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    219:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    220:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    221:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                    },
               'AGBT02A_028_06_P4': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{213:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    214:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    215:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    216:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{213:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    214:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    215:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    216:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                    },
               'AGBT02A_028_06_P5': {'0':{'spw':0,
                                          'pol':0,
                                          'n_list':{209:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    210:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    211:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    212:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                    }
                                          },
                                     '1':{'spw':0,
                                          'pol':1,
                                          'n_list':{209:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    210:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    211:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.},
                                                    212:{'degree0':5,
                                                         'degree1':5,
                                                         'v0':8.0,
                                                         'dv':10.,
                                                         'v0_hrrl':-8.9,
                                                         'dv_hrrl':35.}
                                                   }
                                         },
                                    },
              }

if __name__ == '__main__':
    
    project = sys.argv[1] #'AGBT02A_028_02_S5'
    band = sys.argv[2]
    subset = sys.argv[3] #'3'
    plot = int(sys.argv[4]) #1
    
    basename = '{0}_{1}'.format(project, band)
    
    processes[basename](basename, 
                        proc_params[basename][subset]['spw'], 
                        proc_params[basename][subset]['pol'],
                        proc_params[basename][subset]['sig'],
                        #np.arange(258,262,1),
                        #[278, 281],
                        np.sort(list(proc_params[basename][subset]['n_list'].keys())),
                        fit_dict=proc_params[basename][subset]['n_list'],
                        plot=plot)