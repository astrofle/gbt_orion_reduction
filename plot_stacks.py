#!/usr/bin/env python

import glob

import numpy as np
import pylab as plt

from crrlpy import utils

import matplotlib
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

if __name__ == '__main__':
    
    off_y = 0.2
    
    stacks = np.array(glob.glob('stack_results/*.ascii'))
    #utils.natural_sort(stacks)
    
    qns = np.array([int(stack.split('_')[-1].split('.')[0][1:-1]) for stack in stacks])
    p = np.argsort(qns)
    qns = qns[p][:-1]
    stacks = stacks[p][:-1]
    print(qns, stacks)
    
    fig = plt.figure(frameon=False, figsize=(3,3))
    fig.suptitle(r'Orion GBT spectra of CRRLs')
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    
    for i,stack in enumerate(stacks):
        
        data = np.ma.masked_invalid(np.loadtxt(stack))
        
        ax.plot(data[:,0], data[:,1] + (len(qns) - i)*off_y, drawstyle='steps-pre', label=r'$C{0}\alpha$'.format(qns[i]), lw=0.6)
        
    ax.legend(loc=0)    
    
    ax.minorticks_on()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    
    ax.set_xlabel(r'$v_{\rm{lsr}}$ (km s$^{-1}$)')
    ax.set_ylabel(r'$T_{\rm{A}}$ (K)')
    
    #plt.tight_layout()
    
    plt.savefig('stacks.pdf',
                bbox_inches='tight', 
                pad_inches=0.06)
    plt.close(fig)
    plt.close()