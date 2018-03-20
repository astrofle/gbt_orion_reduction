#!/usr/bin/env python

import re
import sys
import glob
import argparse
import numpy as np
import pylab as plt

def parse_args():
    """
    """
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('specs', type=str, nargs='+',
                        help="Spectra to plot (list).")
    parser.add_argument('plot', type=str,
                        help="Output plot filename (string).")
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    output = args.plot
    infiles = args.specs
    infiles.sort()
    print(infiles)
    
    fig = plt.figure(frameon=False)
    fig.suptitle(r'Orion spectra')
    ax = fig.add_subplot(1, 1, 1, adjustable='datalim')
    
    for i,f in enumerate(infiles):
        
        data = np.ma.masked_invalid(np.loadtxt(f))
        
        qn = re.findall('n\d+', f)[0]
    
        ax.step(data[:,0]*1e-3, data[:,1], ls='-', drawstyle='steps-pre', lw=1, where='pre', label='{0}'.format(qn))
    #ax.step(data1[:,0]*1e-3, data1[:,1], 'b-', drawstyle='steps', lw=1, where='pre', label='LL')
    
    ax.legend(loc=0)
    
    ax.set_xlabel('Velocity (km s$^{-1}$)')
    ax.set_ylabel('$T_{A}$ (arbitrary units)')
    
    ax.minorticks_on()
    
    plt.savefig(output, 
                bbox_inches='tight', 
                pad_inches=0.06)
    plt.close()
