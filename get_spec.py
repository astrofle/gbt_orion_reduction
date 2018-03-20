#!/usr/bin/env python

import sys

import numpy as np
import pylab as plt

from scipy import constants

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from gbt_tools import gbtt
from crrlpy import crrls
from crrlpy import utils

# Flagging tool
sys.path.insert(0, '/home/pedro/Documents/PhD/scripts/lofar-strw/trunk/aoflagger/aoflagger')
import sumthreshold

from datetime import datetime
startTime = datetime.now()

def rfi_flag(frame_data, flagging_threshold, flag_window_lengths, threshold_shrink_power):
    """
    """

    # Convert data to a Z score.
    frame_data = np.ma.asarray(frame_data, dtype=np.float32)
    frame_data -= frame_data.mean()
    frame_data_std = frame_data.std()
    if frame_data_std != 0.0:
        frame_data /= frame_data_std

    # Look for flags in the 2D data.
    flags = sumthreshold.sum_threshold_2d(frame_data,
                                          np.copy(frame_data.mask),
                                          flagging_threshold,
                                          window_lengths=flag_window_lengths,
                                          threshold_shrink_power=threshold_shrink_power)

    return flags

if __name__ == '__main__':
    
    basename = sys.argv[1] #'AGBT02A_028_04'
    band = sys.argv[2] #'S2'
    spw = int(sys.argv[3]) #0
    pol = int(sys.argv[4]) #0
    sig = int(sys.argv[5]) #0
    plot = int(sys.argv[6]) #1
    
    filein = '{0}_{1}.raw.acs.fits'.format(basename, band)
    fileout = '{0}_{1}_orion_spec_spw{2}_pol{3}_sig{4}'.format(basename, band, spw, pol, sig)
    output_spec_dir = 'specs'
    inspection_plot_dir = 'get_spec_plots'
    vrng = 250e3 # m s-1
    #source = '05h35m17.3s,-05d23m28s' # Orion A J2000
    source = '05h35m14.16s,-05d22m21.5s' # Orion KL J2000
    source = '05h35m14.374s,-05d22m29.55s' 
    edges = 0.2
    avgf = 1
    line = 'RRL_CIalpha'
    mean_dopcor = True
    flag_rfi = False
    flag_rfi2 = False
    calibrate = True

    hdu = fits.open(filein)
    
    table = hdu[1].data
    
    data = table['DATA']
    print(data.shape)
    
    # Remove edge channels
    nchan = data.shape[1]
    ch0flg = int(nchan*edges/2.)
    chfflg = int(nchan*(1. - edges/2.))
    data = data[:,ch0flg:chfflg]
    
    cal = table['CAL']
    ifnum = table['IFNUM']
    plnum = table['PLNUM']
    signal = table['SIG']
    
    if_mask = (ifnum == spw)
    pol_mask = (plnum == pol)
    cal_mask = (cal == 'F')
    if sig == 0:
        sig_mask = (signal == 'F')
    else:
        sig_mask = (signal == 'T')
    
    ra = table['CRVAL2']
    dec = table['CRVAL3']
    coo = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
    sou = SkyCoord(source.split(',')[0], source.split(',')[1], frame='fk5')
    
    dist = sou.separation(coo)
    dist_mask = (dist < 0.62*u.deg)
    
    mask = (dist_mask == True) & (if_mask == True) & (cal_mask == True) & (sig_mask == True) & (pol_mask == True)
    mask_on = (dist_mask == True) & (if_mask == True) & (cal_mask != True) & (sig_mask == True) & (pol_mask == True)
    
    spec = data[mask]
    spec_on = data[mask_on]
    
    tsys = table['TSYS'][mask]
    tcal = table['TCAL'][mask]
    
    # Build the frequency axis
    freq = gbtt.get_freq_axis(table[mask], chstart=0, chstop=-1, apply_doppler=False)
    freq = freq[:,ch0flg:chfflg]*1e-6
    #print(doppler)
    print('Velocity frame: {0}'.format(table.field('veldef')[mask][0]))
    
    # Check Doppler correction to LSR frame.
    timestamps = table.field('TIMESTAMP')[mask]
    timestampsstr = np.array(['{0}T{1}'.format('-'.join(t.split('_')[:3]), t.split('_')[-1]) for t in timestamps])
    
    if mean_dopcor:
        dopcor = gbtt.doppler_correct(coo[0].ra.deg, coo[0].dec.deg, timestampsstr[len(timestampsstr)/2]) #
        dopcoridx = -1
    else:
        dopcor = np.array([gbtt.doppler_correct(coo[0].ra.deg, coo[0].dec.deg, t) for t in timestampsstr])
        #dopcoridx = [0:-1,-1]
        dopcoridx = [slice(0, len(dopcor)),-1]
    
    # Check for lines within the spectrum
    freq_lims = np.sort([freq[0,0], freq[0,-1]])
    qns,restfreqs = crrls.find_lines_sb(freq_lims, line)
    print(qns)
    
    for i,f in enumerate(restfreqs):
        
        this_output = '{0}_n{1:.0f}'.format(fileout, qns[i])
        
        f0 = f - crrls.dv2df(f, vrng)
        ff = f + crrls.dv2df(f, vrng)
        
        ch0rrl = utils.best_match_indx(f0, freq[0])
        chfrrl = utils.best_match_indx(ff, freq[0])
        ch0rrl,chfrrl = np.sort([ch0rrl,chfrrl])
        
        spec_rrl = np.ma.masked_invalid(spec[:,ch0rrl:chfrrl])
        spec_cal = np.ma.masked_invalid(spec_on[:,ch0rrl:chfrrl])
        freq_rrl = np.ma.masked_invalid(freq[:,ch0rrl:chfrrl])
        print('Extracted spectrum has a shape of: {0}'.format(spec_rrl.shape))
        
        if plot:
            plt.imshow(spec_rrl, origin='lower', aspect='auto')
            plt.colorbar()
            plt.xlabel('Channel')
            plt.ylabel('Time')
            plt.savefig('{0}/{1}_rrl_raw.pdf'.format(inspection_plot_dir, this_output))
            plt.close()
            
            plt.imshow(spec_cal, origin='lower', aspect='auto')
            plt.colorbar()
            plt.xlabel('Channel')
            plt.ylabel('Time')
            plt.savefig('{0}/{1}_cal_raw.pdf'.format(inspection_plot_dir, this_output))
            plt.close()
            
        # Flag RFI.
        if flag_rfi:
            length_of_max_dimension = max(spec_rrl.shape)
            flagging_threshold = 5.0
            threshold_shrink_power = 0.35
            flag_window_lengths =  2**np.arange(int(np.ceil(np.log(length_of_max_dimension)/np.log(2.))))
            
            flags_rrl = rfi_flag(spec_rrl, flagging_threshold, flag_window_lengths, threshold_shrink_power)
            flags_cal = rfi_flag(spec_cal, flagging_threshold, flag_window_lengths, threshold_shrink_power)
        
            spec_rrl_flg = np.ma.masked_array(spec_rrl, mask=flags_rrl)
            spec_cal_flg = np.ma.masked_array(spec_cal, mask=flags_cal)
        
            if plot:
                plt.imshow(spec_rrl_flg, origin='lower', aspect='auto')
                plt.colorbar()
                plt.xlabel('Channel')
                plt.ylabel('Time')
                plt.savefig('{0}/{1}_rrl_flagged.pdf'.format(inspection_plot_dir, this_output))
                plt.close()
                
                plt.imshow(spec_cal_flg, origin='lower', aspect='auto')
                plt.colorbar()
                plt.xlabel('Channel')
                plt.ylabel('Time')
                plt.savefig('{0}/{1}_cal_flagged.pdf'.format(inspection_plot_dir, this_output))
                plt.close()
        
        else:
            spec_rrl_flg = np.ma.masked_invalid(spec_rrl)
            spec_cal_flg = np.ma.masked_invalid(spec_cal)
        
        # Calibrate.
        spec_rrl_cal = spec_rrl_flg
        
        if calibrate:
            
            tcal_arr = np.array([tcal]*spec_rrl.shape[1]).T
            gtcal = (spec_cal_flg - spec_rrl_flg)
            #gtcal = np.ma.masked_less(gtcal, 0)
            for j in range(len(gtcal)):
                if np.sum(gtcal[j].mask) <= 0.3*len(gtcal[j].mask):
                    gtcal_poly_fit = np.polyfit(np.ma.masked_where(gtcal[j].mask, freq_rrl[j]).compressed(), gtcal[j].compressed(), 0)
                    gtcal_poly = np.poly1d(gtcal_poly_fit)
                    gtcal_poly_vals = gtcal_poly(freq_rrl[j].data)
                
                    spec_rrl_flg[j] = np.ma.masked_where(gtcal[j].mask, spec_rrl_flg[j])
                    spec_rrl_cal[j] = spec_rrl_flg[j]*tcal_arr[j]/gtcal_poly_vals
                else:
                    spec_rrl_flg[j] = np.ma.masked_where([True]*len(spec_rrl_flg[j]), spec_rrl_flg[j])
                    spec_rrl_cal[j] = np.ma.masked_where([True]*len(spec_rrl_flg[j]), spec_rrl_flg[j])
            
            if plot:
                plt.imshow(spec_rrl_cal, origin='lower', aspect='auto')
                plt.colorbar()
                plt.xlabel('Channel')
                plt.ylabel('Time')
                plt.savefig('{0}/{1}_rrl_calibrated.pdf'.format(inspection_plot_dir, this_output))
                plt.close()
                
        # Flag again.
        if flag_rfi2:
            
            flags_rrl = rfi_flag(spec_rrl_cal, flagging_threshold, flag_window_lengths, threshold_shrink_power)
            spec_rrl_cal_flg = np.ma.masked_array(spec_rrl_cal, mask=flags_rrl)
            
            if plot:
                plt.imshow(spec_rrl_cal_flg, origin='lower', aspect='auto')
                plt.colorbar()
                plt.xlabel('Channel')
                plt.ylabel('Time')
                plt.savefig('{0}/{1}_rrl_calibrated_flagged.pdf'.format(inspection_plot_dir, this_output))
                plt.close()
        
        else:
            spec_rrl_cal_flg = np.ma.masked_invalid(spec_rrl_cal)
            
        spec_rrl_avg = np.ma.average(spec_rrl_cal_flg, axis=0, weights=np.power(tsys, -2.))
        spec_rrl_avg.fill_value = np.nan
        
        vel_rrl = crrls.freq2vel(f, freq[0,ch0rrl:chfrrl]) - np.mean(dopcor[dopcoridx])*1e3 # m s-1
        print('Mean Doppler correction: {0} km s-1'.format(np.mean(dopcor[dopcoridx])))
        print('Observed velocity resolution: {0}'.format(abs(vel_rrl[0] - vel_rrl[1])))
        
        ## Average in velocity
        #averaging_values = np.array(list(gbtt.factors(len(vel_rrl))))
        #averaging_factor = averaging_values[np.argmin(abs(averaging_values - avgf))]
        #print('Averaging by a factor of {0} in frequency'.format(averaging_factor))
        #spec_rrl_avg2 = spec_rrl_avg.reshape((spec_rrl_avg.shape[0]//averaging_factor),+(averaging_factor)).mean(axis=1)
        #vel_rrl_avg = vel_rrl.reshape((vel_rrl.shape[0]//averaging_factor),+(averaging_factor)).mean(axis=1)
        
        #print('Output velocity resolution: {0}'.format(abs(vel_rrl_avg[0] - vel_rrl_avg[1])))
        
        spec_out = '{0}/{1}.ascii'.format(output_spec_dir, this_output)
        print('Saving spectrum to: {0}'.format(spec_out))
        np.savetxt(spec_out, np.c_[vel_rrl, spec_rrl_avg.filled()])

print('Script run time: {0}'.format(datetime.now() - startTime))