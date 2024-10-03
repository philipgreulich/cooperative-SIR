# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import sys
import seaborn as sns
import matplotlib as mpl
from scipy import integrate as itgr
from scipy import special as sp

### Read data ########################################

######### Parameters for Figure display ##############
    
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markeredgewidth'] = 3
mpl.rcParams['lines.markersize'] = 13
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['errorbar.capsize'] = 2
mpl.rcParams['lines.color'] = 'k'
mpl.rcParams.update({'figure.autolayout': True})
    

df = pd.read_csv("Fig 2 A-E.csv") ## Load data
    
################ Extract and prepare data #################################
##### channel on = single platelets per droplet, channel off = platelet collectives per droplet ####

hepes=df[(df['Conc']=='HEPES')]       # all induced clusters at t=-4
cvx_on_00001=df[np.logical_and(df['Conc']=='CVX 0.01',df['Chip']=='ON')]
cvx_on_0001=df[np.logical_and(df['Conc']=='CVX 0.1',df['Chip']=='ON')]
cvx_on_0003=df[np.logical_and(df['Conc']=='CVX 0.3',df['Chip']=='ON')]
cvx_on_001=df[np.logical_and(df['Conc']=='CVX 1',df['Chip']=='ON')]
cvx_on_003=df[np.logical_and(df['Conc']=='CVX 3',df['Chip']=='ON')]
cvx_on_010=df[np.logical_and(df['Conc']=='CVX 10',df['Chip']=='ON')]
cvx_on_100=df[np.logical_and(df['Conc']=='CVX 100',df['Chip']=='ON')] 

cvx_off_00001=df[np.logical_and(df['Conc']=='CVX 0.01',df['Chip']=='OFF')]
cvx_off_0001=df[np.logical_and(df['Conc']=='CVX 0.1',df['Chip']=='OFF')] 
cvx_off_0003=df[np.logical_and(df['Conc']=='CVX 0.3',df['Chip']=='OFF')] 
cvx_off_001=df[np.logical_and(df['Conc']=='CVX 1',df['Chip']=='OFF')]
cvx_off_003=df[np.logical_and(df['Conc']=='CVX 3',df['Chip']=='OFF')]  
cvx_off_010=df[np.logical_and(df['Conc']=='CVX 10',df['Chip']=='OFF')]
cvx_off_100=df[np.logical_and(df['Conc']=='CVX 100',df['Chip']=='OFF')] 


str = "FL4-H"


### remove outliers ######################3
cutoff = 10000
cvx_on_00001_noutl = cvx_on_00001[str][np.logical_and(cvx_on_00001[str] < cutoff,cvx_on_00001[str] > 100)]
cvx_on_0001_noutl = cvx_on_0001[str][np.logical_and(cvx_on_0001[str] < cutoff,cvx_on_0001[str] > 100)]
cvx_on_0003_noutl = cvx_on_0003[str][np.logical_and(cvx_on_0003[str] < cutoff,cvx_on_0003[str] > 100)]
cvx_on_001_noutl = cvx_on_001[str][np.logical_and(cvx_on_001[str] < cutoff,cvx_on_001[str] > 100)]
cvx_on_003_noutl = cvx_on_003[str][np.logical_and(cvx_on_003[str] < cutoff,cvx_on_003[str] > 100)]
cvx_on_010_noutl = cvx_on_010[str][np.logical_and(cvx_on_010[str] < cutoff,cvx_on_010[str] > 100)]
cvx_on_100_noutl = cvx_on_100[str][np.logical_and(cvx_on_100[str] < cutoff,cvx_on_100[str] > 100)]


############### Set columns to equal length #######################3
max_len = max(len(cvx_on_00001_noutl),len(cvx_on_0001_noutl),len(cvx_on_0003_noutl),len(cvx_on_001_noutl),len(cvx_on_003_noutl),len(cvx_on_010_noutl),len(cvx_on_100_noutl))
min_len = min(len(cvx_on_00001_noutl),len(cvx_on_0001_noutl),len(cvx_on_0003_noutl),len(cvx_on_001_noutl),len(cvx_on_003_noutl),len(cvx_on_010_noutl),len(cvx_on_100_noutl))


cvx_on_00001_drop = cvx_on_00001_noutl[:min_len]
cvx_on_0001_drop  = cvx_on_0001_noutl[:min_len]
cvx_on_0003_drop  = cvx_on_0003_noutl[:min_len]
cvx_on_001_drop  = cvx_on_001_noutl[:min_len]
cvx_on_003_drop  = cvx_on_003_noutl[:min_len]
cvx_on_010_drop  = cvx_on_010_noutl[:min_len]
cvx_on_100_drop  = cvx_on_100_noutl[:min_len]

np.savetxt("cvx_on_drop",cvx_on_00001_drop)

########### Generate data array for plot for channel on  #################
channel_on_arr = np.array([cvx_on_00001_drop,cvx_on_0001_drop,cvx_on_0003_drop,cvx_on_001_drop,cvx_on_003_drop,cvx_on_010_drop,cvx_on_100_drop])
channel_on_df = pd.DataFrame(channel_on_arr)


############# viooin plot for whole distribution of droplet's fluorescence, for channel on ########### 
plt.yscale('log')
sns.violinplot(data=channel_on_df.T)
plt.show()

cvx_off_00001_noutl = cvx_off_00001[str][np.logical_and(cvx_off_00001[str] < cutoff,cvx_off_00001[str] > 100)]
cvx_off_0001_noutl = cvx_off_0001[str][np.logical_and(cvx_off_0001[str] < cutoff,cvx_off_0001[str] > 100)]
cvx_off_0003_noutl = cvx_off_0003[str][np.logical_and(cvx_off_0003[str] < cutoff,cvx_off_0003[str] > 100)]
cvx_off_001_noutl = cvx_off_001[str][np.logical_and(cvx_off_001[str] < cutoff,cvx_off_001[str] > 100)]
cvx_off_003_noutl = cvx_off_003[str][np.logical_and(cvx_off_003[str] < cutoff,cvx_off_003[str] > 100)]
cvx_off_010_noutl = cvx_off_010[str][np.logical_and(cvx_off_010[str] < cutoff,cvx_off_010[str] > 100)]
cvx_off_100_noutl = cvx_off_100[str][np.logical_and(cvx_off_100[str] < cutoff,cvx_off_100[str] > 100)]

max_len = max(len(cvx_off_00001_noutl),len(cvx_off_0001_noutl),len(cvx_off_0003_noutl),len(cvx_off_001_noutl),len(cvx_off_003_noutl),len(cvx_off_010_noutl),len(cvx_off_100_noutl))
min_len = min(len(cvx_off_00001_noutl),len(cvx_off_0001_noutl),len(cvx_off_0003_noutl),len(cvx_off_001_noutl),len(cvx_off_003_noutl),len(cvx_off_010_noutl),len(cvx_off_100_noutl))


cvx_off_00001_drop = cvx_off_00001_noutl[:min_len]
cvx_off_0001_drop  = cvx_off_0001_noutl[:min_len]
cvx_off_0003_drop  = cvx_off_0003_noutl[:min_len]
cvx_off_001_drop  = cvx_off_001_noutl[:min_len]
cvx_off_003_drop  = cvx_off_003_noutl[:min_len]
cvx_off_010_drop  = cvx_off_010_noutl[:min_len]
cvx_off_100_drop  = cvx_off_100_noutl[:min_len]

########### Generate data array for plot for channel off #################
channel_off_arr = np.array([cvx_off_00001_drop,cvx_off_0001_drop,cvx_off_0003_drop,cvx_off_001_drop,cvx_off_003_drop,cvx_off_010_drop,cvx_off_100_drop])
channel_off_df = pd.DataFrame(channel_off_arr)


############# viooin plot for whole distribution of droplet's fluorescence, for channel off ########### 
plt.yscale('log')
sns.violinplot(data=channel_off_df.T)
plt.show()



############# Get average fluorescence levels ########################
lengths_on = [len(cvx_on_00001[str]),len(cvx_on_0001[str]),len(cvx_on_0003[str]),len(cvx_on_001[str]),len(cvx_on_003[str]),len(cvx_on_010[str]),len(cvx_on_100[str])]
lengths_off = [len(cvx_off_00001[str]),len(cvx_off_0001[str]),len(cvx_off_0003[str]),len(cvx_off_001[str]),len(cvx_off_003[str]),len(cvx_off_010[str]),len(cvx_off_100[str])]

channel_on_av = [np.average(cvx_on_00001[str]),np.average(cvx_on_0001[str]),np.average(cvx_on_0003[str]),np.average(cvx_on_001[str]),np.average(cvx_on_003[str]),np.average(cvx_on_010[str]),np.average(cvx_on_100[str])]
channel_off_av = [np.average(cvx_off_00001[str]),np.average(cvx_off_0001[str]),np.average(cvx_off_0003[str]),np.average(cvx_off_001[str]),np.average(cvx_off_003[str]),np.average(cvx_off_010[str]),np.average(cvx_off_100[str])]

channel_on_std = [np.std(cvx_on_00001[str]),np.std(cvx_on_0001[str]),np.std(cvx_on_0003[str]),np.std(cvx_on_001[str]),np.std(cvx_on_003[str]),np.std(cvx_on_010[str]),np.std(cvx_on_100[str])]
channel_off_std = [np.std(cvx_off_00001[str]),np.std(cvx_off_0001[str]),np.std(cvx_off_0003[str]),np.std(cvx_off_001[str]),np.std(cvx_off_003[str]),np.std(cvx_off_010[str]),np.std(cvx_off_100[str])]


############# Set up plot 1 (channel on = single platelets per droplet) ########################
x = [0.01,0.1,0.3,1,3,10,100]

plt.xscale('log')
plt.yscale('log')

plt.ylim(100, 10000)

cmean = 4.2
clogstd = 0.6
c0=0.3

plt.errorbar(x, channel_on_av,yerr=channel_on_std/np.sqrt(lengths_on),fmt='k+') ### Plot single-platelet fluorescence level for each convulxin concentration (*not rescaled*)
plt.show()

plt.xscale('log')
plt.yscale('log')

plt.ylim(100, 10000)

plt.show()


plt.xscale('log')
#plt.yscale('log')

plt.ylim(0, 1)

channel_on_av_norm = (channel_on_av - channel_on_av[0])/(max(channel_on_av) - channel_on_av[0])
channel_off_av_norm = (channel_off_av - channel_off_av[0])/(max(channel_off_av) - channel_off_av[0])

channel_on_std_norm = channel_on_std/(max(channel_on_av) - channel_on_av[0])
channel_off_std_norm = channel_off_std/(max(channel_off_av) - channel_off_av[0])

plt.xlabel('Stimulant concentration [ng/mL]')
plt.ylabel('Activated proportion (CDF)')

plt.gca().set_aspect(1)

plt.plot(x, channel_on_av_norm,'k+') ## Plot single-platelet fluorescence level for each convulxin concentration (*data*, *rescaled* and background fluorescence removed)
plt.plot(x, 0.5*(1+sp.erf((np.log(x)-np.log(cmean))/(np.sqrt(2)*clogstd))),"b-") ### Plot fit for above (log-normal distribtution)
plt.savefig("CDF_exp")


###################################################################### 
################## Set up ODEs for model prediction ##################
######################################################################
  

nmax_d = 25  # maximal number of cooperative platelets ("m" in main text) for fitting log-normal distribution
nmax = 15  # maximal number of cooperative platelets ("m" in main text) for ODE solutions

r = 25000   # refault value of r (reproductive number)
y = np.zeros(nmax+1)
tend = 1.0

nmean = 14  # mean value of distribution of m
nstd = 2   # standard deviation of distribution of m
c0=0.3   # concentration generated by single platelet


###### generate ODEs ########################
def dfdt(t,y,m):
    dydt = np.zeros(nmax+1);
    for j in range(1,nmax+1):
        jeff = max(0,j-m)
        dydt[j] = - r * y[j] * y[0]**jeff
        
    dydt[0] = -np.sum(dydt) - y[0]
    return dydt


###### generate log-normal distribution with discrete probabilities ########################
def lognorm_int(n,nmean,nstd,nmax_d):
    lognorm = np.zeros(nmax_d+1)
    for m in range(1,nmax_d+1):
        lognorm[m] = 1/(2*np.pi*nstd**2)*np.exp(-(np.log(m) - np.log(nmean))**2/(2*nstd**2))
        
    norm = 1/np.sum(lognorm)
    return norm*lognorm[n]


###### Prepare plot ########################
plt.xlim(0.05,3.5)
plt.ylim(0.0,105)
# plt.xscale('log')

x_r = np.array([0.1])
x_r = np.append(x_r,np.arange(c0,(nmax)*c0,c0))

plt.xlabel('Stimulant concentration [ng/mL]')
plt.ylabel('Relative activation [%]')

# ax = plt.axes()
# ax.set_box_aspect(1)

plt.errorbar(x, 100*channel_off_av_norm,yerr=channel_off_std_norm/np.sqrt(lengths_off),fmt='k+')  ###### Plot data of platelet collectives (channel off)

formats=['k-','r-','y-','g-','c-','b-','m-']
ctr=0

############## Plot model predictions for varying values of r ######################33
for r in [1000,3000,10000,30000,100000]:
    y0 = np.zeros(nmax+1)

    ################## for m=0 ############################
    m0=0
    y0[0]=0 
    for n in range(1,nmax+1):
        y0[n] = lognorm_int(n,nmean,nstd,nmax_d)   ## draw initial conditions from fitted ditribution of platelet sub-populations per m
        
    s0tot_before = np.sum(y0)
    y0[0]=0.00001
    y_t = itgr.solve_ivp(dfdt, (0,tend), y0,args=(m0,))  ## Solve ODE

    stot_t = np.sum(y_t.y[1:],axis=0)
    s0totvec = np.full(len(stot_t),s0tot_before)
    r_t = s0totvec - stot_t
    r_t_over = s0totvec - stot_t - y[0]
    # plt.plot(y_t.t,r_t)
    # plt.plot(y_t.t,r_t_over)
    stot = np.array([stot_t[len(stot_t)-1]])
    y0_m = np.array([y[0]])
    r_m = np.array([s0tot_before - stot[0]])


    ################## for m>0 ############################
    for m in range(1,nmax+1):
        y0[0]=0
        for n in range(1,nmax+1):
            y0[n] = lognorm_int(n,nmean,nstd,nmax_d)
            
        s0tot_before = np.sum(y0)
        y0[0] = sum(y0[1:m+1])
        for n in range(1,m+1):
            y0[n] = 0
        print("y0 for m=",m,"is",y0,"\n")
        
        y_t = itgr.solve_ivp(dfdt, (0,tend), y0,args=(m,))
        
        stot_t = np.sum(y_t.y[1:],axis=0)
        s0totvec = np.full(len(stot_t),s0tot_before)
        r_t = s0totvec - stot_t
        r_t_over = s0totvec - stot_t - y[0]
        # plt.plot(y_t.t,r_t)
        # plt.plot(y_t.t,r_t_over)
        stot_curr = stot_t[len(stot_t)-1]
        y0_m_curr = y[0]
        r_m_curr = s0tot_before - stot_curr
        r_m = np.append(r_m,r_m_curr) 
        stot = np.append(stot,stot_curr)
        y0_m = np.append(y0_m,y0_m_curr)
        


    # plt.xscale('log')
    # plt.yscale('log')


    plt.plot(x_r,100*r_m/r_m[len(r_m)-1],formats[ctr])   ###### Plot model predictions of platelet collectives (channel off)
    ctr=ctr+1
    #plt.savefig("microfluidics_vs_model")

    # plt.xlim(0.05,5.0)
    # plt.ylim(0.0,1.1)
    # # plt.xscale('log')

    # plt.plot(x_r,stot,marker='o')
    # plt.plot(x_r,s0tot_before - stot - y0_m,marker='x')

    # plt.plot(y_t.t,y_t.y[0])
    # plt.plot(y_t.t,y_t.y[1])

plt.savefig("microfluidics_vs_model")

x=[1,2,3,4,5]
n1=0.01
n2=0.4
y=[n1,n2,1-2*n2-2*n1,n2,n1]

plt.xlabel(r'$m$')
plt.ylabel(r'$n_m^{(0)}$')

plt.bar(x, y)
plt.savefig("distributions_n_c_04")










