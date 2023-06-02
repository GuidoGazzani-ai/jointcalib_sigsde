"""
Created on Sat Jan 28 18:08:38 2023

@author: Guido Gazzani & Janka Moeller
"""

import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm

import numpy as np

print('Torch version',torch.__version__)
print('Numpy version',np.__version__)
print('pandas version:',pd.__version__)
print('Joblib version:',pd.__version__)


# In[2]:

loss_flag="SOFT_INDICATOR_VEGA_DELTA"
 #"LINEAR_VEGA_DELTA" #"LP" # "LINEAR" "SOFT_INDICATOR" #

power=2
user="Guido" 
config="config9"
vix_only=False

lambda_coeff=0.35

maturities_vix=[r'14.0days',r'28.0days',r'49.0days',r'77.0days',r'105.0days',r'140.0days',r'259.0days']
maturities_spx=[r'14.0days',r'44.0days',r'58.0days',r'79.0days',r'107.0days',r'135.0days',r'170.0days',r'289.0days']

maturities_joint=[r'14.0days',r'28.0days',r'44.0days',r'49.0days',r'58.0days',r'77.0days',r'79.0days',r'105.0days',r'107.0days',r'135.0days',r'140.0days',r'170.0days',r'259.0days',r'289.0days']


index_sel_maturities_vix=[0,1,2]
index_sel_maturities_spx=[0,2,4,6]


day=r'/20210602'

maturities_name= "Mat_SPX"+str(index_sel_maturities_spx)+"_VIX"+str(index_sel_maturities_vix)





if vix_only:
     moneyness_upperdev_vix=[1.5,1.5,2,2,3,3,3]
     moneyness_lowerdev_vix=[0.1,0.1,0.2,0.2,0.2,0.2,0.2]
     moneyness_upperdev_spx=[0.035,0.05,0.2,0.25,0.3,0.35,0.45,0.5]
     moneyness_lowerdev_spx=[0.05,0.3,0.2,0.25,0.3,0.4,0.5,0.5]

else:

     moneyness_upperdev_vix=[1.2,1.2,2.3,3,3,3.2,3.5,3.8]
     moneyness_lowerdev_vix=[0.1,0.1,0.2,0.2,0.2,0.2,0.2]
     
     moneyness_upperdev_spx=[0.05,0.05,0.2,0.25,0.3,0.35,0.45,0.5]
     moneyness_lowerdev_spx=[0.08,0.3,0.2,0.25,0.3,0.4,0.5,0.5]
     


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_ivol(price, spot, strike, T, r, d):

    def BS_price(sigma):
        d_1= 1/(sigma*np.sqrt(T))*(np.log(spot/strike) + (r-d+sigma**2/2)*T)
        d_2= d_1-sigma*np.sqrt(T)
        
        N_1= norm.cdf(d_1) #scipy.stats.norm.cdf
        N_2= norm.cdf(d_2) #scipy.stats.norm.cdf
        
        #print(N_1, flush=True)
        return N_1*spot*np.exp(-d*T)-N_2*strike*np.exp(-r*T) - price
    
    root = fsolve(BS_price, 1)[-1] #scipy.optimize.fsolve
    
    if np.isclose(BS_price(root), 0.0):
        return root
    
    else:
        return -99.99




if vix_only:
    save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/vix_only/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)
else:
    save_init_dir= r'/scratch.global/ag_cu/Codes_'+user+'/calibrated_parameters/ALL_JOINT_same_ell/'+maturities_name+r'/'+config+r"/"+loss_flag+str(power)+'_lambda='+str(lambda_coeff)



os.chdir(save_init_dir)
print(os.getcwd())

vix_futures= np.load('optimal_VIX_futures.npy')


list_strikes_vix=[]
save_calib_iv_vix=[]    
save_iv_bid_vix=[]   
save_iv_ask_vix=[]   


for i, idx in enumerate(index_sel_maturities_vix):
    os.chdir(save_init_dir)
    model_prices_vix= np.load('prices_optimal_VIX_'+str(i)+'.npy')

    maturity_vix=maturities_vix[idx]
    os.chdir(r'/scratch.global/ag_cu/Data/VIX/Processed Data'+day)
    
    df_vix=pd.read_csv('ivol_data_maturity_'+maturity_vix)
    df_vix['strike']=df_vix['strike']/1000
    df_vix['open_interest']=df_vix['open_interest']/100 #not useful?
    scaling_flag_and_cut=True
    spot=df_vix['spot'][0]
    
    strikes_vix=np.array(df_vix['strike'])
    prices_vix=np.array(df_vix['mid'])
    bid_vix=np.array(df_vix['best_bid'])
    ask_vix=np.array(df_vix['best_offer'])
    rate_vix=df_vix['rate'][0]/100
    divi_vix=df_vix['divi'][0]/100
    

        
    idx_lowest_moneyness=find_nearest(strikes_vix,(1-moneyness_lowerdev_vix[idx])*spot)
    idx_highest_moneyness=find_nearest(strikes_vix,(1+moneyness_upperdev_vix[idx])*spot)
    
    strikes_vix=strikes_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    prices_vix=prices_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    bid_vix=bid_vix[idx_lowest_moneyness:idx_highest_moneyness+1]
    ask_vix=ask_vix[idx_lowest_moneyness:idx_highest_moneyness+1]    
        
    list_strikes_vix.append(strikes_vix)
       
        
    maturity_vix=int(maturity_vix.split('.')[0])/365.25
    

    
    iv_per_mat_VIX=[find_ivol(prices_vix[k], np.exp(-rate_vix*maturity_vix)*spot, strike, maturity_vix, rate_vix, divi_vix) for k,strike in enumerate(strikes_vix)]
        
    bid_per_mat_VIX=[find_ivol(bid_vix[k], np.exp(-rate_vix*maturity_vix)*spot, strike, maturity_vix, rate_vix, divi_vix) for k,strike in enumerate(strikes_vix)]
    ask_per_mat_VIX=[find_ivol(ask_vix[k], np.exp(-rate_vix*maturity_vix)*spot, strike, maturity_vix, rate_vix, divi_vix) for k,strike in enumerate(strikes_vix)]
    
   
    
    iv_calibrated_VIX=[find_ivol(model_prices_vix[k], np.exp(-rate_vix*maturity_vix)*vix_futures[i], strike, maturity_vix, rate_vix, divi_vix) for k,strike in enumerate(strikes_vix)]
    

    os.chdir(save_init_dir)
    plt.figure()
    plt.plot(strikes_vix, iv_calibrated_VIX, label="calibrated",color='blue', linestyle='None',marker='o',alpha=0.5, markersize = 5.0)
    plt.plot(strikes_vix, bid_per_mat_VIX,  label="bid",color='red', linestyle='None',marker='+')
    plt.plot(strikes_vix, ask_per_mat_VIX, label="ask",color='red', linestyle='None',marker='+')

    plt.axvline(x=spot, ls="-.", label="market future", color="red")
    plt.axvline(x=vix_futures[i], ls="--", label="model future", color="blue")
    
    plt.legend()
    plt.title("IVOL VIX_"+str(i))
    plt.savefig("calibrated_ivol_VIX_"+str(i))
    plt.show()
    
    
    
    save_calib_iv_vix.append(iv_calibrated_VIX)
    save_iv_bid_vix.append(bid_per_mat_VIX) 
    save_iv_ask_vix.append(ask_per_mat_VIX)   
    
    print('Absolute relative error futures:', np.abs((spot-vix_futures[i])/spot))
    



np.save('arr_strikes_vix.npy',np.array(list_strikes_vix))
np.save('arr_calib_iv_vix.npy',np.array(save_calib_iv_vix))
np.save('arr_iv_bid_vix.npy',np.array(save_iv_bid_vix))
np.save('arr_iv_ask_vix.npy',np.array(save_iv_ask_vix))


    
save_arr_strike_spx=[]    
save_calib_iv_spx=[]    
save_iv_bid_spx=[]   
save_iv_ask_spx=[]   
  
for i, idx in enumerate(index_sel_maturities_spx):
    os.chdir(save_init_dir)
    model_prices_spx= np.load('prices_optimal_SPX_'+str(i)+'.npy')
    
    maturity_spx=maturities_joint[idx]

    nbr_strikes_spx=[21,21,21,21,21,21,21,21]
        
    
    
    os.chdir(r'/scratch.global/ag_cu/Data/SPX/Processed Data'+day)
    
    df_spx=pd.read_csv('ivol_data_maturity_'+maturity_spx)
    df_spx['strike']=df_spx['strike']/1000
    df_spx['open_interest']=df_spx['open_interest']/100
    spot=df_spx['spot'][0]
    
    strikes_spx=np.array(df_spx['strike'])
    prices_spx=np.array(df_spx['mid'])
    bid_spx=np.array(df_spx['best_bid'])
    ask_spx=np.array(df_spx['best_offer'])
    rate_spx=df_spx['rate'][0]/100
    divi_spx=df_spx['divi'][0]/100
    
    #maturity_spx=44/365.25
    maturity_spx=int(maturity_spx.split('.')[0])/365.25
    
    strikes_spx=(strikes_spx/spot)
    prices_spx=(prices_spx/spot)
    bid_spx=(bid_spx/spot)
    ask_spx=(ask_spx/spot)
    
    nbr_stk=len(strikes_spx)
    
    idx_lowest_moneyness=find_nearest(strikes_spx,1-moneyness_lowerdev_spx[i])
    idx_highest_moneyness=find_nearest(strikes_spx,1+moneyness_upperdev_spx[i])
    
    strikes_spx=strikes_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
    prices_spx=prices_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
    bid_spx=bid_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
    ask_spx=ask_spx[idx_lowest_moneyness:idx_highest_moneyness+1]
    
    integers=np.array([int(b) for b in np.linspace(0,len(strikes_spx)-1,nbr_strikes_spx[i])])
    strikes_spx=strikes_spx[integers]    
    prices_spx=prices_spx[integers]
    bid_spx=bid_spx[integers]
    ask_spx=ask_spx[integers]
    
    iv_per_mat=[find_ivol(prices_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
        
    bid_per_mat=[find_ivol(bid_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
    ask_per_mat=[find_ivol(ask_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]
    
    
    iv_calibrated=[find_ivol(model_prices_spx[k], 1, strike, maturity_spx, rate_spx, divi_spx) for k,strike in enumerate(strikes_spx)]

    os.chdir(save_init_dir)
    plt.figure()
    plt.plot(strikes_spx, iv_calibrated, label="calibrated",color='blue', linestyle='None',marker='o',alpha=0.5, markersize = 5.0)
    plt.plot(strikes_spx, bid_per_mat, label="bid",color='red', linestyle='None',marker='+')
    plt.plot(strikes_spx, ask_per_mat, label="ask",color='red', linestyle='None',marker='+')
   
    plt.legend()
    plt.title("IVOL SPX_"+str(i))
    
    plt.savefig("calibrated_ivol_SPX_"+str(i))
    
    plt.show()
    
    save_arr_strike_spx.append(strikes_spx)
    save_calib_iv_spx.append(iv_calibrated)
    save_iv_bid_spx.append(bid_per_mat) 
    save_iv_ask_spx.append(ask_per_mat)   
    
    plt.figure()
    plt.plot(strikes_spx,model_prices_spx, label="calibrated",color='blue', linestyle='None',marker='o',alpha=0.5, markersize = 5.0)
    plt.plot(strikes_spx, bid_spx, label="bid",color='red', linestyle='None',marker='+')
    plt.plot(strikes_spx, ask_spx, label="ask",color='red', linestyle='None',marker='+')
    
    plt.legend()
    plt.title("Prices SPX_"+str(i))
    plt.savefig("calibrated_prices_SPX_bidask_"+str(i))
    plt.show()

    
np.save('arr_strikes_spx.npy',np.array(save_arr_strike_spx))
np.save('arr_calib_iv_spx.npy',np.array(save_calib_iv_spx))
np.save('arr_iv_bid_spx.npy',np.array(save_iv_bid_spx))
np.save('arr_iv_ask_spx.npy',np.array(save_iv_ask_spx))