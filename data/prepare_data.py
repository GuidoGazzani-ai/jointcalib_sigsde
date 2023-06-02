# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:05:11 2023

@author: Janka Moeller and Guido Gazzani
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import fsolve


parser = argparse.ArgumentParser()

parser.add_argument('-I','--index', type=str,  help='index: choose between VIX or SPX')
parser.add_argument('-D','--date', type=int, help='date in format YYYYMMDD, make sure its a trading day')
args = parser.parse_args()

spot_date=args.date

index=args.index  # "VIX" or "SPX"

cluster_flag=False

if index=="VIX":
    if cluster_flag==True:
        os.chdir('/scratch.global/ag_cu/Data/VIX/IVYOPPRCD_CSV')
    else:
        os.chdir(r'C:\Users\...\Data\VIX')
elif index=="SPX":
    if cluster_flag==True:
        os.chdir('/scratch.global/ag_cu/Data/SPX/IVYOPPRCD_CSV')
    else:
        os.chdir(r'C:\Users\...\SPX')
        



df_raw= pd.read_csv("IVYOPPRCD_"+str(spot_date)[:-2]+".csv", sep=",")#, dtype={"#NAME?":'str'})

columns_to_keep=['date', 'strike', 'expiration', 'call/put','best bid','best offer','implied volatility', 'volume','open interest','last trade date']

df=df_raw[columns_to_keep]


rename_dict={}
for c in df.columns:
    rename_dict[c]= c.replace(" ","_" )
    
df= df.rename(columns=rename_dict)
 


df_call=df.iloc[list(np.where(np.array(df['call/put'])=="C")[0])]


df_call_group= dict(tuple(df_call.groupby(["date"])))

df=df_call_group[spot_date]

def reformate_func(x):
    result=[]
    for i in x:
        result.append(str(i)[:4]+"-"+str(i)[4:6]+"-"+str(i)[6:]) 
    return result


df[['date','expiration']] = df[['date','expiration']].apply(reformate_func)


df[['date','expiration']] = df[['date','expiration']].apply(pd.to_datetime)

df["time_to_maturity_in_days"]= (df['expiration']-df['date'])/np.timedelta64(1, 'D')


df_per_mat= dict(tuple(df.groupby(["time_to_maturity_in_days"])))

for k in list(df_per_mat.keys()):
    df_per_mat[k]= (df_per_mat[k])[["strike", "best_bid", "best_offer", "implied_volatility", "volume", "open_interest", "last_trade_date"]]



if index=="SPX":
    option_path=os.getcwd()
    
    spot_path= '/scratch.global/ag_cu/Data/SPX/IVYSECPRD_CSV'
    
    os.chdir(spot_path)
    
    #df_spot_raw= pd.read_csv("IVYSECPRD_201905.csv", sep=",", index_col="date")#, dtype={"#NAME?":'str'})
    df_spot_raw= pd.read_csv("IVYSECPRD_"+str(spot_date)[:-2]+".csv", sep=",", index_col="date")#, dtype={"#NAME?":'str'})
    
    df_spot=df_spot_raw[["Close Price"]]
    
    spot=df_spot.loc[spot_date][0]
    
elif index=="VIX":
    option_path=os.getcwd()

    spot_path= '/scratch.global/ag_cu/Data/VIX/Future_Prices'
        
    os.chdir(spot_path)
    
    #df_spot_raw= pd.read_csv("IVYSECPRD_201905.csv", sep=",", index_col="date")#, dtype={"#NAME?":'str'})
    df_spot_raw= pd.read_csv("GI.NA.IVYFUTPRCD_"+str(spot_date)[:-2]+".csv", sep=",")#, index_col="date")#, dtype={"#NAME?":'str'})
    rename_dict={}
    for c in df_spot_raw.columns:
        rename_dict[c]= c.replace(" ","_" )
    
    df_spot_raw= df_spot_raw.rename(columns=rename_dict)
    df_spot_call_group= dict(tuple(df_spot_raw.groupby(["date"])))

    df_spot=df_spot_call_group[spot_date]
    
    df_spot=df_spot.assign(mid_price= lambda x: 1/2*(x.high_price+x.low_price)  )  
    
    df_spot[['date','expiration']] = df_spot[['date','expiration']].apply(reformate_func)

    df_spot[['date','expiration']] = df_spot[['date','expiration']].apply(pd.to_datetime)

    df_spot["time_to_maturity_in_days"]= (df_spot['expiration']-df_spot['date'])/np.timedelta64(1, 'D')

    df_spot_per_mat= dict(tuple(df_spot.groupby(["time_to_maturity_in_days"])))
    
    



zero_curve_path= '/scratch.global/ag_cu/Data/Zero Curves'
os.chdir(zero_curve_path)

col_name=["date", "days", "rate"]

#df_rate_raw= pd.read_csv("IVYZEROCD_201905.txt", sep=r"\s+", header=None, names=col_name )#, dtype={"#NAME?":'str'})
df_rate_raw= pd.read_csv("IVYZEROCD_"+str(spot_date)[:-2]+".txt", sep=r"\s+", header=None, names=col_name )#, dtype={"#NAME?":'str'})

df_rate_group=dict(tuple(df_rate_raw.groupby(["date"])))

df_rate=df_rate_group[spot_date][["days", "rate"]]

df_rate=df_rate.set_index("days")




if index=="SPX":
    divi_path= '/scratch.global/ag_cu/Data/SPX/Dividends'
    os.chdir(divi_path)

    df_divi_raw= pd.read_csv("IVYIDXDVD_"+str(spot_date)[:-2]+".csv", sep=",", index_col="date")#, dtype={"#NAME?":'str'})

    df_divi=df_divi_raw[["dividend_yield"]]

    divi=df_divi.loc[spot_date][0]

elif index=="VIX":
    divi=0





def find_ivol(price, spot, strike, T, r, d):
    r=r/100
    d=d/100
    T=T/365.25

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
    
    
    
def bid_ivol(x):
    result=[]
    for i in range(len(x.best_bid.values)):
        result.append(find_ivol(x.best_bid.values[i],x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i], x.rate.values[i], x.divi.values[i]))
        #result.append(implied_volatility(x.best_bid.values[i], x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i]/365.25, x.rate.values[i]/100,0,"c"))
    return result

def ask_ivol(x):
    result=[]
    for i in range(len(x.best_bid.values)):
        result.append(find_ivol(x.best_offer.values[i],x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i], x.rate.values[i], x.divi.values[i]))
        #result.append(implied_volatility(x.best_offer.values[i], x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i]/365.25, x.rate.values[i]/100,0,"c"))
    return result

def mid_ivol(x):
    result=[]
    for i in range(len(x.best_bid.values)):
        result.append(find_ivol(x.mid.values[i],x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i], x.rate.values[i], x.divi.values[i]))
        #result.append(implied_volatility(x.best_offer.values[i], x.spot.values[i], x.strike.values[i]/1000,  x.days_to_mat.values[i]/365.25, x.rate.values[i]/100,0,"c"))
    return result
    



def get_rate_per_mat(k):
    shortest= df_rate.index[0]
    longest= df_rate.index[-1]
    if k in df_rate.index:
        return df_rate.loc[k].values
    elif k < shortest:
        slope= df_rate.loc[shortest].values/ shortest
        return k*slope
    elif k < longest:
        lb= max(np.where(df_rate.index< k)[0])
        ub= lb+1
        
        slope= (df_rate.iloc[ub].values- df_rate.iloc[lb].values)/(df_rate.index[ub]-df_rate.index[lb])
        intersect= df_rate.iloc[ub].values- slope*df_rate.index[ub]
        
        return k*slope + intersect
        


if index=="SPX":
    save_dir='/scratch.global/ag_cu/Data/SPX/Processed Data'
elif index=="VIX":
    save_dir='/scratch.global/ag_cu/Data/VIX/Processed Data'
   
os.chdir(save_dir)

os.makedirs(str(spot_date), exist_ok=True)
os.chdir(str(spot_date))
os.makedirs("Plots", exist_ok=True)

df_per_mat_cleaned={}

for k in list(df_per_mat.keys()):
    
    rate=get_rate_per_mat(k)[0]
    df_per_mat[k]["rate"]= rate 
        
    if index=="SPX":
        df_per_mat[k]["spot"]= spot
    elif index=="VIX":
        df_per_mat[k]["spot"]= df_spot_per_mat[k][["settlement_price"]].values[0][0]
    df_per_mat[k]["days_to_mat"]= k
    df_per_mat[k]["divi"]=divi
    
    df_per_mat[k]=df_per_mat[k].drop(df_per_mat[k][
        (df_per_mat[k].implied_volatility==-99.99)].index)
        
    df_per_mat[k]=df_per_mat[k].assign(mid= lambda x: (x.best_offer-x.best_bid)/2+x.best_bid  )
        
    if index=="VIX":
        df_per_mat[k].to_csv("ivol_data_maturity_"+str(k)+"days")




for k in list(df_per_mat.keys()):   
    
    df_per_mat[k]= df_per_mat[k].assign(moneyness= lambda x: x.strike/(1000*x.spot)  )
    df_per_mat[k]= df_per_mat[k].assign(log_moneyness= lambda x: np.log(x.moneyness)   )
    
    df_per_mat[k]= df_per_mat[k].assign(bid_ivol= bid_ivol )
    df_per_mat[k]= df_per_mat[k].assign(ask_ivol= ask_ivol )
    df_per_mat[k]= df_per_mat[k].assign(mid_ivol= mid_ivol )
    
    df_per_mat[k]=df_per_mat[k].assign(bps_bid_ivol= lambda x: (x.mid_ivol-x.bid_ivol)*10000  )
    df_per_mat[k]=df_per_mat[k].assign(bps_ask_ivol= lambda x: (x.ask_ivol-x.mid_ivol)*10000 )

       
    df_per_mat_cleaned[k]=df_per_mat[k].drop(df_per_mat[k][
        (df_per_mat[k].bid_ivol==-99.99)].index)
    
    df_per_mat_cleaned[k]=df_per_mat_cleaned[k].drop(df_per_mat_cleaned[k][
        (df_per_mat_cleaned[k].ask_ivol==-99.99)].index)
    
    df_per_mat_cleaned[k]=df_per_mat_cleaned[k].drop(df_per_mat_cleaned[k][
        (df_per_mat_cleaned[k].mid_ivol==-99.99)].index)
    
    df_per_mat_cleaned[k]=df_per_mat_cleaned[k].sort_values(by=['strike'])

    
    df_per_mat_cleaned[k].to_csv("ivol_data_maturity_"+str(k)+"days")
  
    plt.figure()
    plt.plot(df_per_mat_cleaned[k].moneyness.values, df_per_mat_cleaned[k].implied_volatility.values, label="mkt_ivol", color="cornflowerblue", linewidth=1)
    plt.plot(df_per_mat_cleaned[k].moneyness.values, df_per_mat_cleaned[k].mid_ivol.values, label="mid", color="blue", linewidth=1)
    plt.plot(df_per_mat_cleaned[k].moneyness.values, df_per_mat_cleaned[k].bid_ivol.values, "+",label="bid", color="lightsalmon")
    plt.plot(df_per_mat_cleaned[k].moneyness.values, df_per_mat_cleaned[k].ask_ivol.values, "+", label="ask", color="plum")
    plt.xlabel("moneyness")
    plt.ylabel("implied volatility")
    
    if index=="SPX":
        plt.title('Implied Volatility Smile of SPX Option with T='+str(int(k))+'Days')
    else:
        plt.title('Implied Volatility Smile of VIX Option with T='+str(int(k))+'Days')

    plt.legend()
    plt.savefig("Plots/IVOL_Smile_"+str(int(k))+'Days')
    plt.show()


















