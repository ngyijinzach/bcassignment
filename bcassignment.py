# -*- coding: utf-8 -*-

"""
technical analysis screener using
- rsi
- bollinger bands
- stochastics
- macd crossover

generate buy/sell signals 

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import pandas_datareader as pdr
import plotly.graph_objects as go

# data start/end date & tickers
start_date = '2020-01-01'
end_date = '2020-11-26'
tickers = ['AAPL','AMD','AMZN','BABA','FB','M', 'MSFT','NFLX','NVDA','TSLA']

# ta indicators parameters           
rsi_length=14
bband_length=14
bband_std = 2
stoch_k = 14 
stoch_d = 3 
stoch_smoothk = 3
macd_fast = 12
macd_slow = 26
macd_signal = 9

# ta indicators scoring functions - return buy/sell signal and score
def rsi_score(rsi):
    if rsi > 30 and rsi < 70:
        return 'Neutral', 3
    elif rsi <= 30 and rsi > 15:
        return 'Buy', 2
    elif rsi <= 15:
        return 'Strong Buy', 1
    elif rsi >= 70 and rsi < 85:
        return 'Sell', '4'
    else:
        return 'Strong Sell', '5'

def bband_score(last, lower, mean, upper, bband_std):
    std = (upper-mean)/bband_std
    zscore = (last-mean)/std
    
    if zscore < 2 and zscore > -2:
        return 'Neutral', 3
    elif zscore <= -2 and zscore > 2.5:
        return 'Buy', 2
    elif zscore <= -2.5:
        return 'Strong Buy', 1
    elif zscore >= 2 and zscore < 2.5:
        return 'Sell', 4
    else :
        return 'Strong Sell', 5

def stoch_score(k): 
    if k < 80 and k > 20:
        return 'Neutral', 3
    elif k <= 20 and k > 10:
        return 'Buy' , 2
    elif k <= 10:
        return 'Strong Buy', 1
    elif k >= 90: 
        return 'Strong Sell', 5
    else:
        return 'Sell', 4

def macd_score(macd,macds):
    signal = pd.Series(index=macd.index, data=np.nan, name='signal')
    score =  pd.Series(index=macd.index, data=np.nan, name='score')
    for i in range(1,len(macd)):
        try: 
            # bullish crossover
            if macd.iloc[i-1] < macds.iloc[i-1] and macd.iloc[i] > macds.iloc[i]:
                signal.iloc[i] = 'Buy'
                score.iloc[i] = 2
            # bearish crossover
            elif macd.iloc[i-1] > macds.iloc[i-1] and macd.iloc[i] < macds.iloc[i]:
                signal.iloc[i] = 'Sell'
                score.iloc[i] = 4
            else:
                signal.iloc[i] = 'Neutral'
                score.iloc[i] = 3
        except:
            continue
    return signal, score

def score_to_signal(score):
    if score <1.5:
        return 'Strong Buy'
    elif score < 3:
        return 'Buy'
    elif score == 3:
        return 'Neutral'
    elif score <= 4.5:
        return 'Sell'
    else:
        return 'Strong Sell'
 
# create indicator variable names
rsi = 'RSI' + '(' + str(rsi_length) + ')'
lower_bband = 'L_BBAND' + '(' + str(bband_length) + ',' + str(bband_std) + ')'
mean = 'AVG' + '(' + str(bband_length) + ')'
upper_bband = 'U_BBAND' + '(' + str(bband_length) + ',' + str(bband_std) + ')'
stochk = 'STOCH%K' + '(' + str(stoch_k) + ',' + str(stoch_d) + ',' + str(stoch_smoothk) + ')'
stochd ='STOCH%D' + '(' + str(stoch_k) + ',' + str(stoch_d) + ',' + str(stoch_smoothk) + ')'
macd = 'MACD' + '(' + str(macd_fast) + ',' + str(macd_slow) + ',' + str(macd_signal) + ')'
macdh ='MACDH' + '(' + str(macd_fast) + ',' + str(macd_slow) + ',' + str(macd_signal) + ')'
macds = 'MACDS' + '(' + str(macd_fast) + ',' + str(macd_slow) + ',' + str(macd_signal) + ')'    
 
# create screener table
screener = pd.DataFrame(index=tickers,
                        columns=['LAST', rsi, lower_bband,mean,upper_bband,stochk,stochd,macd,macdh,macds,
                                 'RSIsignal','RSIscore','BBsignal','BBscore','STOCHsignal','STOCHscore','MACDsignal','MACDscore',
                                 'AverageScore','Signal'])
     
# loop through tickers to compute and store ta indicators values/scores 
for ticker in tickers:                
    # retrieve security data
    df = pdr.DataReader(ticker, data_source='yahoo', start=start_date,end=end_date) 
    
    # create ta indicators
    df[rsi] = ta.rsi(df['Adj Close'], length=rsi_length) # rsi
    df[[lower_bband,mean,upper_bband]] = ta.bbands(df['Adj Close'], length=bband_length, std=bband_std) # bollinger bands
    df[[stochk,stochd]] = ta.stoch(df.High,df.Low,df['Adj Close'],k=stoch_k,d=stoch_d,smooth_k=stoch_smoothk)
    df[[macd,macdh,macds]] = ta.macd(df['Adj Close'], fast=macd_fast,slow=macd_slow,signal=macd_signal)
    
    # push data to screener 
    screener.loc[ticker] = df.drop(['High','Low','Open','Close','Volume'],1).iloc[-1].round(2) # latest recent indicators values, to 2 decimals
    screener.loc[ticker,'LAST'] = df['Adj Close'].iloc[-1].round(2) # last price, to 2 decimals
    screener.loc[ticker,'MACDsignal'] = macd_score(df[macd],df[macds])[0].iloc[-1] # macd signal - need data series hence here
    screener.loc[ticker,'MACDscore'] = macd_score(df[macd],df[macds])[1].iloc[-1] # macd score - need data series hence here

# other indicator scores
screener['RSIsignal'] = screener.apply(lambda x: rsi_score(x[rsi])[0],axis=1)
screener['RSIscore'] = screener.apply(lambda x: rsi_score(x[rsi])[1],axis=1)
screener['BBsignal'] = screener.apply(lambda x: bband_score(x['LAST'],x[lower_bband],x[mean],x[upper_bband],bband_std)[0],axis=1)
screener['BBscore'] = screener.apply(lambda x: bband_score(x['LAST'],x[lower_bband],x[mean],x[upper_bband],bband_std)[1],axis=1)
screener['STOCHsignal'] = screener.apply(lambda x: stoch_score(x[stochk])[0],axis=1)
screener['STOCHscore'] = screener.apply(lambda x: stoch_score(x[stochk])[1],axis=1)
screener['AverageScore'] = screener[['RSIscore','BBscore','STOCHscore','MACDscore']].mean(axis=1)
screener['Signal'] = screener.apply(lambda x: score_to_signal(x.AverageScore),axis=1)

# add stock name to left col
screener = screener.rename_axis('Stock').reset_index()

# export
fig = go.Figure(data=[go.Table(
    columnwidth = [20,20,20,40,20,40,40,40,30,30,30,30,25,25,20,30,35,30,30,35,25],
    header=dict(values=list(screener.columns),
                fill_color='paleturquoise',
                align='center',
                font=dict(size=4)),
    cells=dict(values=screener.transpose().values.tolist(),
               fill_color='lavender',
               align='center',
               font=dict(size=5)))
])
fig.update_layout(
    margin=dict(l=10, r=10, t=20, b=20),
)
fig.show()
fig.write_image("fig1.svg")

