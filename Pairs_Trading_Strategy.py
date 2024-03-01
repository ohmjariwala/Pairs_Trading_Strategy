# Author: Ohm Jariwala
# Date: 1/6/2024
# Pairs Trading Strategy

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import datetime as dt
from math import sqrt


# Downloading stock data
start_date= dt.datetime(2010, 1, 1)
end_date= dt.datetime.now()
stocks = ['AAPL', 'MSFT', 'NVDA', 'ARGO', 'CRM', 'CSCO', 'ACN', 'ADBE', 'TXN', 'ORCL'] #XLK
data = yf.download(stocks, start_date, end_date)['Adj Close']
data=data.dropna()

#Fitting a linear regression using Augmented Dickey-Fuller test

#Null Hypothesis: The time series is non-stationary(mean, variance, and/or covariance of the series are not constant)
#Alternate Hypothesis: The time series is stationary(mean, variance, and/or covariance of the series are constant)

lag_order=5
cointegrated= {}

#Compare each stock against others
for stock1 in stocks:
    
    current_cointegrated_stocks= []
    
    for stock2 in stocks:
        #Skipping same stock comparison
        if stock1 == stock2:
            continue
        
        X= np.array(data[stock1]).reshape(-1,1)
        Y= np.array(data[stock2]).reshape(-1,1)
        
        model=LinearRegression()
        model.fit(X,Y)
        
        alpha= model.coef_[0]
        c= model.intercept_
        epsilon= Y- (alpha * X + c)
        
        test=adfuller(epsilon, maxlag= lag_order, regression='c')
        p= test[1]
        
        if (p <0.01):
            current_cointegrated_stocks.append((stock2, round(p, 5)))
            
    cointegrated[stock1]= current_cointegrated_stocks

# Stores the values and the stocks of the cointegration
cointegrated_values= cointegrated

# Identify the most cointegrated stock by finding the max cointegration within the dictionary
most_cointegrated_stock = max(cointegrated_values, key=lambda x: len(cointegrated_values[x]))
    
# Lower p values are indicative of stronger cointegration
# Finding the most cointegrated stocks and keeping the stock's most cointegrated counterpart

for stock, tuples in cointegrated.items():
    if not tuples:
        continue
    pvalues= [tup[1] for tup in tuples]
    minimum_p= min(pvalues)
    cointegrated[stock] = [tup for tup in tuples if tup[1]==minimum_p][0][0]


# Finding the most volatile stocks
returns= data.pct_change().dropna()

volatilities={}
for stock in stocks:
    volatility= np.std(returns[stock]) * sqrt(21)
    volatilities[stock] = volatility

# Identify the most volatile stock by finding the max within the dictionary
most_volatile_stock = max(volatilities, key=lambda x: volatilities[x])

returns= data.pct_change().dropna()
returns= np.log1p(returns)

buys=[]
sell=[]
for i in range(1, len(returns) + 1):
    returns_difference= returns[most_volatile_stock][0:i] - returns[most_cointegrated_stock][0:i]
    mean= np.mean(returns_difference)
    std= np.std(returns_difference)
    
    zscore= (returns_difference - mean)/std
    
    buy_signal= np.zeros_like(zscore)
    buy_signal[zscore > 2] =1 # Most Volatile undervalued, Most Cointegrated overvalued
    buy_signal[zscore < 2] = -1 # Most Cointegrated undervalued, Most Volatile overvalued
    
    if i > 1:
        buy_signal= buy_signal[-1]
        buys.append(buy_signal)
        
data=data[2:]
data['Buy']= [int(buy) for buy in buys]

# IMPLEMENTATION OF STRATEGY

# Set initial capital value and maximum trade size (free to change)
initial_cap= 1000000
max_trade_size= initial_cap * 0.15

# Initialize position and P&L
MV_position=0 # MV= Most Volatile Stock Position
MC_position=0 # MC= Most Cointegrated Stock Position
pnl= []
reduced_cap= initial_cap
profit_percentage= 0

for i in range(len(data)):
    
    if abs(data[most_volatile_stock][i] - data [most_cointegrated_stock][i]) < 0.5 * np.std(data[most_volatile_stock] - data[most_cointegrated_stock]) or pnl[-1] < -0.02 * pnl[-2]:
        initial_cap += MV_position * data[most_volatile_stock][i] + MC_position * data[most_cointegrated_stock][i]
        MV_position= 0
        MC_position= 0
    else:
        
        trade_size= min( 0.5 * initial_cap, max_trade_size)
        buy_signal= data['Buy'][i]
        if (buy_signal == 1):
            trade_MV= trade_size / data[most_volatile_stock][i]
            trade_MC= -trade_size/ data[most_cointegrated_stock][i]
        elif (buy_signal == -1):
            trade_MV = -(trade_size)/data[most_volatile_stock][i]
            trade_MC= trade_size /data[most_cointegrated_stock][i]
        else:
            trade_MV, trade_MC= 0
        
        # Execution of trade
        MV_position += trade_MV
        MC_position += trade_MC
        initial_cap -= (trade_MV * data[most_volatile_stock][i]) + (trade_MC * data[most_cointegrated_stock][i])
        reduced_cap= initial_cap - pnl[-1]
        
        if (reduced_cap < initial_cap * 0.5):
            break
    pnl.append(MV_position * (data[most_volatile_stock][i] - data[most_volatile_stock][0]) + MC_position * (data[most_cointegrated_stock][i] - data[most_cointegrated_stock][0]))
    
# Plot the P&L
plt.plot(pnl)
plt.title(f"Pairs Trading Strategy with {most_volatile_stock} and {most_cointegrated_stock}")
plt.xlabel('Days')
plt.ylabel('P&L')
plt.show()
