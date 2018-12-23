# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:19:48 2018

@author: Furqan
"""
"""
Intro:
This program tends to calculate CVaR which is an enhancement  to VaR. The idea
is to calculate the risk that a portfolio is exposed to.
 
It uses the past observed distribution of portfolio returns to estimate what 
your future losses might be at difference likelihood levels and picks the
best combination of tickers to make you richer than others!

The likelihood level chosen is 99%. There is a chance of 1% error given the 
assumptions.

This back-tester starts by training on one month data provided a correct start
date is given. (see point 3 in important notes)
After each iteration the back tester gets another extra month of data to train
itself and pick the best combination.

"""

"""
Important Notes:
1) reset allocation if you re-run the program, as the allocation changes
   while back-testing and you might end up working out incorrect results.
   
2) The script might contain errors and it does. They will be sorted out
    in the newer version.
    
3) start date should always be a first date of month eg. 01/01/2017
   if given 30/12/16 back tester would only have 1-day data to analyze
   and select a best combination of stock in the first iteration.
   
4) number of stocks selected is a hyper parameter which needs to be tuned as per
   your risk appetite
    
I AM NOT LIABLE FOR ANY LOSSES INCURRED!
"""

import numpy as np
import pandas as pd
import time as t
import datetime
from datetime import datetime
from datetime import timedelta
import datetime as dt


def converter(start_date):
    convert=datetime.strptime(start_date, "%Y-%m-%d")
    return convert

def delta_time(converter,n_days):
    new_date = converter + timedelta(days=n_days)
    return new_date

def data(symbols,start_date,end_date):
    dates=pd.date_range(start_date,end_date)
    df=pd.DataFrame(index=dates)
    df_temp = pd.read_csv('KSE30.csv', index_col='Date')
    df=df.join(df_temp)
    df=df.fillna(method='ffill')
    df=df.fillna(method='bfill')
    return df

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns=(df/df.shift(1))-1
    df=df.fillna(value=0)
    daily_returns=daily_returns[1:]
    return daily_returns

def mat_alloc_auto(symbols, allocation):
    n = len(symbols)
    mat_alloc = np.zeros((n,n), dtype='float')
    for i in range(0,n):
        mat_alloc[i,i] = allocation / n
    return mat_alloc

def var_calculator(data_frame, start_date, end_date):
    value_at_risk_matrix = []
    returns_daily = compute_daily_returns(data_frame)
    
    for symbol in kse30_symbols:
        returns_matrix = returns_daily.loc[start_date : end_date,'{}'.format(symbol)]
        return_matrix = np.array(returns_matrix)
        value_at_risk = np.percentile(return_matrix, 100 * (1-0.99))
        cvarcalc = np.nanmean(returns_matrix < value_at_risk)
        value_at_risk_matrix.append(cvarcalc)
    var_df = pd.DataFrame(data = value_at_risk_matrix, index=kse30_symbols)
    return var_df

def stock_picker(data_frame, start_date, start_date_test, end_date_test):
    var_df = var_calculator(data_frame, start_date, start_date_test)
    var_df.sort_values(by = 0 ,axis = 0, ascending = False, inplace = True)
    symbols_to_invest = var_df.index.values.tolist()[0:number_of_stocks_selected]
    symbols_to_invest_df = data_frame.loc[start_date_test:end_date_test ,symbols_to_invest]
    symbols_to_invest_returns = compute_daily_returns(symbols_to_invest_df)
    return symbols_to_invest, symbols_to_invest_returns



t0 = t.time()   
kse30_symbols = ['OGDC', 'PPL', 'POL', 'MARI', 'NBP', 'BAFL', 'HBL', 'UBL', 'MCB', 
                 'BAHL', 'FCCL', 'DGKC', 'LUCK', 'EFERT', 'FFC', 'ENGRO', 'HUBC',
                 'KAPCO', 'EPCL', 'SSGC', 'SNGP', 'PSO', 'TRG', 'PAEL', 'ISL','NML',
                 'SEARL', 'HCAR', 'MTL', 'ATRL']

start_date = '2017-01-01'
end_date = '2017-03-31'

allocation = 100000

number_of_stocks_selected = 15   
    
#Allocation amount
allocation_portfolio = 100000
equal_allocation = 100000

# Data feed
data_frame = data(kse30_symbols,start_date,end_date)
df_array = np.array(data_frame)

# Convert start and end date to date time format
start_date = converter(start_date)
end_date = converter(end_date)

#Create monthly range of dates
start_rng = pd.date_range(start_date, end_date, freq = 'M')

#Create end date range
end_rng = pd.date_range(start_rng[1], end_date, freq = 'M')


portfolio_amount_matrix = [allocation]
equal_amount_matrix =[allocation]
dfs_returns = []
dfs_cum_sum = []
equal_weight_dfs = []
i = 0
for start_date_test, end_date_test in zip(start_rng, end_rng):
    if start_date_test != end_date_test:
        #print(start_date_test)
        #print(end_date_test)

        #Creating filterd portfolio
        stock_list_invest, stock_list_invest_returns = stock_picker(data_frame, 
                                                                    start_date, 
                                                                    start_date_test, 
                                                                    end_date_test)
        print('List of stocks picked: ', stock_list_invest)
        allocation_matrix = mat_alloc_auto(stock_list_invest, allocation_portfolio)
        valuation = np.dot(stock_list_invest_returns,allocation_matrix)
        valuation = np.sum(valuation, axis=1, keepdims=True)
        valuation = np.divide(valuation,portfolio_amount_matrix)
        dates_portfolio = pd.date_range(delta_time(start_date_test,1),end_date_test)
        portfolio_returns = pd.DataFrame(data=valuation, index=dates_portfolio, 
                                         columns = ['Portfolio Returns'])
        dfs_returns.append(portfolio_returns)

        #change in allocation portfolio
        allocation_portfolio = allocation_portfolio + portfolio_returns.cumsum()['Portfolio Returns'].iloc[-1] *allocation_portfolio
        result_dir = 'E://Stock_Calculation/'
        portfolio_returns.to_csv(result_dir + '{}.csv'.format(i),columns=['Portfolio Returns'],index_label=['Date'])
        portfolio_amount_matrix = [allocation_portfolio]

        #Equal weight portfolio working
        equal_weight_allocation_matrix = mat_alloc_auto(kse30_symbols, equal_allocation)
        equal_weight_dataframe = data_frame.loc[start_date_test:end_date_test ,]
        equal_weight_return = compute_daily_returns(equal_weight_dataframe)
        equal_weight_valuation = np.dot(equal_weight_return,equal_weight_allocation_matrix)
        equal_weight_valuation = np.sum( equal_weight_valuation, axis=1, keepdims=True)
        equal_weight_valuation = np.divide(equal_weight_valuation,equal_amount_matrix)
        dates_equal_weight_portfolio = pd.date_range(delta_time(start_date_test,1),end_date_test)
        equal_weight_portfolio_returns = pd.DataFrame(data=equal_weight_valuation, index=dates_portfolio, columns = ['Equal Weight Portfolio Returns'])
        equal_weight_dfs.append(equal_weight_portfolio_returns)

        #change in allocation equal weight portfolio     
        equal_allocation = equal_allocation + equal_weight_portfolio_returns.cumsum()['Equal Weight Portfolio Returns'].iloc[-1] * equal_allocation
        equal_amount_matrix = [equal_allocation]
        
        i += 1
        print('Allocation Portfolio: {:,}'.format(allocation_portfolio), 
              'Equal Weightage Portfolio: {:,}'.format(equal_allocation))
        
    else:
        break



result_df = pd.concat(dfs_returns)
cum_result = result_df.cumsum()
equal_result_df = pd.concat(equal_weight_dfs)
cum_equal_result = equal_result_df.cumsum()
frames = [cum_result, cum_equal_result]
portfolio = pd.concat(frames, axis=1)
portfolio.to_csv('E://Stock_Calculation/portfolio.csv',index=True)
print("Equal weightage Portfolio returns are ", 
      cum_equal_result['Equal Weight Portfolio Returns'].iloc[-1] * 100,'%')
print("Portfolio returns are ", cum_result['Portfolio Returns'].iloc[-1] * 100,'%')
t1 = t.time()
print('Exec time is ', t1-t0, 'seconds')