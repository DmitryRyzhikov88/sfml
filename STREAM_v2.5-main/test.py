## -------- Import the necessary libraries -------------
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
from pandas_datareader import data as web
import datetime


import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials
import httplib2
list_of_tickers = ['AAPL', 'INTC', 'REMX', 'TME', 'AEIS']

print(list_of_tickers)
start_year = 2010
end = datetime.datetime.now()

p = pdr.DataReader(list_of_tickers, 'yahoo', datetime.datetime(start_year,1,1), end)['Adj Close']
#p = pd.read_csv('SMA-Breakeout-2017-2021-1.csv', low_memory=False)

data_pc = p.pct_change()


def plot_returns_dd(portfolio):
    # ----------- Sharpe ratio ------------------
    sharpe_ratio = np.mean(portfolio.str_returns) / np.std(portfolio.str_returns) * (252 ** 0.5)
    #     print('The Sharpe ratio is %.2f ' % sharpe_ratio)

    # ----------- Cumulative strategy returns ------------------
    portfolio['cum_str_returns'] = (portfolio['str_returns'] + 1).cumprod()

    # ----------- Drawdown ------------------
    # Calculate the running maximum
    running_max = np.maximum.accumulate(portfolio['cum_str_returns'].dropna())
    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1
    # Calculate the percentage drawdown
    drawdown = (portfolio['cum_str_returns']) / running_max - 1
    max_dd = drawdown.min() * 100
    #     print('The maximum drawdown is %.2f' % max_dd)

    cumulative_return = (portfolio['cum_str_returns'].iloc[-2] - 1) * 100
    #     print('cum_str_returns', cumulative_return)

    return portfolio


def get_strategy_returns_breakout(portfolio_breakout):
    portfolio_breakout = portfolio.copy()
    print(portfolio_breakout)
    # Calculate the breakout indicator values
    portfolio_breakout['high'] = portfolio_breakout.value.rolling(window=3).max()
    # Create a trading signal
    portfolio_breakout['signal'] = np.where(portfolio_breakout.value >= portfolio_breakout.high, 1, 0)
    # Calculate the strategy returns
    portfolio_breakout['str_returns'] = portfolio_breakout['returns'].shift(-1) * portfolio_breakout['signal']
    print(portfolio_breakout)
    return portfolio_breakout


def get_strategy_returns_sma_breakout(portfolio_sma_breakout):
    portfolio_sma_breakout = portfolio.copy()
    print(portfolio_sma_breakout)
    # Calculate the simple moving average
    sma10 = portfolio_sma_breakout.value > portfolio_sma_breakout.value.rolling(window=10).mean()
    # Calculate the breakout indicator values
    breakout = portfolio_sma_breakout.value >= portfolio_sma_breakout.value.rolling(window=3).max()
    # Create a trading signal
    portfolio_sma_breakout['signal'] = np.where(sma10 & breakout, 1, 0)
    # Calculate the strategy returns
    portfolio_sma_breakout['str_returns'] = portfolio_sma_breakout['returns'].shift(-1) * portfolio_sma_breakout[
        'signal']
    print(portfolio_sma_breakout)

    return portfolio_sma_breakout


def get_strategy_returns_sma(portfolio_sma):
    portfolio_sma = portfolio.copy()
    print(portfolio_sma)
    # Calculate the simple moving average of period 10
    portfolio_sma['sma10'] = portfolio_sma.value.rolling(window=10).mean()
    # Create a trading signal
    portfolio_sma['signal'] = np.where(portfolio_sma.value > portfolio_sma.sma10, 1, 0)
    # Calculate the strategy returns
    portfolio_sma['str_returns'] = portfolio_sma['returns'].shift(-1) * portfolio_sma['signal']
    print(portfolio_sma)
    print(print('+' * 50))
    return portfolio_sma


# Calculate the standard deviation
# data_std = np.std(p, axis = 0)
for i in range(2021 - start_year):
    data_std = data_pc.loc[f'{start_year + i}-01-01':f'{start_year + i}-12-31'].std() * (252 ** 0.5) * 100
    #     print('Date:', start_year+i, 'STD:', data_std)
    vol_sorted = data_std.sort_values(ascending=False)
    top_decile = vol_sorted[:int(len(data_std) * 0.2)]
    #     print('top_decile', top_decile)
    data = p
    data['Date'] = data.index.tolist()
    data = data.set_index(data['Date'].dt.year)
    # Retrieve data in stock_list from 2019 January onwards
    stock_data = data.loc[start_year + i + 1, top_decile.index]
    #     print('*'*50)
    #     print(stock_data)
    #     print('*'*50)
    # Calculate the daily percentage change of prices
    stock_data_pc = stock_data.pct_change()
    # Create a new dataframe called portfolio
    portfolio = pd.DataFrame()

    # Calculate the average returns of stocks
    portfolio['returns'] = stock_data_pc.mean(axis=1)
    # Calculate cumulative returns of portfolio
    portfolio['value'] = (portfolio + 1).cumprod()
    # Drop any rows with nan values
    portfolio = portfolio.dropna()
    print(portfolio.head(11))
    portfolio_sma = get_strategy_returns_sma(portfolio)
    portfolio_breakout = get_strategy_returns_breakout(portfolio)
    print(portfolio.head(11))
    portfolio_sma_breakout = get_strategy_returns_sma_breakout(portfolio)
    #
    portfolio_sma_result = plot_returns_dd(portfolio_sma)
    portfolio_breakout_result = plot_returns_dd(portfolio_breakout)
    portfolio_sma_breakout_result = plot_returns_dd(portfolio_sma_breakout)
    print(portfolio.head(11))
    print(portfolio_sma.head(11))
    print(portfolio_breakout.head(11))
    print(portfolio_sma_breakout.head(11))
    print('*' * 50)




