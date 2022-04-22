import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb


def moving_average_plot(COMPANY):
 data = wb.DataReader(COMPANY, data_source='yahoo', start='2014-1-1')
 # This selects the 'Adj Close' column
 close = data['Adj Close']
 # This converts the date strings in the index into pandas datetime format:
 close.index = pd.to_datetime(close.index)
 #     close.plot()

 #     return plt.show()

 sma50 = close.rolling(window=50).mean()
 sma50
 plt.style.use('fivethirtyeight')
 # The size for our chart:


 sma20 = close.rolling(window=20).mean()
 #     plt.figure(figsize = (12,6))


 priceSma_df = pd.DataFrame({
  'Adj Close': close,
  'SMA 20': sma20,
  'SMA 50': sma50
 })



 sma200 = close.rolling(window=200).mean()
 priceSma_df['SMA 200'] = sma200
 priceSma_df

 # Our start and end dates:
 start = '2019'
 end = '2020'
 plt.figure(figsize=(12, 6))
 # Plotting price and three SMAs with start and end dates:
 plt.plot(priceSma_df[start:end]['Adj Close'], label='SPY Adj Close', linewidth=2)
 plt.plot(priceSma_df[start:end]['SMA 20'], label='20 day rolling SMA', linewidth=1.5)
 plt.plot(priceSma_df[start:end]['SMA 50'], label='50 day rolling SMA', linewidth=1.5)
 plt.plot(priceSma_df[start:end]['SMA 200'], label='200 day rolling SMA', linewidth=1.5)
 plt.xlabel('Date')
 plt.ylabel('Adjusted closing price ($)')
 plt.title('Price with Three Simple Moving Averages')
 plt.legend()

 return plt.show()
