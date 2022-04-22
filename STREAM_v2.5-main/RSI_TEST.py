# Подключаем необходимые библиотеки
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import talib

# import cufflinks as cf
from plotly import plot


# Задаем тикер
ticker = 'PICK'
# Задаем диапазон дат в котором нужно собирать все данные по тикерам
start = dt.datetime(2010,1,1).date()
#end = dt.datetime(2010,12,31).date()
end = dt.datetime.today().date() # сегодняшняя дата, чтобы не менять вручную.
# Скачиваем данные
DF = pd.DataFrame(yf.download([ticker],start,end)['Adj Close'])
DF.columns = ['Stock Price']
# Задаем индикаторы
DF['RSI'] = talib.RSI(DF['Stock Price'])
DF.dropna(inplace=True)
df = DF.copy()
df.head
# Задаем стратегию
RSI_treshold = 50
df['positions'] = np.where(df['RSI'] > RSI_treshold,1,0)
df['positions'].sum()

df1 = pd.DataFrame()
df1 = df.copy()
for i in range(len(df['positions'])):
    try:
        if df['positions'][i] < df['positions'][i+1]: #> df['positions'][i+2]:
            df1['positions'][i+1] = 0
    except:
        pass

df['positions'] = df1['positions']

# Визуализация работы стратегии


df.plot(secondary_y = ['positions'], title = "RSI & positions")

df[['Stock Price', 'positions']].plot(secondary_y = ['positions'], title = 'Stock Price & positions')

# Расчет доходности
df['Bay_&_hold_return'] = df['Stock Price'].pct_change()
df['RST_strategy_return'] = df['Bay_&_hold_return']*df['positions'].shift(1)
df['Bay_&_hold_equity'] = (df['Bay_&_hold_return'] + 1).cumprod()
df['RST_strategy_equity'] = (df['RST_strategy_return'] + 1).cumprod()
# Визуализация доходности
(df[['Bay_&_hold_equity', 'RST_strategy_equity']].plot(title = "BH VS RSI"))


# Расчет Сортино и максимального падения
df['cum_str_returns_bh'] = (df['Bay_&_hold_return'] + 1).cumprod()
running_max_BH = np.maximum.accumulate(df['cum_str_returns_bh'].dropna())
drawdown_BH = (df['cum_str_returns_bh'])/running_max_BH - 1
max_dd_BH = drawdown_BH.min()*100
df['cum_str_returns_RSI'] = (df['RST_strategy_return'] + 1).cumprod()
running_max_RSI = np.maximum.accumulate(df['cum_str_returns_RSI'].dropna())
drawdown_RSI = (df['cum_str_returns_RSI'])/running_max_BH - 1
max_dd_RSI = drawdown_RSI.min()*100
risk_free_rate = 0.05
trading_days = 252
daily_risk_free_return = risk_free_rate/trading_days
excess_daily_returns_BH = df['Bay_&_hold_return'] - daily_risk_free_return
average_daily_returns_BH = df['Bay_&_hold_return'].mean()
net_returns_BH = df['Bay_&_hold_return'] - average_daily_returns_BH
negative_returns_BH = net_returns_BH[net_returns_BH < 0]
semi_dev_BH = np.sqrt(np.sum((negative_returns_BH**2))/len(df))
sortino_ratio_BH = (excess_daily_returns_BH.mean()/semi_dev_BH) * np.sqrt(trading_days)
excess_daily_returns_RSI = df['RST_strategy_return'] - daily_risk_free_return
average_daily_returns_RSI = df['RST_strategy_return'].mean()
net_returns_RSI = df['Bay_&_hold_return'] - average_daily_returns_RSI
negative_returns_RSI = net_returns_BH[net_returns_RSI < 0]
semi_dev_RSI = np.sqrt(np.sum((negative_returns_RSI**2))/len(df))
sortino_ratio_RSI = (excess_daily_returns_RSI.mean()/semi_dev_RSI) * np.sqrt(trading_days)

# Выводим результаты
print('With RSI treshold', RSI_treshold)
print('Return from Buy and Hold: ', round(100*(df['Bay_&_hold_equity'][-1]-1),2), '%')
print('Return from RSI strategy: ', round(100*(df['RST_strategy_equity'][-1]-1),2), '%')
print('The Sharpe ratio B&H strategy ', round(np.mean(df['Bay_&_hold_return'])/np.std(df['Bay_&_hold_return'])*(252**0.5),2), '%' )
print('The Sharpe ratio RSI strategy ', round(np.mean(df['RST_strategy_return'])/np.std(df['RST_strategy_return'])*(252**0.5),2), '%' )
print('The Sortino ratio B&H strategy %.2f' % sortino_ratio_BH)
print('The Sortino ratio RSI strategy %.2f' % sortino_ratio_RSI)
print('The maximum drawdown B&H is %.2f' % max_dd_BH)
print('The maximum drawdown RSI is %.2f' % max_dd_RSI)
