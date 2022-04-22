import requests
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pandas_ta as pta
import yfinance as yf

today_d = datetime.date.today().strftime("%Y-%m-%d")
six_mo = datetime.date.today() + relativedelta(months=-6)
six_mo = six_mo.strftime("%Y-%m-%d")
key = '97539e5b6d2151e8ff034e24e8410c89'
url = 'https://api.stlouisfed.org/fred/series/observations'


def vix_snp500():

    params = {
        'series_id': 'VIXCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }
    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    DF = pd.DataFrame(vix_list, index=date_list, columns=['VIX'])

    DF['RSI'] = pta.rsi(DF['VIX'])
    DF.dropna(inplace=True)

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    RSI = {
        'current': round(DF['RSI'][-1], 2),
        'min': round(DF['RSI'].min(), 2),
        'max': round(DF['RSI'].max(), 2)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='SNP500'
        )
    )

    fig.update_layout(
        title="VIX",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig_rsi = go.Figure()

    fig_rsi.add_trace(
        go.Scatter(
            x=date_list,
            y=DF['RSI'],
            name='RSI'
        )
    )

    fig_rsi.update_layout(
        title="RSI VIX",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, fig_rsi, VIX, RSI


def vix_gold():

    params = {
        'series_id': 'GVZCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }

    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='GVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index for Gold",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, VIX


def vix_euro():

    params = {
        'series_id': 'EVZCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }
    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='EuVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index for Europe",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, VIX


def vix_russell():

    params = {
        'series_id': 'RVXCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }
    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='RVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index fo Russell 2000",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, VIX


def vix_emerg():

    params = {
        'series_id': 'VXEEMCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }
    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='EmVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index for Emerging Countries",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, VIX


def vix_nasdaq():

    params = {
        'series_id': 'VXNCLS',
        'realtime_start': six_mo,
        'realtime_end': today_d,
        'observation_start': six_mo,
        'observation_end': today_d,
        'api_key': key,
        'file_type': 'json',
    }
    response = requests.get(url, params=params).json()

    response = response['observations']
    print(response)

    vix_list = []
    date_list = []
    for i in range(len(response)):
        if len(response[i]['value']) == 1:
            continue
        else:
            vix_list.append(float(response[i]['value']))
            date_list.append(response[i]['date'])

    DF = pd.DataFrame(vix_list, index=date_list, columns=['VIX'])

    DF['RSI'] = pta.rsi(DF['VIX'])
    DF.dropna(inplace=True)

    VIX = {
        'current': vix_list[-1],
        'min': min(vix_list),
        'max': max(vix_list)
    }

    RSI = {
        'current': round(DF['RSI'][-1], 2),
        'min': round(DF['RSI'].min(), 2),
        'max': round(DF['RSI'].max(), 2)
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=date_list,
            y=vix_list,
            name='NVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index for NASDAQ",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig_rsi = go.Figure()

    fig_rsi.add_trace(
        go.Scatter(
            x=date_list,
            y=DF['RSI'],
            name='RSI'
        )
    )

    fig_rsi.update_layout(
        title="RSI NASDAQ",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, fig_rsi, VIX, RSI


def vix_china():

    ticker = 'KWEB'
    end = datetime.date.today().strftime("%Y-%m-%d")
    start = datetime.date.today() + relativedelta(years=-1)
    start = start.strftime("%Y-%m-%d")

    ohlc = pd.DataFrame()
    ohlc['Adj Close'] = yf.download(ticker, start, end)['Adj Close']
    print(ohlc)

    ohlc['returns'] = np.log(ohlc / ohlc.shift(-1))
    window_len = int(len(ohlc) / 2)
    gh = ohlc['returns'].rolling(window_len).std() * (252 ** 0.5)
    gh.dropna(inplace=True)
    print(gh)

    days_10 = ohlc['Adj Close'][-10:]
    ret_10 = np.log(days_10 / days_10.shift(-1))
    daily_std_10 = np.std(ret_10)
    std_10 = daily_std_10 * 252 ** .5
    CURRENT = round(std_10 * 100, 2)
    MAX = round(gh.max() * 100, 2)
    MIN = round(gh.min() * 100, 2)
    print(std_10)

    VIX_CHINA = {
        'current': CURRENT,
        'min': MIN,
        'max': MAX
    }
    print(VIX_CHINA)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=gh.index,
            y=gh * 100,
            name='CVIX'
        )
    )

    fig.update_layout(
        title="Volatility Index for China",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig, VIX_CHINA


def fair_cost():
    df = pd.read_html('http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/spearn.htm')
    df = df[0]
    sp_df = pd.DataFrame()

    sp_df['Year'] = df[0][1:]
    sp_df['Earnings Yield'] = df[1][1:]
    sp_df['Dividend Yield'] = df[2][1:]
    sp_df['S&P 500'] = df[3][1:]
    sp_df['Earnings'] = df[4][1:].astype(float)
    sp_df['Dividends'] = df[5][1:].astype(float)
    sp_df['Payout Ratio'] = df[6][1:]

    clean_df = sp_df

    clean_df = clean_df.set_index('Year')

    clean_df['earnings_growth'] = clean_df['Earnings'] / clean_df['Earnings'].shift(1) - 1
    clean_df['dividend_growth'] = clean_df['Dividends'] / clean_df['Dividends'].shift(1) - 1
    clean_df['earnings_10yr_mean_growth'] = clean_df['earnings_growth'].expanding(10).mean()
    clean_df['dividends_10yr_mean_growth'] = clean_df['dividend_growth'].expanding(10).mean()

    print(clean_df)

    valuations = []

    terminal_growth = 0.04  # Рост доходности через 10 лет
    discount_rate = 0.08  #
    payout_ratio = 0.50

    eps_growth_2020 = (11.88 + 17.76 + 28.27 + 31.78) / (34.95 + 35.08 + 33.99 + 35.72) - 1
    eps_2020 = clean_df.iloc[-1]['Earnings'] * (1 + eps_growth_2020)
    eps_next = 28.27 + 31.78 + 32.85 + 36.77
    eps_growth = [0, (clean_df.iloc[-1]['Earnings']) / eps_next - 1,
                  0.18, 0.14, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08]

    value_df = pd.DataFrame()
    value_df['earnings'] = (np.array(eps_growth) + 1).cumprod() * eps_next
    value_df['dividends'] = payout_ratio * value_df['earnings']
    value_df['year'] = [i for i in range(2021, 2031)]
    value_df.set_index('year', inplace=True)

    pv_dividends = 0
    pv_list = []
    for i in range(value_df.shape[0]):
        pv_dividends += value_df['dividends'].iloc[i] / (1 + discount_rate) ** i
        pv_list.append(value_df['dividends'].iloc[i] / (1 + discount_rate) ** i)

    terminal_value = value_df['dividends'].iloc[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

    fair_value = round(pv_dividends + terminal_value / (1 + discount_rate) ** 10, 2)
    sp_cost = float(sp_df['S&P 500'][len(sp_df)])
    print('Fair Value: ', pv_dividends + terminal_value / (1 + discount_rate) ** 10)
    valuations.append(pv_dividends + terminal_value / (1 + discount_rate) ** 10)
    return fair_value, sp_cost
