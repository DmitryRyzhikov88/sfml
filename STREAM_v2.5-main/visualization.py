import datetime
from urllib.request import urlopen, Request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as pta
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas_datareader import data as web
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn import mixture as mix
from yahoo_fin.stock_info import *
from yahoofinancials import  YahooFinancials


yf.pdr_override()

KEY = '2105b9f242d47b69fc73f0f2205be048'
#KEY = 'd7c445cd5c6eaee5d52540f8a96d0e7f'


def net_income_plot(size, IS):
    #IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={KEY}').json()
    income_statement = pd.DataFrame().from_dict(IS).fillna(0)
    print(income_statement)

    x = np.arange(income_statement['netIncome'].size)
    print(x)
    fit = np.polyfit(x, income_statement['netIncome'], deg=1)
    fited = np.poly1d(fit)
    print(ff)

    print(income_statement['netIncome'])
    res = fited(x)
    print(res)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=income_statement['date'][0:size],
            y=income_statement['netIncome'][0:size],
            name="Upper Band"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=income_statement['date'],
            y=res,
            name='Trend'
        )
    )

    fig.update_layout(
        title={
            'text': "Net Income Plot",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Net Income"
    )

    return fig


def net_income_plot_2(ticker):
    i_s = get_income_statement(ticker, yearly=True)
    data_plot = i_s.iloc[4].astype(float)

    x = np.arange(data_plot.size)
    print(x)
    fit = np.polyfit(x, data_plot.values, deg=1)
    fited = np.poly1d(fit)
    print(fited)

    res = fited(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=data_plot.index,
            y=data_plot['netIncome'],
            name="Upper Band"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data_plot.index,
            y=res,
            name='Trend'
        )
    )

    fig.update_layout(
        title={
            'text': "Net Income Plot",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Net Income"
    )

    return fig


def earning_date_plot(ticker):

    yahoo_financials = YahooFinancials(ticker)
    dat = yahoo_financials.get_stock_earnings_data()
    dat = dat[ticker]
    dat = dat['earningsData']
    res = list(dat.values())
    res = res[4]
    res = res[0]
    format = "%Y-%m-%d"
    result = res['fmt']

    return result


def revenue_plot(size, IS):
    #IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={KEY}').json()
    revenue = pd.DataFrame().from_dict(IS).fillna(0)

    x = np.arange(revenue['revenue'].size)
    print(x)
    fit = np.polyfit(x, revenue['revenue'], deg=1)
    fited = np.poly1d(fit)
    res = fited(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=revenue['date'][0:size],
            y=revenue['revenue'][0:size],
            name="Upper Band"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=revenue['date'],
            y=res,
            name='Trend'
        )
    )

    fig.update_layout(
        title={
            'text': "Revenue Plot",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Revenue"

    )
    return fig


def book_value_years_plot(size, KM):
    # KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={KEY}').json()
    book = pd.DataFrame().from_dict(KM).fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=book['date'][0:size],
            y=book['bookValuePerShare'][0:size],
            name="Upper Band"
        )
    )
    fig.update_layout(
        title={
            'text': "Book Value Per Share Plot",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Book Value Per Share"

    )
    return fig


def dividend_pershare_plot(ticker, size):
    dividend = pd.DataFrame(yf.Ticker(ticker).dividends)
    dividend['date'] = dividend.index.tolist()
    dividend.groupby(dividend['date'].dt.year).sum()
    dividend_year = dividend.groupby(dividend['date'].dt.year).sum()[::-1]

    print(dividend_pershare_plot)
    print(dividend)
    print(dividend_year)
    print(dividend_year.index.tolist()[0:size])
    print(dividend_year[0:size])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=dividend_year.index.tolist()[0:size],
            y=dividend_year['Dividends'][0:size],
            name="Book Value Per Share"
        )
    )

    fig.update_layout(
        title={
            'text': "Dividend per Share Plot",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Dividend per Share"
    ),

    return fig


def freeCashFlow_plot(size, CF, CF_TTM):
    # Добавление CF за последние 4 месяца TTF
    # CF = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={KEY}').json()

    # CF_TTM = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&apikey={KEY}').json()

    cash_flow = pd.DataFrame(CF)
    cash_flow_TTM = pd.DataFrame(CF_TTM)
    q_cash_flow_statement = cash_flow_TTM.set_index('date').iloc[:4]
    q_cash_flow_statement = cash_flow_TTM.apply(pd.to_numeric, errors='coerce')
    ttm_cash_flow_statement = cash_flow_TTM[0:4].sum()  # sum up last 4 quarters to get TTM cash flow

    cash_flow_statement = cash_flow[::-1].append(ttm_cash_flow_statement.rename('TTM')).drop(['netIncome'],
                                                                                             axis=1)
    final_cash_flow_statement = cash_flow_statement[::-1].reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=final_cash_flow_statement['date'][0:size + 1],
            y=final_cash_flow_statement['freeCashFlow'][0:size + 1],
            name="Upper Band"
        )
    )

    fig.update_layout(
        title="Free Cash Flow",
        xaxis_title="Date",
        yaxis_title="Free Cash Flow",
        title_x=0.5,
        legend=dict(x=.5, xanchor="center", orientation="h"),
        margin=dict(l=0, r=0, t=30, b=0)),

    return fig


def margin_plot2(PF):
    # PF = requests.get(f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={KEY}').json()
    company_profile = pd.DataFrame().from_dict(PF).fillna(0)
    sector_company = company_profile['sector'][0]
    sector_margin_company = pd.read_pickle("sector_margin.pkl")
    sector_margin_company = pd.DataFrame(sector_margin_company)

    overview_Gross_Margin = sector_margin_company[sector_company]['Gross Margin']
    overview_Operating_Margin = sector_margin_company[sector_company]['Operating Margin']
    overview_Net_Profit_Margin = sector_margin_company[sector_company]['Net Profit Margin']

    hist_data = [overview_Gross_Margin, overview_Operating_Margin, overview_Net_Profit_Margin]

    group_labels = ['Gross Margin', 'Operating Margin', 'Net Profit Margin']

    figure = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    figure.update_layout(
        title="Распределение маржинальностей",
        title_x=0.5,
        legend=dict(x=.5, xanchor="center", orientation="h"),
        margin=dict(l=0, r=0, t=30, b=0))

    fig = go.Figure(figure)

    return fig


def scored_news(COMPANY):
    finwiz_url = 'https://finviz.com/quote.ashx?t='

    news_tables = {}
    tickers = [COMPANY]

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
        response = urlopen(req)
        html = BeautifulSoup(response)
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    news = news_tables[COMPANY]
    news_tr = news.findAll('tr')

    for i, table_row in enumerate(news_tr):
        a_text = table_row.a.text
        td_text = table_row.td.text
        if i == 7:
            break

    parsed_news = []

    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()
            if len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            ticker = file_name.split('_')[0]
            parsed_news.append([ticker, date, time, text])

    vader = SentimentIntensityAnalyzer()

    columns = ['ticker', 'date', 'time', 'headline']
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news.head()

    plt.rcParams['figure.figsize'] = [14, 8]
    mean_scores = parsed_and_scored_news.groupby(['ticker', 'date']).mean()
    mean_scores = mean_scores.unstack()
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()

    fig, ax = plt.subplots()

    ax.bar(mean_scores.index, mean_scores[f'{COMPANY}'], width=4)

    plt.grid(color='black', which='major', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(color='black', linestyle=':', linewidth=0.3, which='minor')

    # fig = go.Figure(ax.bar(mean_scores.index, mean_scores[f'{COMPANY}'], width=4))

    fig = go.Figure(
        data=[go.Bar(x=mean_scores.index, y=mean_scores[f'{COMPANY}'])],
        layout=go.Layout(
            title=go.layout.Title(text="Sentiment Interest By Scored News")
        )
    )

    return fig


def stock_price_PE(size, IS_quarter, EV_quarter):

    # IS_quarter = requests.get(
    #   f'https://financialmodelingprep.com/api/v3/income-statement/{COMPANY}?&apikey={KEY}').json()

    # f'https://financialmodelingprep.com/api/v3/income-statement/{COMPANY}?period=quarter&apikey={KEY}').json()

    # EV_quarter = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/enterprise-values/{COMPANY}?&apikey={KEY}').json()

    # print('IS_quarter')
    # print(IS_quarter)
    # print('EV_quarter')
    # print(EV_quarter)

    enterprise_value_quarter = pd.DataFrame().from_dict(EV_quarter).fillna(0)
    income_statement_quarter = pd.DataFrame().from_dict(IS_quarter).fillna(0)

    PE_qarter = enterprise_value_quarter['marketCapitalization'] / income_statement_quarter[
        'netIncome']

    # PE_year = enterprise_value[:size]['marketCapitalization'] / income_statement[:size]['netIncome']

    # _______________Stock Price and PE Quarterly__________________________________

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=income_statement_quarter['date'][0:size + 1],
            y=PE_qarter[0:size + 1],
            name="PE Annually"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=enterprise_value_quarter['date'][0:size + 1],
            y=enterprise_value_quarter['stockPrice'][0:size + 1],
            name="Stock Price"
        )
    )
    fig.update_layout(
        title="Stock Price and PE Annually",
        xaxis_title="Date",
        yaxis_title="Dollar",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def debt_to_assets(BL, EV):

    # BL = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{COMPANY}?apikey={KEY}').json()
    # EV = requests.get(f'https://financialmodelingprep.com/api/v3/enterprise-values/{COMPANY}?apikey={KEY}').json()

    balance = pd.DataFrame().from_dict(BL).fillna(0)
    enterprise_value = pd.DataFrame().from_dict(EV).fillna(0)

    plot_date = pd.to_datetime(balance['date'][:10].reset_index(drop=True), format='%Y-%m-%d').dt.year

    debt_to_capital = balance['totalDebt'][0:10] / enterprise_value['marketCapitalization'][:10]
    debt_to_assets = balance['totalDebt'][0:10] / balance['totalAssets'][0:10]

    fig = go.Figure(
        data=[go.Bar(x=balance['date'][0:10],
                     y=debt_to_capital,
                     name="Debt To Capital"
                     )],
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )
    fig.add_trace(
        go.Bar(x=balance['date'][0:10],
               y=debt_to_assets,
               name="Debt To Assets"
               )

    )

    fig.update_layout(
        title="Debt to Assets and Capitalization",
        xaxis_title="Date",
        yaxis_title="Ratio",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0))

    return fig


def dividends_payout(size, RT):
    # RT = requests.get(f'https://financialmodelingprep.com/api/v3/ratios/{COMPANY}?apikey={KEY}').json()
    ratios = pd.DataFrame().from_dict(RT).fillna(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ratios['date'][0:size],
            y=ratios['dividendYield'][0:size].round(2) * 100,
            name="Dividend Yield"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ratios['date'][0:size],
            y=ratios['payoutRatio'][0:size].round(2) * 100,
            name="Payout Ratio"
        )
    )
    fig.update_layout(
        title="Dividends Payout",
        xaxis_title="Date",
        yaxis_title="Percents",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0))

    return fig


def major_multipliers(size, IS, EV, KM, CF):
    # IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{COMPANY}?apikey={KEY}').json()
    # EV = requests.get(f'https://financialmodelingprep.com/api/v3/enterprise-values/{COMPANY}?apikey={KEY}').json()
    # KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{COMPANY}?apikey={KEY}').json()
    # CF = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{COMPANY}?apikey={KEY}').json()

    income_statement = pd.DataFrame().from_dict(IS).fillna(0)
    enterprise_value = pd.DataFrame().from_dict(EV).fillna(0)
    key_metrics = pd.DataFrame().from_dict(KM).fillna(0)
    cash_flow = pd.DataFrame().from_dict(CF).fillna(0)

    PE_year = enterprise_value[:size]['marketCapitalization'] / income_statement[:size]['netIncome']
    PS = enterprise_value[:size]['marketCapitalization'] / income_statement[:size]['revenue']
    PBV = enterprise_value[:size]['stockPrice'] / key_metrics['bookValuePerShare'][:size]
    PFCF = enterprise_value[:size]['marketCapitalization'] / cash_flow['freeCashFlow'][:size]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=enterprise_value['date'][0:size],
            y=PE_year,
            name="PE"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=enterprise_value['date'][0:size],
            y=PS,
            name="PS"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=enterprise_value['date'][0:size],
            y=PBV,
            name="PBV"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=enterprise_value['date'][0:size],
            y=PFCF,
            name="PFCF"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=key_metrics['date'][0:size],
            y=key_metrics['capexToRevenue'][:size],
            name="CAPEX"
        )
    )
    fig.update_layout(
        title="Major Multipliers",
        xaxis_title="Date",
        yaxis_title="Dollar",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0))

    return fig


def revenue_operating_income_fcf(size, IS, CF):
    # IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{COMPANY}?apikey={KEY}').json()
    # CF = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{COMPANY}?apikey={KEY}').json()

    income_statement = pd.DataFrame().from_dict(IS).fillna(0)
    cash_flow = pd.DataFrame().from_dict(CF).fillna(0)

    fig = go.Figure(
        data=[go.Bar(x=income_statement['date'][0:size],
                     y=income_statement['revenue'][:size],
                     name="Revenue"
                     )],
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )
    fig.add_trace(
        go.Bar(x=income_statement['date'][0:size],
               y=income_statement['operatingIncome'][:size],
               name="Operating Income"
               )

    )
    fig.add_trace(
        go.Bar(x=income_statement['date'][0:size],
               y=income_statement['netIncome'][:size],
               name="Net Income"
               )

    )
    fig.add_trace(
        go.Bar(x=income_statement['date'][0:size],
               y=cash_flow['freeCashFlow'][:size],
               name="Free Cash Flow"
               )

    )

    fig.update_layout(
        title="Revenue, Operating Income, Net Income, Free Cash Flow",
        xaxis_title="Date",
        yaxis_title="Dollar",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0))

    return fig


def roa(size, IS, BL):
    # IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{COMPANY}?apikey={KEY}').json()
    # BL = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{COMPANY}?apikey={KEY}').json()

    balance = pd.DataFrame().from_dict(BL).fillna(0)
    income_statement = pd.DataFrame().from_dict(IS).fillna(0)

    ROA = income_statement['netIncome'][:size] / balance['totalAssets'][:size]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=income_statement['date'][0:size],
            y=ROA,
            name="ROA"
        )
    )
    fig.update_layout(
        title="ROA",
        xaxis_title="Date",
        yaxis_title="Dollar",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


# Стадии рынка

def market_stage(ticker):
    df = web.get_data_yahoo(ticker, start='2019-01-01', end='2021-04-26')
    df = df[['Open', 'High', 'Low', 'Close']]
    df['open'] = df['Open'].shift(1)
    df['high'] = df['High'].shift(1)
    df['low'] = df['Low'].shift(1)
    df['close'] = df['Close'].shift(1)

    df = df[['open', 'high', 'low', 'close']]
    df = df.dropna()

    unsup = mix.GaussianMixture(n_components=4,
                                covariance_type="spherical",
                                n_init=100,
                                random_state=42)
    unsup.fit(np.reshape(df, (-1, df.shape[1])))
    regime = unsup.predict(np.reshape(df, (-1, df.shape[1])))
    df['Return'] = np.log(df['close'] / df['close'].shift(1))
    Regimes = pd.DataFrame(regime, columns=['Regime'], index=df.index) \
        .join(df, how='inner') \
        .assign(market_cu_return=df.Return.cumsum()) \
        .reset_index(drop=False) \
        .rename(columns={'index': 'Date'})

    order = [0, 1, 2, 3]
    fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, aspect=2, height=4)
    fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
    plt.show()

    for i in order:
        print('Mean for regime %i: ' % i, unsup.means_[i][0])
        print('Co-Variance for regime %i: ' % i, (unsup.covariances_[i]))

# Технические показатели


def act_cost(COMPANY, size):

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = yf.download(COMPANY, start, end)
    print(DF)

    max_idx = argrelextrema(np.array(DF['Volume'].values), np.greater, order=4)
    min_idx = argrelextrema(np.array(DF['Volume'].values), np.less, order=4)

    DF['peaks'] = np.nan
    DF['lows'] = np.nan

    for i in max_idx:
        DF['peaks'][i] = DF['Volume'][i]
    for i in min_idx:
        DF['lows'][i] = DF['Volume'][i]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['Close'],
            name="Peaks"
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['Volume'],
            name='Volume'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['peaks'],
            marker=dict(
                size=10,
                symbol='triangle-up'
            ),
            name='Peaks'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['lows'],
            marker=dict(
                size=10,
                symbol='triangle-down'
            ),
            name='Lows'
        ),
        secondary_y=False
    )

    fig.update_layout(
        title="Stock price and Volume",
        xaxis_title="Date",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Volume cost", secondary_y=False)
    fig.update_yaxes(title_text="Stock price", secondary_y=True)

    return fig


def candle(COMPANY, size):

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = pd.DataFrame(yf.download([COMPANY], start, end))
    DF.dropna(inplace=True)
    print(DF)

    op = DF['Open'].astype(float)
    hi = DF['High'].astype(float)
    lo = DF['Low'].astype(float)
    cl = DF['Close'].astype(float)

    fig = go.Figure(data=[go.Candlestick(
        x=DF.index,
        open=op,
        high=hi,
        low=lo,
        close=cl)])

    fig.update_layout(
        title="Candle",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def volume(COMPANY, size):

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = pd.DataFrame(yf.download([COMPANY], start, end))
    DF.dropna(inplace=True)

    max_idx = argrelextrema(np.array(DF['Volume'].values), np.greater, order=4)
    min_idx = argrelextrema(np.array(DF['Volume'].values), np.less, order=4)

    print(max_idx)

    DF['peaks'] = np.nan
    DF['lows'] = np.nan

    for i in max_idx:
        DF['peaks'][i] = DF['Volume'][i]
    for i in min_idx:
        DF['lows'][i] = DF['Volume'][i]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['Volume'],
            name='Volume'
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['peaks'],
            marker=dict(
                size=10,
                symbol='triangle-up'
            ),
            name='Peaks'
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['lows'],
            marker=dict(
                size=10,
                symbol='triangle-down'
            ),
            name='Lows'
        )
    )

    fig.update_layout(
        title="Volume",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def macd(COMPANY, size):

    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df['hist'] = df['macd'] - df['signal']
        df.dropna(inplace=True)
        return df

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()
    dataMACD = yf.download(COMPANY, start, end)
    op = dataMACD['Open'].astype(float)
    hi = dataMACD['High'].astype(float)
    lo = dataMACD['Low'].astype(float)
    cl = dataMACD['Close'].astype(float)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()

    print(df_macd)

    figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    figMACD.add_trace(
        go.Candlestick(
            x=df_macd['Date'],
            open=op,
            high=hi,
            low=lo,
            close=cl,
            name="Price for " + str(size) + " Year(s)"
        ),
        row=1, col=1
    )

    # figMACD.add_trace(
    #    go.Scatter(
    #        x=df_macd['Date'],
    #        y=df_macd['Adj Close'],
    #        name="Price for " + str(size) + "Year(s)"
    #    )
    # )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema12'],
            name="EMA 12 Over Last " + str(size) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema26'],
            name="EMA 26 Over Last " + str(size) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['macd'],
            name="MACD Line"
        ),
        row=2, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['signal'],
            name="Signal Line"
        ),
        row=2, col=1
    )

    figMACD.add_trace(
        go.Bar(
            x=df_macd['Date'],
            y=df_macd['hist'],
            name="MACD Histogram"
        ),
        row=2, col=1
    )

    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0,
    ))

    figMACD.update_layout(xaxis_rangeslider_visible=False)

    figMACD.update_yaxes(tickprefix="$")

    return figMACD


def rsi(COMPANY, size):
    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = pd.DataFrame(yf.download([COMPANY], start, end)['Adj Close'])
    DF.columns = ['Stock Price']

    DF['RSI'] = pta.rsi(DF['Stock Price'])
    DF.dropna(inplace=True)

    # Задаем стратегию
    RSI_treshold = 50
    DF['positions'] = np.where(DF['RSI'] > RSI_treshold, 1, 0)

    print(DF['positions'])
    print(DF.index)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['RSI'],
            name="RSI"
        )
    )

    fig.update_layout(
        title="RSI",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def psar(COMPANY, size):

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = pd.DataFrame(yf.download([COMPANY], start, end))
    DF.dropna(inplace=True)
    pSAR = DF.ta.psar(high=DF['High'], low=DF['Low'], close=DF['Close'], af0=0.02, af=0.02, max_af=0.2)
    op = DF['Open'].astype(float)
    hi = DF['High'].astype(float)
    lo = DF['Low'].astype(float)
    cl = DF['Close'].astype(float)

    result = []
    for i in range(len(pSAR)):
        if np.isnan(pSAR['PSARl_0.02_0.2'][i]):
            result.append(pSAR['PSARs_0.02_0.2'][i])
        else:
            result.append(pSAR['PSARl_0.02_0.2'][i])
    DF['SAR'] = result

    fig = go.Figure(data=[go.Candlestick(
        x=DF.index,
        open=op,
        high=hi,
        low=lo,
        close=cl)])

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['SAR'],
            name="SAR"
        )
    )

    fig.update_layout(
        title="Parabolic SAR",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def agrmax(COMPANY, size):
    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = pd.DataFrame(yf.download([COMPANY], start, end))
    DF.dropna(inplace=True)

    max_idx = argrelextrema(np.array(DF['Close'].values), np.greater, order=3)
    min_idx = argrelextrema(np.array(DF['Close'].values), np.less, order=3)
    print(max_idx)
    DF['peaks'] = np.nan
    DF['lows'] = np.nan
    for i in max_idx:
        DF['peaks'][i] = DF['Close'][i]
    for i in min_idx:
        DF['lows'][i] = DF['Close'][i]

    print(DF['lows'])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['Close'],
            name='Price'
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['peaks'],
            marker=dict(
                size=10,
                symbol='triangle-up'
            ),
            name='Peaks'
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=DF.index,
            y=DF['lows'],
            marker=dict(
                size=10,
                symbol='triangle-down'
            ),
            name='Lows'
        )
    )

    fig.update_layout(
        title="Extremes",
        xaxis_title="Date",
        yaxis_title="USD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def vix(size):
    VIX = '^VIX'
    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF = yf.download(VIX, start, end)
    print(DF)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=DF.index,
            y=DF['Adj Close'],
            name="Peaks"
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

    return fig


def pairTrade(COMPANY1, COMPANY2, size):

    start = datetime.datetime(2022 - size, 1, 1).date()
    end = datetime.datetime.today().date()

    DF1 = pd.DataFrame(yf.download([COMPANY1], start, end))
    DF1.dropna(inplace=True)
    DF2 = pd.DataFrame(yf.download([COMPANY2], start, end))
    DF2.dropna(inplace=True)

    DF1['Close'] = DF1['Close'] / DF1['Close'].max()
    DF2['Close'] = DF2['Close'] / DF2['Close'].max()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=DF1.index,
            y=DF1['Close'],
            name="Price for " + str(COMPANY1)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=DF2.index,
            y=DF2['Close'],
            name="Price for " + str(COMPANY2)
        )
    )

    fig.update_layout(
        title="Normalized Pairs Trading",
        xaxis_title="Date",
        yaxis_title="NUSD",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def balance_sheet(COMPANY, size):
    assets = []
    liabilities = []
    equity = []

    BL = requests.get(
        f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{COMPANY}?period=quarter&limit={size}&apikey={KEY}').json()

    balance = pd.DataFrame().from_dict(BL).fillna(0)

    assets = balance['totalAssets']

    liabilities = balance['totalLiabilities']

    equity = balance['totalStockholdersEquity']

    fig = go.Figure(data=[
        go.Bar(name='Assets', x=balance.date, y=assets),
        go.Bar(name='Liabilities', x=balance.date, y=liabilities),
        go.Bar(name='Equity', x=balance.date, y=equity)
    ])

    fig.update_layout(barmode='stack', title='Balance Sheet Latest Quarter')

    return fig


def balance_bars(COMPANY):
    assets = []
    liabilities = []
    equity = []

    msft = yf.Ticker(COMPANY)
    df = msft.balance_sheet.iloc[:, :1].dropna()

    df['values'] = df.values
    df['index'] = df.index.tolist()

    print(df)

    numberz = []

    for i in range(len(df)):
        if 'Debt' in df['index'][i]:
            numberz.append(i)
        elif "Liab" in df['index'][i]:
            numberz.append(i)
        elif "Accounts Payable" in df['index'][i]:
            numberz.append(i)

    df1 = df.drop(df.iloc[numberz]['index'].values, axis=0)
    df2 = df.iloc[numberz]

    df1 = df1.drop(index=('Total Assets'))
    df2 = df2.drop(index=('Total Liab'))


    p1 = [''] * len(df1)
    p2 = [''] * len(df2)

    #p1 = ['Total Assets'] + [''] + (['Total Assets'] * (len(df1) - 2))
    #p2 = [''] + (['Total Liab'] * (len(df2) - 1))

    fig1 = go.Figure(go.Treemap(
        labels=df1['index'],
        values=df1['values'],
        parents=p1
    ))

    fig2 = go.Figure(go.Treemap(
        labels=df2['index'],
        values=df2['values'],
        parents=p2
    ))

    fig1.update_layout(title='Balance Treemap Assets Latest Year')
    fig2.update_layout(title='Balance Treemap Liabilities Latest Year')

    return fig1, fig2


def institutional_investor(COMPANY, institutional_period):
    change = []
    date = []
    institutions = requests.get(
        f'https://financialmodelingprep.com/api/v3/institutional-holder/{COMPANY}?apikey={KEY}').json()
    # print(institutions[0]['dateReported'])
    # print(institutions[0]['change'])

    for item in institutions:
        # print(item)

        change.append(item['change'])
        date.append(item['dateReported'])

    institutions_DF = pd.DataFrame()
    institutions_DF['change'] = change
    institutions_DF['date'] = date
    institutions_DF.index = pd.to_datetime(institutions_DF['date'])
    # institutions_DF.index().to_datetime()
    # print(institutions_DF.head(50))
    # print(institutions_DF.resample('D').sum())

    institutions_DF_month = institutions_DF.resample('M').sum()
    institutions_DF_month['change'] = institutions_DF_month['change'].replace(0, np.nan)
    institutions_DF_month = institutions_DF_month.dropna()

    institutions_DF_year = institutions_DF.resample('Y').sum()

    # print(institutions_DF_month)
    # print(institutions_DF_month.index)
    #
    #
    # print('institutions_DF_year')
    # print(institutions_DF_year.index.year)
    # print(pd.DatetimeIndex(institutions_DF_year.index).year)

    fig = go.Figure()

    if institutional_period == 'Quarter':

        fig = go.Figure(data=[
            go.Bar(name='Quarterly', x=institutions_DF_month.index, y=institutions_DF_month['change']),
            # go.Bar(name='Equity', x=balance.date, y=equity)
        ])
    elif institutional_period == 'Year':
        fig = go.Figure(data=[
            go.Bar(name='Year', x=institutions_DF_year.index.year, y=institutions_DF_year['change']),
            # go.Bar(name='Equity', x=balance.date, y=equity)
        ])

    fig.update_layout(barmode='stack', title='Analysing Institutional Investor Transactions')

    return fig


def investor_transactions(data):
    # data = data.set_index('Trade Date')
    data.index = pd.to_datetime(data['Trade Date'])
    for i in range(0, len(data['Value'])):
        data['Value'][i] = float(data['Value'][i].replace('$', '').replace(',', ''))
    # print(data['Value'].resample('M').sum())
    # data = data.resample('M').sum()
    # print(data.resample('M').sum())
    # print(data['Value'])

    fig = go.Figure(data=[
        # go.Bar(name='Full', x=data.index.tolist(), y=data['Value']),
        # go.Bar(name='Equity', x=balance.date, y=equity)
        go.Bar(name='Q', x=data['Value'].resample('Q').sum().index.tolist(), y=data['Value'].resample('Q').sum()),
        # go.Bar(name='Equity', x=balance.date, y=equity)
    ])

    fig.add_trace(
        go.Scatter(
            x=data['Filing Date'],
            y=data['Value'],
            name="Investor transactions"
        )
    )

    fig.update_layout(barmode='stack', title='Analysing Inside Trading')

    return fig

# Визуализация портфолио


def cumRetPlotIdeal(cRDATA):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cRDATA.index,
            y=cRDATA,
            name='Cumulative Return'
        )
    )

    fig.update_layout(
        title="Cumulative Returns Calculated",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig


def cumRetPlot(data):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data,
            name='Cum Ret'
        )
    )

    fig.update_layout(
        title="Cumulative Returns Calculated",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        title_x=0.5,
        legend=dict(x=.5, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig
