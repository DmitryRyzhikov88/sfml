# Для запуска скрипта, написать команду "streamlit run stream_main.py"
# Должен быть установлен путь к папке где находится файл
# pip install torchvision

import datetime as dt
from datetime import datetime

import cvxpy
import empyrical as ep
import mibian
import numpy as np
import streamlit as st
import yfinance as yf
from hurst import compute_Hc
from pypfopt import DiscreteAllocation
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import shapiro
from vix_functions import vix_snp500, vix_gold, vix_euro, vix_russell, vix_emerg, vix_nasdaq, vix_china, fair_cost
from yahoofinancials import YahooFinancials

yf.pdr_override()

from visualization import *

ticker = st.sidebar.text_input(
    'Choose a Company',
    'INTC')

infoType = st.sidebar.radio(
    "Choose an info type",
    ('Fundamental', 'Technical', 'Valuation', 'Options', 'Statistics', 'Sentiment', 'Portfolio', 'Volatility Index')
)

KEY = '2105b9f242d47b69fc73f0f2205be048'
#KEY = 'd7c445cd5c6eaee5d52540f8a96d0e7f'

try:
    stock = yf.Ticker(ticker)
except:
    pass

if (infoType == 'Fundamental'):
    stock = yf.Ticker(ticker)
    info = stock.info
    st.title('Company Profile')
    st.subheader(info['longName'])
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown(
        '** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', ' + info['country'])
    st.markdown('** Website **: ' + info['website'])
    try:
        st.markdown('** Earnings Date **: ' + str(stock.calendar.loc['Earnings Date'][0]).split(' ')[0] + ' / '
                    + str(stock.calendar.loc['Earnings Date'][1]).split(' ')[0])
    except:
        pass
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])

    fundInfo = {
        'Enterprise Value (USD)': info['enterpriseValue'],
        'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
        'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
        'Net Income (USD)': info['netIncomeToCommon'],
        'Profit Margin Ratio': info['profitMargins'],
        'Forward PE Ratio': info['forwardPE'],
        'PEG Ratio': info['pegRatio'],
        'Price to Book Ratio': info['priceToBook'],
        'Forward EPS (USD)': info['forwardEps'],
        'Beta ': info['beta'],
        'Book Value (USD)': info['bookValue'],
        'Dividend Rate (%)': info['dividendRate'],
        'Dividend Yield (%)': info['dividendYield'],
        'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
        'Payout Ratio': info['payoutRatio']
    }

    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info')
    st.table(fundDF)

    st.subheader('General Stock Info')
    st.markdown('** Market **: ' + info['market'])
    st.markdown('** Exchange **: ' + info['exchange'])
    st.markdown('** Quote Type **: ' + info['quoteType'])

    start = dt.datetime.today() - dt.timedelta(2 * 365)
    end = dt.datetime.today()
    df = yf.download(ticker, start, end)
    df = df.reset_index()
    x = np.arange(df['Adj Close'].size)
    fit = np.polyfit(x, df['Adj Close'], deg=1)
    ff = np.poly1d(fit)
    fig = go.Figure(
        data=go.Scatter(x=df['Date'], y=df['Adj Close'], name='Stock Price')
    )
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=ff(x),
            name='Trend'
        )
    )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Two Years",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    st.write("Next Earning Date is: " + earning_date_plot(ticker))

    marketInfo = {
        "Volume": info['volume'],
        "Average Volume": info['averageVolume'],
        "Market Cap": info["marketCap"],
        "Float Shares": info['floatShares'],
        "Regular Market Price (USD)": info['regularMarketPrice'],
        'Bid Size': info['bidSize'],
        'Ask Size': info['askSize'],
        "Share Short": info['sharesShort'],
        'Short Ratio': info['shortRatio'],
        'Share Outstanding': info['sharesOutstanding']

    }

    st.write(' ')
    st.write('Market Data')

    marketDF = pd.DataFrame(data=marketInfo, index=[0])
    st.table(marketDF)

    # ========================              ЗАГРУЗКА ДАННЫХ С ФИНА      =============================

    IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={KEY}').json()
    KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={KEY}').json()
    CF = requests.get(
        f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={KEY}').json()
    CF_TTM = requests.get(
        f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=quarter&apikey={KEY}').json()
    PF = requests.get(f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={KEY}').json()
    EV = requests.get(f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?&apikey={KEY}').json()
    BL = requests.get(
        f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={KEY}').json()
    RT = requests.get(f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={KEY}').json()

    # ========================               ВИЗУАЛИЗАЦИЯ               =============================
    st.subheader('------ Visualization ------')

    # =========================== Waterfall ===========================

    st.markdown('Income Statement Waterfall')


    def incomeStatementfinancials(tickerz):
        ticker_yf = yf.Ticker(tickerz)
        incomeStatement_financials = pd.DataFrame(ticker_yf.financials)
        return incomeStatement_financials


    incomeStatement = incomeStatementfinancials(ticker)

    try:
        Revenue = incomeStatement.loc['Total Revenue'][0]
        COGS = incomeStatement.loc['Cost Of Revenue'][0] * (-1)
        grossProfit = incomeStatement.loc['Gross Profit'][0]
        RD = incomeStatement.loc['Research Development'][0] * (-1)
        GA = incomeStatement.loc['Selling General Administrative'][0] * (-1)
        operatingExpenses = incomeStatement.loc['Total Operating Expenses'][0] * (-1)
        interest = incomeStatement.loc['Interest Expense'][0] - 1
        EBT = incomeStatement.loc['Income Before Tax'][0]
        incTax = incomeStatement.loc['Income Tax Expense'][0] * -1
        netIncome = incomeStatement.loc['Net Income'][0]
    except:
        pass

    try:

        fig = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=["relative", "relative", "total", "relative", "relative", "total", "relative", "total", "relative",
                     "total"],

            x=["Revenue", "COGS", "Gross Profit", "RD", "G&A", "Operating Expenses", "Interest Expense",
               "Earn Before Tax",
               "Income Tax", "Net Income"],
            textposition="outside",

            text=[Revenue / 100000, COGS / 100000, grossProfit / 100000, RD / 100000, GA / 1000000,
                  operatingExpenses / 1000000,
                  interest / 100000, EBT / 100000, incTax / 100000, netIncome / 100000],

            y=[Revenue, COGS, grossProfit, RD, GA, operatingExpenses, interest, EBT, incTax, netIncome],

            connector={"line": {"color": "rgb(63, 63, 63)"}},

        ))

        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

    # Блок с графиками  из импорта

    st.subheader('Net Income Plot')
    st.write("In business and accounting, net income (also total comprehensive income, net earnings, net profit, bottom"
             " line, sales profit, or credit sales) is an entity's income minus cost of goods sold, expenses, "
             "depreciation and amortization, interest, and taxes for an accounting period.")
    try:
        size_net_income_plot = st.number_input('Insert period Net Income (Year): ', min_value=5, max_value=15, value=5)
        st.plotly_chart(net_income_plot(size_net_income_plot, IS), use_container_width=True)
    except:
        pass


    #st.subheader('Net Income 2')        st.plotly_chart(net_income_plot_2(ticker), use_container_width=True)
   #try:
   #     size_revenue_plot = st.number_input('Insert period Revenue (Year): ', min_value=5, max_value=15, value=5)
   #     st.plotly_chart(revenue_plot(size_revenue_plot, IS), use_container_width=True)
   # except:
   #     pass

    st.subheader('Revenue')
    st.write("Revenue is the total amount of income generated by the sale of goods and services related to the primary"
             " operations of the business. Commercial revenue may also be referred to as sales or as turnover")
    try:
        size_revenue = st.number_input('Insert period Revenue plot (Year): ', min_value=5, max_value=15, value=5)
        st.plotly_chart(revenue_plot(size_revenue, IS), use_container_width=True)
    except: pass

    st.subheader('Book Value')
    st.write("book value is the value of an asset according to its balance sheet account balance. For assets,"
             " the value is based on the original cost of the asset less any depreciation,"
             " amortization or impairment costs made against the asset.")
    try:
        size_book_value_years_plot = st.number_input('Insert period Book Value (Year): ', min_value=5, max_value=15,
                                                     value=5)
        st.plotly_chart(book_value_years_plot(size_book_value_years_plot, KM), use_container_width=True)
    except:
        pass

    st.subheader('Dividend per share')
    st.write("Dividend per share (DPS) is the sum of declared dividends issued by a company for every "
             "ordinary share outstanding.")
    try:
        size_dividend_pershare_plot = st.number_input('Insert period Dividend per Share (Year): ', min_value=5,
                                                      max_value=15, value=5)
        st.plotly_chart(dividend_pershare_plot(ticker, size_dividend_pershare_plot), use_container_width=True)
    except:
        pass

    st.subheader('Stock Price and PE')
    st.write("The price-earnings ratio, also known as P/E ratio, P/E, or PER, is the ratio of a company's share (stock)"
             " price to the company's earnings per share. The ratio is used for valuing companies and to find out"
             " whether they are overvalued or undervalued.")
    try:
        size_stock_price_PE = st.number_input('Insert period Stock Price and PE (Year): ', min_value=5, max_value=15,
                                              value=5)
        st.plotly_chart(stock_price_PE(size_stock_price_PE, IS, EV), use_container_width=True)
        st.plotly_chart(debt_to_assets(BL, EV), use_container_width=True)
    except: pass

    st.subheader('Dividends Payout')
    st.write("The dividend payout ratio is the fraction of net income a firm pays to its stockholders in dividends."
             " The part of earnings not paid to investors is left for investment to provide for future earnings growth."
             " Investors seeking high current income and limited capital growth prefer companies with a high dividend"
             " payout ratio.")
    try:
        size_dividends_payout_plot = st.number_input('Insert period Dividends Payout (Year): ', min_value=5,
                                                     max_value=15, value=5)
        st.plotly_chart(dividends_payout(size_dividends_payout_plot, RT), use_container_width=True)
    except: pass

    st.subheader('Major Multipliers')
    st.write("PE is the ratio of a company's share (stock) price to the company's earnings per share.\n"
             "Price–sales ratio, P/S ratio, or PSR, is a valuation metric for stocks. It is calculated by dividing the "
             "company's market capitalization by the revenue in the most recent year; or, equivalently, "
             "divide the per-share stock price by the per-share revenue.\n"
             "The price-to-book ratio, or P/B ratio, is a financial ratio used to compare a company's current market"
             " value to its book value.\n"
             "Price to free cash flow is an equity valuation metric used to compare a company's per-share market price"
             " to its per-share amount of free cash flow (FCF).\n"
             "Capital expenditure or capital expense (capex or CAPEX) is the money an organization or corporate entity"
             " spends to buy, maintain, or improve its fixed assets, such as buildings, vehicles, equipment, or land.")
    try:
        size_major_multipliers_plot = st.number_input('Insert period Major Multipliers (Year): ', min_value=5,
                                                      max_value=15, value=5)
        st.plotly_chart(major_multipliers(size_major_multipliers_plot, IS, EV, KM, CF), use_container_width=True)
    except: pass

    st.subheader('Revenue Operating')

    try:
        size_revenue_operating_income_fcf_plot = st.number_input('Insert period Revenue Operating Income FCF (Year): ',
                                                                 min_value=5, max_value=15, value=5)
        st.plotly_chart(revenue_operating_income_fcf(size_revenue_operating_income_fcf_plot, IS, CF),
                        use_container_width=True)
    except: pass

    st.subheader('Return of Assets')
    st.write("The return on assets (ROA) shows the percentage of how profitable a company's assets are in generating"
             " revenue.This number tells you what the company can do with what it has, i.e. how many dollars of "
             "earnings they derive from each dollar of assets they control.")
    try:
        size_roa_plot = st.number_input('Insert period ROA (Year): ', min_value=5, max_value=15, value=5)
        st.plotly_chart(roa(size_roa_plot, IS, BL), use_container_width=True)
    except: pass


    #try:
    #    Balance_Sheet = st.number_input('Insert period Balance Sheet (Quarter): ', min_value=1, max_value=150, value=4)
    #    st.plotly_chart(balance_sheet(ticker, Balance_Sheet), use_container_width=True)
    #except:
    #    pass


    st.subheader('Balance Assets and Liabs')
    st.write("Balance sheet assets are listed as accounts or items that are ordered by liquidity. "
             "Liquidity is the ease with which a firm can convert an asset into cash.\n"
             "Balance sheet liabilities are obligations the company has to other parties. They are classified as"
             " current liabilities (settled in less than 12 months) and non-current liabilities"
             " (settled in more than 12 months).")
    try:
        balance_bars_accets, balance_bars_debt = balance_bars(ticker)
        # Balance_bar = st.number_input('Insert period Balance Sheet (Year): ', min_value=3, max_value=150, value=4)
        st.plotly_chart(balance_bars_accets, use_container_width=True)
        st.plotly_chart(balance_bars_debt, use_container_width=True)
    except: pass


if (infoType == 'Technical'):

    def calcMovingAverage(data, size):
        df = data.copy()
        df['sma'] = df['Adj Close'].rolling(size).mean()
        df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df


    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df


    def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + 2 * df['Adj Close'].rolling(size).std(ddof=0)
        df["bold"] = df["sma"] - 2 * df['Adj Close'].rolling(size).std(ddof=0)
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df


    def pairCorrCalc(COMPAY1, COMPANY2, size):

        start = datetime.datetime(2022 - size_pt, 1, 1).date()
        end = datetime.datetime.today().date()
        DF1 = pd.DataFrame(yf.download([COMPAY1], start, end))
        DF1.dropna(inplace=True)
        DF2 = pd.DataFrame(yf.download([COMPANY2], start, end))
        DF2.dropna(inplace=True)
        pData = pd.DataFrame()
        pData['First Close'] = DF1['Close']
        pData['Second Close'] = DF2['Close']
        corr_mat = pData.corr(method='pearson')
        result = corr_mat['First Close'][1]
        'Correlation coefficient %.2f' % result

        #return corr_mat


    st.title('Technical Indicators')

    st.subheader('Stock price')

    try:
        size_stock = st.number_input("Insert period for stock's cost (Year): ", min_value=1, max_value=15, value=1,
                                     key=2)
        st.plotly_chart(act_cost(ticker, size_stock), use_container_width=True)
    except:
        pass


    # Скользящие средние

    try:
        st.subheader('Moving Average')

        # with coMA1:
        numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2)

        # with coMA2:
        windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=5)

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        dataMA = yf.download(ticker, start, end)
        df_ma = dataMA.reset_index()

        df_ma_20 = calcMovingAverage(dataMA, 20)
        df_ma_20 = df_ma_20.reset_index()

        df_ma_50 = calcMovingAverage(dataMA, 50)
        df_ma_50 = df_ma_50.reset_index()

        df_ma_200 = calcMovingAverage(dataMA, 200)
        df_ma_200 = df_ma_200.reset_index()

        df_ma_hand = calcMovingAverage(dataMA, windowSizeMA)
        df_ma_hand = df_ma_hand.reset_index()

        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x=df_ma['Date'],
                y=df_ma['Adj Close'],
                name="Prices Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma_20['Date'],
                y=df_ma_20['sma'],
                name="SMA" + str(20) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma_50['Date'],
                y=df_ma_50['sma'],
                name="SMA" + str(50) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma_200['Date'],
                y=df_ma_200['sma'],
                name="SMA" + str(200) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma_hand['Date'],
                y=df_ma_hand['sma'],
                name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        figMA.update_layout(legend_title_text='Trend')
        figMA.update_yaxes(tickprefix="$")

        st.plotly_chart(figMA, use_container_width=True)
    except:
        pass

    '''
    st.subheader('Volume')

    try:
        size_volume = st.number_input('Insert period for Volume (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(volume(ticker, size_volume), use_container_width=True)
    except: pass
    '''

    st.subheader('Candle pattern')

    try:
        size_candle = st.number_input('Insert period for Candle Pattern (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(candle(ticker, size_candle), use_container_width=True)
    except:
        pass

    st.subheader('Moving Average Convergence Divergence (MACD)')

    try:
        size_macd = st.number_input('Insert period MACD (Year): ', min_value=1, max_value=15, value=1, key=2)
        st.plotly_chart(macd(ticker, size_macd), use_container_width=True)
    except:
        pass

    st.subheader('RSI')

    try:
        size_rsi = st.number_input('Insert period RSI (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(rsi(ticker, size_rsi), use_container_width=True)
    except:
        pass

    st.subheader('Parabolic SAR')

    try:
        size_psar = st.number_input('Insert period PSAR (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(psar(ticker, size_psar), use_container_width=True)
    except:
        pass

    st.subheader('Agrmax points')

    try:
        size_agrmax = st.number_input('Insert period Agrmax points (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(agrmax(ticker, size_agrmax), use_container_width=True)
    except:
        pass

    st.subheader('Volatility Index (VIX)')

    try:
        size_vix = st.number_input('Insert period Volatility Index (Year): ', min_value=1, max_value=15, value=1)
        st.plotly_chart(vix(size_vix), use_container_width=True)
    except:
        pass

    st.subheader('Earning Date')

    try:
        st.plotly_chart(earning_date_plot(ticker), use_container_width=True)
    except: pass


    # Пробивание полос Болинжера

    # try:
    windowSizeBoll_signal = 5

    startBoll_signal = dt.datetime.today() - dt.timedelta(16)
    endBoll_signal = dt.datetime.today()
    dataBoll_signal = yf.download(ticker, startBoll_signal, endBoll_signal)
    df_boll_signal = calcBollinger(dataBoll_signal, windowSizeBoll_signal)
    df_boll_signal = df_boll_signal.reset_index()

    for k in range(len(df_boll_signal)):
        if df_boll_signal['Adj Close'][k] > df_boll_signal['bolu'][k]:
            boll_signal = 'Bull'
        if df_boll_signal['Adj Close'][k] < df_boll_signal['bold'][k]:
            boll_signal = 'Bear'
        else:
            boll_signal = 'Flat'

    # Полосы Болинжера

    st.subheader('Bollinger Band')

    st.markdown(f'Momentum in 10 days: __{boll_signal}__')

    coBoll1, coBoll2 = st.columns(2)
    with coBoll1:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key='1')

    with coBoll2:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key='2')

    startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker, startBoll, endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bolu'],
            name="Upper Band"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['sma'],
            name="SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bold'],
            name="Lower Band"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['Adj Close'],
            name="Adj Close"
        )
    )

    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))

    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)


    # Сигналы стадии рынка бычьи\медвежьи
    try:
        df_check = web.get_data_yahoo(ticker, start=f'{2021 - 2}-{1}-01', end=dt.datetime.today())
        df_check = df_check[['Open', 'High', 'Low', 'Close']]
        df_check['open'] = df_check['Open'].shift(1)
        df_check['high'] = df_check['High'].shift(1)
        df_check['low'] = df_check['Low'].shift(1)
        df_check['close'] = df_check['Close'].shift(1)

        df_check = df_check[['open', 'high', 'low', 'close']]
        df_check = df_check.dropna()

        unsup_check = mix.GaussianMixture(n_components=4,
                                          covariance_type="spherical",
                                          n_init=100,
                                          random_state=42)
        unsup_check.fit(np.reshape(df_check, (-1, df_check.shape[1])))
        regime_check = unsup_check.predict(np.reshape(df_check, (-1, df_check.shape[1])))

        first_chek = 0
        sec_check = 0
        for i in range(len(regime_check[::-1])):
            try:
                if regime_check[i] != regime_check[i + 1]:
                    sec_check = regime_check[i]
                    first_chek = regime_check[i + 1]
            except:
                pass

        if unsup_check.means_[first_chek][0] > unsup_check.means_[sec_check][0]:
            signal = 'Bull'
        else:
            signal = 'Bear'
    except:
        pass

    # Определение стадий рынка

    try:
        st.subheader('Market Stage')

        st.markdown(f'Signal is: __{signal}__')

        coBoll1, coBoll2 = st.columns(2)
        with coBoll1:
            numYearStage = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key='3')

        with coBoll2:
            monthYearStage = st.number_input('Insert period (Month): ', min_value=1, max_value=12, value=1, key='4')

        df = web.get_data_yahoo(ticker, start=f'{2021 - numYearStage}-{monthYearStage}-01', end=dt.datetime.today())
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
        fig22 = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, aspect=2, height=4)
        fig22.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
        plt.show()


        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # выводим средние значения таблица
        name_mean = []
        ean_value = []

        for i in order:
            name_mean.append(f'Mean for regime {i}:')
            ean_value.append(unsup.means_[i][0])

        mean_data = pd.DataFrame(columns=name_mean)
        # print(signal)
        for j in range(len(ean_value)):
            mean_data[f'Mean for regime {j}:'] = [ean_value[j]]
        # st.subheader('Fundamental Info')
        st.table(mean_data)

    except:
        pass


    st.subheader('Pairs Trading')

    try:
        size_pt = st.number_input('Insert period for Pairs Trading (Year): ', min_value=1, max_value=15, value=1)
        ticker_pair = st.text_input('Insert company ticker for Pairs Trading: ', value='AAPL')
        st.plotly_chart(pairTrade(ticker, ticker_pair, size_pt), use_container_width=True)
        pairCorrCalc(ticker, ticker_pair, size_pt)
    except:
        pass


if (infoType == 'Valuation'):

    st.header('General company information')
    ticker_input = st.text_input('Please enter your company ticker here:', ticker)
    yahoo_financials = YahooFinancials(ticker_input)

    IS = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker_input}?apikey={KEY}').json()
    KM = requests.get(f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker_input}?apikey={KEY}').json()
    CF = requests.get(
        f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker_input}?apikey={KEY}').json()
    CF_TTM = requests.get(
        f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker_input}?period=quarter&apikey={KEY}').json()
    PF = requests.get(f'https://financialmodelingprep.com/api/v3/profile/{ticker_input}?apikey={KEY}').json()
    EV = requests.get(f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker_input}?&apikey={KEY}').json()
    BL = requests.get(
        f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker_input}?apikey={KEY}').json()
    RT = requests.get(f'https://financialmodelingprep.com/api/v3/ratios/{ticker_input}?apikey={KEY}').json()

    ####################################################################################################################
    #### Revenue Growth
    is_data = pd.DataFrame().from_dict(IS).fillna(0)
    revenue = is_data['revenue']
    rev_s = revenue.pct_change()
    std = rev_s.std() * 100
    cagr = ((revenue[0] / revenue[4]) ** .2) - 1
    cagr = cagr * 100

    ### Expected Margin Change
    margin = is_data['netIncome'] / revenue
    emc = np.std(margin) * 100

    ### Current EPS
    eps = yahoo_financials.get_earnings_per_share()

    ### CAD
    dividend = pd.DataFrame(yf.Ticker(ticker).dividends)
    tmp = dividend.iloc[len(dividend)-1]
    cad = tmp[0]

    ### Expected Payout Ratio Changes
    ratios = pd.DataFrame().from_dict(RT).fillna(0)
    payout_ratio = ratios['payoutRatio']
    eprc = payout_ratio.std() * 100

    ### PE10
    income_statement = pd.DataFrame().from_dict(IS).fillna(0)
    enterprise_value = pd.DataFrame().from_dict(EV).fillna(0)
    PE_year = enterprise_value['marketCapitalization'] / income_statement['netIncome']

    pe10 = np.mean(PE_year)
    ####################################################################################################################

    col11, col12, col13 = st.columns(3)

    with col11:
        st.subheader('Revenue growth')
        mean_input = (st.number_input('Mean revenue growth rate (in %)', value=cagr))
        stddev_input = (st.number_input('Revenue growth rate std. dev. (in %)', value=std))

    with col12:
        st.subheader('Expected margin change')
        expected_margin_change = (st.number_input('Expected margin change (in %)', value=emc))

    with col13:
        st.subheader('Inflation')
        price_growth = (st.number_input('Inflation (in %)', value=3))

    col21, col22, col23 = st.columns(3)

    with col21:
        st.subheader('Current year EPS')
        year_EPS = (st.number_input('EPS', value=eps))

    with col22:
        st.subheader('Expected reduction in the number of shares')
        expected_reduction = (st.number_input('Expected reduction in the number of shares (in %)'))

    with col23:
        st.subheader('Current annual dividend')
        annual_dividend = (st.number_input('Dividends', value=cad))

    col31, col32, col33 = st.columns(3)

    with col31:
        st.subheader('Expected Payout Ratio Changes')
        payout_ratio_changes = (st.number_input('Payout Ratio  (in %)', eprc))

    with col32:
        st.subheader('Expected PE in 10 years')
        mean_PE = (st.number_input('Mean PE in 10 years', pe10))

    simulation_iterations = 1000
    discount_rate = 8

    #                   Функция для расчета данных и записи их в датасет


    def calculation(revenue_growth_normal, price_growth, expected_margin_change, expected_reduction,
                    payout_ratio_changes,
                    year_EPS, annual_dividend, mean_PE, discount_rate):
        final_data = pd.DataFrame()

        # Expected Revenue Growth
        expected_revenue_growth = revenue_growth_normal + price_growth
        # print('expected_revenue_growth', expected_revenue_growth)
        final_data['Expected Revenue Growth'] = [expected_revenue_growth]
        # Expected net income growth
        expected_net_income_growth = (1 + expected_revenue_growth) * (1 + expected_margin_change) - 1
        # print('expected_net_income_growth', expected_net_income_growth)
        final_data['Expected net income growth'] = [expected_net_income_growth]
        # EPS Growth Rate
        EPS_growth_rate = (1 + expected_net_income_growth) / (1 - expected_reduction) - 1
        # print('EPS_growth_rate', EPS_growth_rate)
        final_data['EPS Growth Rate'] = [EPS_growth_rate]
        # Expected dividend growth

        expected_dividend_growth = (1 + EPS_growth_rate) * (1 + payout_ratio_changes) - 1
        # print('expected_dividend_growth', expected_dividend_growth)
        final_data['Expected dividend growth'] = [expected_dividend_growth]

        eps_list = [year_EPS * (1 + EPS_growth_rate)]
        dps_list = [annual_dividend * (1 + expected_dividend_growth)]

        for i in range(9):
            eps_list.append(eps_list[-1] * (1 + EPS_growth_rate))
            dps_list.append(dps_list[-1] * (1 + expected_dividend_growth))

        # Share price in 10 years

        share_price_in_10_years = eps_list[-1] * mean_PE
        # print('share_price_in_10_years', share_price_in_10_years)
        final_data['Share price in 10 years'] = [share_price_in_10_years]
        # Dividend yield in 10 years

        dividend_yield_in_10_years = dps_list[-1] / share_price_in_10_years
        # print('dividend_yield_in_10_years', dividend_yield_in_10_years)
        final_data['Dividend yield in 10 years'] = [dividend_yield_in_10_years]

        # Discounted Dividend Amount
        discounted_dividend_amount_list = [dps_list[-1]]

        for i in range(len(dps_list) - 1):
            discounted_dividend_amount_list.append(dps_list[::-1][i + 1] * (1 + discount_rate) ** (i + 1))

        discounted_dividend_amount = np.sum(discounted_dividend_amount_list)
        # print('discounted_dividend_amount', discounted_dividend_amount)
        final_data['Discounted Dividend Amount'] = [discounted_dividend_amount]

        # Total Value in 10 years
        total_value_in_10_years = share_price_in_10_years + discounted_dividend_amount
        # print('total_value_in_10_years', total_value_in_10_years)
        final_data['Total Value in 10 years'] = [total_value_in_10_years]
        # Current buy price (fundamentally calculated)
        current_buy_price = total_value_in_10_years / (1 + discount_rate) ** 10
        final_data['Current buy price (fundamentally calculated)'] = [current_buy_price]

        # print('current_buy_price', current_buy_price)

        return final_data


    # ~~~~~~~~~~~~~~~~~~~~~  Расчетная часть  ~~~~~~~~~~~~
    status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))
    if status_radio == 'Search':

        revenue_growth_normal_list = []

        for i in range(int(simulation_iterations)):
            revenue_growth_normal_list.append(np.random.normal(mean_input, stddev_input))

        revenue_growth_normal = [np.max(revenue_growth_normal_list), np.min(revenue_growth_normal_list)]

        # print(np.max(revenue_growth_normal_list))
        # print(np.min(revenue_growth_normal_list))
        st.write('Minimum')
        final_data_min = calculation(np.min(revenue_growth_normal_list), price_growth, expected_margin_change, \
                                     expected_reduction, payout_ratio_changes, year_EPS, annual_dividend, mean_PE,
                                     discount_rate)
        st.dataframe(final_data_min.T.rename({0: 'Value'}, axis=1).round(3))

        st.write('Maximum')
        final_data_max = calculation(np.max(revenue_growth_normal_list), price_growth, expected_margin_change, \
                                     expected_reduction, payout_ratio_changes, year_EPS, annual_dividend, mean_PE,
                                     discount_rate)
        st.dataframe(final_data_max.T.rename({0: 'Value'}, axis=1).round(3))

        st.write('Mean')
        final_data_max = calculation(np.mean(revenue_growth_normal_list), price_growth, expected_margin_change, \
                                     expected_reduction, payout_ratio_changes, year_EPS, annual_dividend, mean_PE,
                                     discount_rate)
        st.dataframe(final_data_max.T.rename({0: 'Value'}, axis=1).round(3))

# ----------------- Калькулятор продажи PUT

if (infoType == 'Options'):
    st.header('PUT sale calculator')

    current_price = yf.download(ticker).iloc[-1]['Close']

    col11, col12, col13, col14 = st.columns(4)

    with col11:
        st.subheader('Strike PUT')
        strike_put = (st.number_input('Strike PUT'))

    with col12:
        st.subheader('Contract price (Premium) PUT')
        premium = (st.number_input('Contract price (Premium) PUT'))

    with col13:
        st.subheader('Days to expiration PUT')
        days_to_expiration = (st.number_input('Days to expiration PUT'))

    with col14:
        st.subheader('Dividends before expiration PUT')
        dividends_before_expiration = (st.number_input('Dividends before expiration'))

    # ================= calculated

    status_radio = st.radio('Please click Calculated PUT when you are ready.', ('Entry', 'Calculated PUT'))
    if status_radio == 'Calculated PUT':
        try:
            profit_for_the_period = premium / (strike_put - premium)
            st.write('Profit for the period = ', round(profit_for_the_period, 3), '%')

            annual_percentage_yield = (profit_for_the_period / days_to_expiration) * 365
            st.write('Annual percentage yield = ', round(annual_percentage_yield, 3) * 100, '%')

            purchase_price = strike_put - premium
            st.write('Purchase price (if executed) = ', round(purchase_price, 3))

            discount_current_market_price = (current_price - purchase_price) / current_price
            st.write('Discount to the current market price = ', round(discount_current_market_price, 2) * 100, '%')
        except ZeroDivisionError:
            st.write("Enter correct values please.")

    # ----------------- Калькулятор продажи CALL

    st.header('CALL sale calculator')

    current_price = yf.download(ticker).iloc[-1]['Close']

    col11, col12, col13, col14 = st.columns(4)

    with col11:
        st.subheader('Strike CALL')
        strike_call = (st.number_input('Strike CALL'))

    with col12:
        st.subheader('Contract price (Premium) CALL')
        premium = (st.number_input('Contract price (Premium) CALL'))

    with col13:
        st.subheader('Days to expiration CALL')
        days_to_expiration = (st.number_input('Days to expiration CALL'))

    with col14:
        st.subheader('Dividends before expiration CALL')
        dividends_before_expiration = (st.number_input('Dividends before expiration CALL'))

    # ================= calculated

    status_radio = st.radio('Please click Calculated CALL when you are ready.', ('Entry', 'Calculated CALL'))
    if status_radio == 'Calculated CALL':
        try:
            st.subheader('If not done')

            profit_for_the_period = premium / current_price
            st.write('Profit for the period = ', round(profit_for_the_period, 2), '%')

            annual_percentage_yield = (profit_for_the_period / days_to_expiration) * 365
            st.write('Annual percentage yield = ', round(annual_percentage_yield, 2), '%')

            st.subheader('If done')

            profit_for_the_period_done = (strike_call + premium + dividends_before_expiration) / current_price - 1
            st.write('Profit for the period = ', round(profit_for_the_period_done, 2), '%')

            annual_percentage_yield_done = (profit_for_the_period_done / days_to_expiration) * 365
            st.write('Annual percentage yield = ', round(annual_percentage_yield_done, 2) * 100, '%')
        except ZeroDivisionError:
            st.write("Enter correct values please")

    # ----------------- Калькулятор теоретической цены и греков

    st.header('Theoretical price and Greeks')

    status_radio_greek = st.radio('Please select option type:', ('PUT', 'CALL'))

    current_price = yf.download(ticker).iloc[-1]['Close']

    col11, col12, col13, col14 = st.columns(4)

    with col11:
        st.subheader('Strike option')
        strike_option = (st.number_input('Strike option'))

    with col12:
        st.subheader('Contract price (Premium) option')
        premium = (st.number_input('Contract price (Premium) option'))

    with col13:
        st.subheader('Days to expiration option')
        days_to_expiration = (st.number_input('Days to expiration option'))

    with col14:
        st.subheader('Risk free rate')
        dividends_before_expiration = (st.number_input('Risk free rate'))

    st.subheader('Implied Volatility')
    implied_volatility = (st.number_input('Implied Volatility'))

    # ================= calculated
    status_radio = st.radio('Please click Calculated when you are ready.', ('Entry values', 'Calculated'))

    greeks = mibian.BS([current_price, strike_option, 0, days_to_expiration],
                       volatility=implied_volatility * 100)

    if status_radio == 'Calculated':
        if status_radio_greek == 'PUT':
            option_price = greeks.putPrice
            delta = greeks.putDelta
            theta = greeks.putTheta
            callPrice_formula = 'putPrice'
        elif status_radio_greek == 'CALL':
            option_price = greeks.callPrice
            delta = greeks.callDelta
            theta = greeks.callTheta
            callPrice_formula = 'callPrice'

        st.write('Contract price = ', round(option_price, 2))
        st.write('Delta = ', round(delta, 2))
        st.write('Gamma = ', round(greeks.gamma, 2))
        st.write('Theta = ', round(theta, 2))
        st.write('Vega = ', round(greeks.vega, 2))

    # ================= калькулятор impliedVolatility

    st.header('Implied Volatility calculator')

    current_price = yf.download(ticker).iloc[-1]['Close']

    status_radio_iv = st.radio('Please select option type: ', ('PUT', 'CALL'))

    col11, col12, col13 = st.columns(3)

    with col11:
        st.subheader('Strike')
        strike = (st.number_input('Strike'))

    with col12:
        st.subheader('Days to expiration')
        days_to_expiration = (st.number_input('Days to expiration'))

    with col13:
        st.subheader('Option Price')
        option_price = (st.number_input('Option Price'))

    status_radio = st.radio('Please click Calculated when you are ready.', ('Entry', 'Calculated'))

    if status_radio == 'Calculated':

        if status_radio_iv == 'PUT':
            impliedVolatility = mibian.BS([current_price, strike, 0, days_to_expiration],
                                          putPrice=option_price).impliedVolatility
        if status_radio_iv == 'CALL':
            impliedVolatility = mibian.BS([current_price, strike, 0, days_to_expiration],
                                          callPrice=option_price).impliedVolatility

        st.write('Implied Volatility = ', round(impliedVolatility, 2) * 100, '%')


if (infoType == 'Statistics'):
    # показатель Херста
    try:
        def hurst_coef(ticker, size):
            # Задаем диапазон дат в котором нужно собирать все данные по тикерам
            start = dt.datetime(2010, 1, 1)
            end = dt.datetime.now()  # сегодняшняя дата, чтобы не менять вручную.
            # Получаем данные из Yahoo. Именно этот способ позволяет получить данные с тикерами в столбцах.
            df = web.get_data_yahoo(ticker, start, end)
            start_size = dt.datetime.today() - dt.timedelta(days=size)

            # print('start_size', start_size)
            start_size_str = str(start_size).split(' ')[0]

            date_hurts = df.loc[start_size_str:]['Close']
            # print(date_hurts)
            'Hurst Coefficient %.2f' % compute_Hc(date_hurts, kind='price')[0]

            if compute_Hc(date_hurts, kind='price')[0] == 0.5:
                'Random walk price'
            elif compute_Hc(date_hurts, kind='price')[0] > 0.5:
                'Trendiness'
            elif compute_Hc(date_hurts, kind='price')[0] < 0.5:
                'Return to mean'

            # return description

    except:
        pass

    try:
        size_hurst_coef = st.number_input('Insert period for Hurst Сoefficient (Day): ', min_value=150, max_value=10000,
                                          value=150)
        hurst_coef(ticker, size_hurst_coef)
    except:
        pass

    st.header('Distributions')
    num_distributions = st.number_input('Insert period of Distribution (Year): ', min_value=1, max_value=25, value=5)

    start = dt.datetime.today() - dt.timedelta(num_distributions * 365)
    end = dt.datetime.today()

    # Daily Distribution

    try:
        ITC = yf.download(ticker, start, end)[['Adj Close']]

        daily_data = ITC.copy().round(4)

        # Calculating daily Log returns
        daily_data['daily_return'] = np.log(daily_data['Adj Close'] / daily_data['Adj Close'].shift())
        daily_data.dropna(inplace=True)

        # Visualizing the daily log returns
        sns.set(style="white", palette="muted", color_codes=True)
        fig2 = plt.figure()

        plt.subplot(1, 2, 1)
        # Plot a simple histogram with binsize determined automatically
        sns.lineplot(daily_data.index, daily_data['daily_return'], color="r")
        plt.ylabel('Actual values of daily_returns')
        plt.title('Actual values of Daily log returns over time')

        plt.subplot(1, 2, 2)
        # Plot a simple histogram with binsize determined automatically
        sns.distplot(daily_data['daily_return'], kde=False, color="r")
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.title('Distribution of Daily log returns')

        st.pyplot(fig2)

        # Normal check

        stat1, p1 = shapiro(daily_data['daily_return'])

        if p1 < 0.01:
            test_result1 = 'NOT NORMAL'
            st.write(test_result1)
            st.write(f'The p-value is {p1}, data is not normally distributed with 99% confidence')

        else:
            test_result1 = 'NORMAL'
            st.write(test_result1)
            st.write(f'The p-value is {p1}, data is normally distributed with 99% confidence')
    except:
        pass

    # Weekly Distribution
    st.write('-' * 50)

    try:
        # Resampling to get weekly Close-Close and weekly return data
        weekly_data = ITC.resample('W').last()
        weekly_data['weekly_return'] = np.log(weekly_data['Adj Close'] / weekly_data['Adj Close'].shift())
        weekly_data.dropna(inplace=True)

        # Plotting the actual values and the distribution of weekly returns
        sns.set(style="white", palette="muted", color_codes=True)
        fig_distribution_weekly = plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(weekly_data.index, weekly_data['weekly_return'], color="g")
        plt.ylabel('Actual values of Weekly_returns')
        plt.title('Actual values of Weekly log returns over time')

        plt.subplot(1, 2, 2)
        sns.distplot(weekly_data['weekly_return'], kde=False, color="g")
        plt.ylabel('Frequency')
        plt.title('Distribution of Weekly log returns')

        plt.tight_layout()

        st.pyplot(fig_distribution_weekly)

        # Normal check

        stat2, p2 = shapiro(weekly_data['weekly_return'])

        if p2 < 0.01:
            test_result2 = 'NOT NORMAL'
            st.write(test_result2)
            st.write(f'The p-value is {p2}, data is not normally distributed with 99% confidence')

        else:
            test_result2 = 'NORMAL'
            st.write(test_result2)
            st.write(f'The p-value is {p2}, data is normally distributed with 99% confidence')

    except:
        pass

    # Monthly Distribution
    st.write('-' * 50)

    try:
        # Resampling to get monthly Close-Close and monthly return data
        monthly_data = ITC.resample('M').last()
        monthly_data['monthly_return'] = np.log(monthly_data['Adj Close'] / monthly_data['Adj Close'].shift())
        monthly_data.dropna(inplace=True)
        monthly_data.head()

        # Plotting the actual values and the distribution of monthly returns
        sns.set(style="white", palette="muted", color_codes=True)

        fig_distribution_monthly = plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(monthly_data.index, monthly_data['monthly_return'], color="b")
        plt.ylabel('Actual values of monthly_returns')
        plt.title('Actual values of monthly log returns over time')

        plt.subplot(1, 2, 2)
        sns.distplot(monthly_data['monthly_return'], kde=False, color="b")
        plt.ylabel('Frequency')
        plt.title('Distribution of monthly log returns over time')

        plt.tight_layout()
        st.pyplot(fig_distribution_monthly)

        # Normal check

        stat3, p3 = shapiro(monthly_data['monthly_return'])

        if p3 < 0.01:
            test_result3 = 'NOT NORMAL'
            st.write(test_result3)
            st.write(f'The p-value is {p3}, data is not normally distributed with 99% confidence')

        else:
            test_result3 = 'NORMAL'
            st.write(test_result3)
            st.write(f'The p-value is {p3}, data is normally distributed with 99% confidence')
    except:
        pass

    # Annually Distribution

    st.write('-' * 50)

    try:

        # Resampling to get monthly Close-Close and monthly return data
        annually_data = ITC.resample('Y').last()
        print(annually_data)
        annually_data['annually_return'] = np.log(annually_data['Adj Close'] / annually_data['Adj Close'].shift())
        annually_data.dropna(inplace=True)

        # Plotting the actual values and the distribution of monthly returns
        sns.set(style="white", palette="muted", color_codes=True)

        fig_distribution_annually = plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(annually_data.index, annually_data['annually_return'], color="b")
        plt.ylabel('Actual values of monthly_returns')
        plt.title('Actual values of monthly log returns over time')

        plt.subplot(1, 2, 2)
        sns.distplot(annually_data['annually_return'], kde=False, color="b")
        plt.ylabel('Frequency')
        plt.title('Distribution of monthly log returns over time')

        plt.tight_layout()
        st.pyplot(fig_distribution_annually)

        # Normal check

        stat4, p4 = shapiro(annually_data['annually_return'])

        if p4 < 0.01:
            test_result4 = 'NOT NORMAL'
            st.write(test_result4)
            st.write(f'The p-value is {p4}, data is not normally distributed with 99% confidence')

        else:
            test_result4 = 'NORMAL'
            st.write(test_result4)
            st.write(f'The p-value is {p4}, data is normally distributed with 99% confidence')

    except:
        pass


if (infoType == 'Sentiment'):

    #st.header('Company Earnings Sentiment Analysis')
    #transcript = requests.get(
    #    f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?apikey={KEY}').json()

    # print(transcript)
    # print(transcript[0]['content'])
    #transcript = transcript[0]['content']

    #sentiment_call = TextBlob(transcript)

    # print(sentiment_call.sentiment)
    # print(sentiment_call.sentences)

    # sentiment_call.sentences
    #negative = 0
    #positive = 0
    #neutral = 0
    #all_sentences = []

    #for sentence in sentiment_call.sentences:
        # print(sentence.sentiment.polarity)
    #    if sentence.sentiment.polarity < 0:
    #        negative += 1
    #    if sentence.sentiment.polarity > 0:
    #        positive += 1
    #    else:
    #        neutral += 1

    #    all_sentences.append(sentence.sentiment.polarity)

    #st.write('Positive: ' + str(positive))
    #st.write('Negative: ' + str(negative))
    #st.write('Neutral: ' + str(neutral))

    #all_sentences = np.array(all_sentences)
    # print('sentence polarity: ' + str(all_sentences.mean()))
    #st.write('Sentence polarity: ' + str(all_sentences.mean()))

    #sentence_positive = ''
    #sentence_negative = ''

    #st.header('Sentiment Call Sentences:')
    #for sentence in sentiment_call.sentences:
    #    if sentence.sentiment.polarity > 0.8:
            # print(type(sentence))
    #        sentence_positive = str(sentence_positive) + '\n' + str(sentence) + '\n'

            # st.write(sentence)

    #    elif sentence.sentiment.polarity < -0.1:
            # print(type(sentence))
    #        sentence_negative = str(sentence_negative) + '\n' + str(sentence) + '\n'
            # print(sentence)
            # st.write(sentence)

    #with st.beta_expander('Positive'):
    #    st.subheader('Positive')
    #    st.write(sentence_positive)

    #with st.beta_expander('Negative'):
    #    st.subheader('Negative')
    #    st.write(sentence_negative)

    # Analysing Institutional Investor Transactions

    st.header('Analysing Inside Trading')

    insider = pd.DataFrame()
    insider1 = pd.read_html(
        f'http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=730&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=500&page=1')
    pd.DataFrame(insider1).to_csv('insider.csv')
    # print(insider1)
    insider1 = insider1[-3]
    # print(insider1)
    st.dataframe(insider1.drop(columns=['X', 'Trade Date', 'Ticker', '1d', '1w', '1m', '6m']))

    st.plotly_chart(investor_transactions(insider1), use_container_width=True)

    #st.header('Analysing Institutional Investor Transactions')

    #institutional_period = st.radio('Please choose the period', ('Year', 'Quarter'))

    #try:
        # institutional_investor_value = st.number_input('Analysing Institutional Investor Transactions (Year): ', min_value=5, max_value=15, value=5, key='21')
        #st.plotly_chart(institutional_investor(ticker, institutional_period), use_container_width=True)
    #except:
    #    pass

    #st.header('Sentiment Interest By Scored News')
    try:

        st.plotly_chart(scored_news(ticker), use_container_width=True)
    except:
        pass

    # TWITTER

    #st.header('TWITTER')

    #stock = yf.Ticker(ticker)
    #info = stock.info

    #with open('bearer_token.txt') as fp:
    #    BEARER_TOKEN = fp.read()
    # ticker = 'AAPL'
    #twitter_list = []

    #sentiment_model = flair.models.TextClassifier.load('en-sentiment')

    #company = info['longName']
    #     try:
    #print('*' * 40)
    #print(company.lower().replace("&", ""))


    #def get_data(tweet):
    #    data = {
    #        'id': tweet['id'],
    #        'created_at': tweet['created_at'],
    #        'text': tweet['text']
    #    }
    #    return data


    # read bearer token for authentication
    #with open('bearer_token.txt') as fp:
    #    BEARER_TOKEN = fp.read()

    # setup the API request
    #endpoint = 'https://api.twitter.com/2/tweets/search/recent'
    #headers = {'authorization': f'Bearer {BEARER_TOKEN}'}
    #params = {
    #    'query': f'({company.lower().replace("&", "")}) (lang:en)',
    #    'max_results': '100',
    #    'tweet.fields': 'created_at,lang'
    #}

    #dtformat = '%Y-%m-%dT%H:%M:%SZ'  # the date format string required by twitter


    # we use this function to subtract 60 mins from our datetime string
    #def time_travel(now, mins):
    #    now = dt.datetime.strptime(now, dtformat)
    #    back_in_time = now - timedelta(minutes=mins)
    #    return back_in_time.strftime(dtformat)


    #now = dt.datetime.now() - timedelta(minutes=200)  # get the current datetime, this is our starting point
    #last_week = now - timedelta(days=6)  # datetime one week ago = the finish line
    #now = now.strftime(dtformat)  # convert now datetime to format for API

    #df = pd.DataFrame()  # initialize dataframe to store tweets

    #while True:

        #if dt.datetime.strptime(now, dtformat) < last_week:
            # if we have reached 7 days ago, break the loop
        #    break
        #pre60 = time_travel(now, 720)  # get 60 minutes before 'now'

        #     print(last_week)
        #     print(now_new)
        #     print(now)

        # assign from and to datetime parameters for the API
        #params['start_time'] = pre60
        #params['end_time'] = now
        #sleep(2)
        #response = requests.get(endpoint,
        #                        params=params,
        #                        headers=headers)  # send the request
        #         print(response)
        #now = pre60  # move the window 60 minutes earlier

        #try:
            #             print(response.json())
            #for tweet in response.json()['data']:
            #    #                 print(tweet)
            #    row = get_data(tweet)  # we defined this function earlier
            #    df = df.append(row, ignore_index=True)

        #             response.json()
        #except:
        #    continue

    #df.drop_duplicates(subset='text', inplace=True)

    # ===============================================  Получаем количественную оценку с массива ====================================================

    #     try:
    #sentiment = []
    #confidance = []

    #print('Please wait')
      #         print(df)
    #try:
    #    for sentence in df['text']:
    #        if sentence.strip() == "":
    #            sentiment.append("")
    #            confidance.append("")
    #        else:
    #            sample = flair.data.Sentence(sentence)
    #            sentiment_model.predict(sample)
                #             print('Please wait')
                #         print(sample.labels[0].value)
                #         print(sample.labels[0].score)
    #            if sample.labels[0].value == 'NEGATIVE':
    #                confidance.append(-1)
    #            else:
    #                confidance.append(1)

        # print(f'confidance: {np.mean(confidance)}')
        # print('*' * 40)
    #    twitter_list.append(np.mean(confidance))
    #except Exception as tttttt:
        # print(f'confidance: {tttttt}')/
    #    twitter_list.append('nan')
        # print(f'confidance: nan')
        # print('*' * 40)

    #st.write('Confidance: ', round(twitter_list[0], 2) * 100, '%')


if (infoType == 'Portfolio'):

    st.header('Portfolio Probability')

    tickers = st.text_input('Insert Portfolio Composition: ', value='AMGN, GILD, MRNA, REGN, VRTX')
    strTickers = tickers.split(', ')
    start_size = st.text_input('Insert start of period for Analysis (YYYY-MM-DD): ',
                                    value='2021-01-01')
    end_size = st.text_input('Insert end of period for Analysis (YYYY-MM-DD): ',
                                  value='2022-01-01')



    def cumProbCount(tickers, start, end):

        ########################################

        benchmark = ['EZA']

        ohlc = yf.download(tickers, start, end)
        prices = ohlc["Close"].dropna()
        latest_prices = prices.iloc[0].reindex(tickers)

        df = pd.DataFrame()
        df['tickers'] = tickers
        df['price'] = latest_prices.tolist()
        currencies = ['CADUSD=X', 'AUDUSD=X', 'TRYUSD=X', 'CHFUSD=X', 'MYRUSD=X', 'MXNUSD=X', 'GBPUSD=X', 'SGDUSD=X', \
                      'HKDUSD=X', 'ZARUSD=X', 'ILSUSD=X', 'JPYUSD=X', 'RUBUSD=X', 'EURUSD=X']
        df_currencies = pd.DataFrame(yf.download(currencies, start, end)['Close'])
        df_currencies = df_currencies.iloc[1]
        for i in range(len(df)):
            cur = 'USD'
            if ".TO" in df['tickers'][i]:
                cur = 'CADUSD=X'
            elif '.V' in df['tickers'][i]:
                cur = 'CADUSD=X'
            elif '.SW' in df['tickers'][i]:
                cur = 'CHFUSD=X'
            elif '.IS' in df['tickers'][i]:
                cur = 'TRYUSD=X'
            elif '.KL' in df['tickers'][i]:
                cur = 'MYRUSD=X'
            elif '.L:' in df['tickers'][i]:
                cur = 'GBPUSD=X'
            elif '.SI' in df['tickers'][i]:
                cur = 'SGDUSD=X'
            elif '.HK' in df['tickers'][i]:
                cur = 'HKDUSD=X'
            elif '.AX' in df['tickers'][i]:
                cur = 'AUDUSD=X'
            elif '.JO' in df['tickers'][i]:
                cur = 'ZARUSD=X'
            elif '.TA' in df['tickers'][i]:
                cur = 'ILSUSD=X'
            elif '.T' in df['tickers'][i]:
                cur = 'JPYUSD=X'
            elif '.ME' in df['tickers'][i]:
                cur = 'RUBUSD=X'
            elif '.MX' in df['tickers'][i]:
                cur = 'MXNUSD=X'
            elif '.MС' in df['tickers'][i]:
                cur = 'EURUSD=X'
            elif '.F' in df['tickers'][i]:
                cur = 'EURUSD=X'
            elif '.AS' in df['tickers'][i]:
                cur = 'EURUSD=X'
            elif '.BR' in df['tickers'][i]:
                cur = 'EURUSD=X'
            elif '.PA' in df['tickers'][i]:
                cur = 'EURUSD=X'
            elif '.MI' in df['tickers'][i]:
                cur = 'EURUSD=X'
            if cur == 'USD':
                df['price'][i] = df['price'][i] * 1
            elif cur == 'GBPUSD=X':
                df['price'][i] = df['price'][i] * df_currencies[cur] / 100
            elif cur == 'ZARUSD=X':
                df['price'][i] = df['price'][i] * df_currencies[cur] / 100
            else:
                df['price'][i] = df['price'][i] * df_currencies[cur]
        df.head()

        #### Если цена портфеля мала, то увеличиваем этот показатель и перезапускаем ячейку
        multiplicator = 1
        #################

        weights = []
        for i in range(len(df)):
            a = round(df['price'].max() / df['price'][i]) * multiplicator
            weights.append(a)
        df['weights'] = weights
        df['weights'] = df['weights'].astype(int)
        df['equity'] = df['weights'] * df['price']
        df['% of portfolio'] = round(df['equity'] / df['equity'].sum(), 3)
        print(df)
        print('')
        print('Бюджет портфеля', round(df['equity'].sum()))
        total_price = round(df['equity'].sum())
        'Бюджет портфеля   %.2f' % round(df['equity'].sum())

        porfolio_price = []

        portfolio = tickers
        quantity = df['weights'].tolist()

        df = pd.DataFrame(columns=portfolio)
        for ticker in portfolio:
            df[ticker] = yf.download(ticker, start, end, progress=False)['Close']
            index_df = yf.download(benchmark, start, end, progress=False)["Close"]
            df = df.fillna(0)

        currency = ['CADUSD=X', 'AUDUSD=X', 'TRYUSD=X', 'CHFUSD=X', 'MYRUSD=X', 'MXNUSD=X', 'GBPUSD=X', 'SGDUSD=X',
                    'HKDUSD=X', 'ZARUSD=X', 'ILSUSD=X', 'JPYUSD=X', 'RUBUSD=X', 'EURUSD=X']
        df_currency = pd.DataFrame()

        for currency in currency:
            df_currency[currency] = yf.download(currency, start, end, progress=False)['Close']

        for column in range(len(df.columns.tolist())):
            if ".V" in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['CADUSD=X']
            elif '.TO' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['CADUSD=X']
            elif '.IS' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['TRYUSD=X']
            elif '.SW' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['CHFUSD=X']
            elif '.KL' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['MYRUSD=X']
            elif '.MX' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['MXNUSD=X']
            elif '.L' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['GBPUSD=X']
            elif '.SI' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['SGDUSD=X']
            elif '.HK' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['HKDUSD=X']
            elif '.AX' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['AUDUSD=X']
            elif '.JO' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['ZARUSD=X'] / 100
            elif '.TA' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['ILSUSD=X']
            elif '.T' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['JPYUSD=X']
            elif '.ME' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['RUBUSD=X']
            elif '.MC' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            elif '.F' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            elif '.AS' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            elif '.BR' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            elif '.PA' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            elif '.MI' in df.columns.tolist()[column]:
                col = df.columns.tolist()[column]
                df[col] = df[col] * df_currency['EURUSD=X']
            else:
                col = df.columns.tolist()[column]
                df[col] = df[col] * 1

        cum_portfolio_returns = (df * quantity).sum(axis=1).pct_change().dropna().cumsum()
        print(cum_portfolio_returns)
        cum_benchmark_returns = index_df.pct_change().dropna().cumsum()

        first_sum_portfolio = sum(df.iloc[2] * quantity)
        current_sum_portfolio = sum(df.iloc[-1] * quantity)
        growth_portfolio = (current_sum_portfolio - first_sum_portfolio) / first_sum_portfolio

        risk_free_rate = 0
        std = (df * quantity).sum(axis=1).pct_change().dropna().std()
        Sharp = ((growth_portfolio - risk_free_rate) / std) * np.sqrt(len(df))

        cumprod_ret = ((df * quantity).sum(axis=1).pct_change().dropna() + 1).cumprod() * 100
        cumprod_market_ret = (index_df.pct_change().dropna() + 1).cumprod() * 100
        cumprod_ret.index = pd.to_datetime(cumprod_ret.index)
        trough_index = (np.maximum.accumulate(cumprod_ret) - cumprod_ret).idxmax()
        peak_index = cumprod_ret.loc[:trough_index].idxmax()
        maximum_drawdown = 100 * (cumprod_ret[trough_index] - cumprod_ret[peak_index]) / cumprod_ret[peak_index]

        # print('********************************')
        '********************************'
        'Первоначальная стоимость портфеля   %.2f' % round(first_sum_portfolio)
        'Текущая стоимость портфеля   %.2f' % round(current_sum_portfolio)
        '********************************'
        'Доходность портфеля   %.2f' % round(growth_portfolio * 100, 2) + '%'
        'Доходность портфеля годовая   %.2f' % round(((growth_portfolio * 100) / len(df)) * 252, 2) + '%'
        'Доходность индекса годовая   %.2f' % round(((cum_benchmark_returns[-1] * 100) / len(df)) * 252, 2) + '%'
        '********************************'
        'Maximum Drawdown   %.2f' % maximum_drawdown + '%'

        porfolio_price.append(current_sum_portfolio)
        return cum_portfolio_returns, total_price


    def cumRasPF(tickers_list, start, end):


        data = yf.download(tickers_list, start, end)

        pData = data['Adj Close']
        rData = pData.pct_change().mean(axis=1).dropna()
        cRDATA = ep.cum_returns(rData)
        cRDATA = cRDATA + 1
        cum_ret = ep.cum_returns_final(rData)
        cum_ret = cum_ret * 100



        maximum_drawdown = ep.max_drawdown(rData)

        '********************************'
        'Annunal Return   %.2f' % (ep.annual_return(rData) * 100) + '%'
        'Cumulative return   %.2f' % cum_ret + '%'
        'Annunal Volatility   %.2f' % (ep.annual_volatility(rData) * 100) + '%'
        'Sharpe ratio   %.2f' % ep.sharpe_ratio(rData)
        'Calmar ratio   %.2f' % ep.calmar_ratio(rData)
        'Stability   %.2f' % ep.stability_of_timeseries(rData)
        '**************************************'
        'Maximum Drawdown   %.2f' % (maximum_drawdown * 100) + '%'

        return cRDATA


    def frontAnalysis(tickers_list, start, end, total_price):


        ohlc = yf.download(tickers_list, start=start, end=end).dropna()
        prices = ohlc["Adj Close"].dropna(how="all")

        print(prices.tail())

        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        print(S)

        mu = expected_returns.ema_historical_return(prices, returns_data=False, compounding=True, span=126,
                                                    frequency=252,
                                                    log_returns=False)
        print(mu)

        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        print(cleaned_weights)

        df = pd.DataFrame.from_dict(cleaned_weights, orient='index')
        print(df)

        perf = ef.portfolio_performance(verbose=True)
        print(perf)

        latest_prices = prices.iloc[-1]  #  prices as of the day you are allocating
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_price)
        print(cvxpy.installed_solvers())
        alloc, leftover = da.lp_portfolio()
        alloc_r = pd.DataFrame.from_dict(alloc, orient='index')
        alloc_r = alloc_r.fillna(0)

        res = pd.DataFrame()
        res['Cleaned Weights'] = df[0]
        res['Discrete Allocation'] = alloc_r
        res = res.transpose().fillna(0)

        return res

    st.subheader('Portfolio Analysis')

    try:
        data_plot = cumRasPF(strTickers, start_size, end_size)
        st.plotly_chart(cumRetPlotIdeal(data_plot), use_container_width=True)
    except:
        pass

    st.subheader('Portfolio Probability Calculation')

    try:
        data_for_plot, port_price = cumProbCount(strTickers, start_size, end_size)
        st.plotly_chart(cumRetPlot(data_for_plot), use_container_width=True)
    except:
        pass

    st.subheader('Portfolio Weights by Frontier Analysis')

    try:
        st.markdown('Frontier analysis result')
        result = frontAnalysis(strTickers, start_size, end_size, port_price)
        st.dataframe(result)
    except:
        pass

if (infoType == 'Volatility Index'):

    st.title("Volatility Index for different Sources")

    st.subheader('Volatility Index S&P500')
    try:
        plot, plot_rsi, VIX, RSI = vix_snp500()

        st.plotly_chart(plot, use_container_width=True)
        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current VIX:    ', VIX['current'])
        with coll12:
            st.write('Min VIX:    ', VIX['min'])
        with coll13:
            st.write('Max VIX:    ', VIX['max'])

        st.plotly_chart(plot_rsi, use_container_width=True)
        coll21, coll22, coll23 = st.columns(3)
        with coll21:
            st.write('Current RSI VIX:    ', RSI['current'])
        with coll22:
            st.write('Min RSI VIX:    ', RSI['min'])
        with coll23:
            st.write('Max RSI VIX:    ', RSI['max'])
    except:
        pass

    st.subheader('Volatility Index for NASDAQ')
    try:
        plot, plot_rsi, VIX, RSI = vix_nasdaq()

        st.plotly_chart(plot, use_container_width=True)
        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index NASDAQ:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index NASDAQ:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index NASDAQ:    ', VIX['max'])

        st.plotly_chart(plot_rsi, use_container_width=True)
        coll21, coll22, coll23 = st.columns(3)
        with coll21:
            st.write('Current RSI NASDAQ:    ', RSI['current'])
        with coll22:
            st.write('Min RSI NASDAQ:    ', RSI['min'])
        with coll23:
            st.write('Max RSI NASDAQ:    ', RSI['max'])

    except:
        pass

    st.subheader('Volatility Index for Gold')
    try:
        plot, VIX = vix_gold()
        st.plotly_chart(plot, use_container_width=True)

        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index Gold:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index Gold:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index Gold:    ', VIX['max'])
    except:
        pass

    st.subheader('Volatility Index for Europe')
    try:
        plot, VIX = vix_euro()
        st.plotly_chart(plot, use_container_width=True)

        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index Europe:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index Europe:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index Europe:    ', VIX['max'])
    except:
        pass

    st.subheader('Volatility Index for Russell 2000')
    try:
        plot, VIX = vix_russell()
        st.plotly_chart(plot, use_container_width=True)

        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index Russell:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index Russell:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index Russell:    ', VIX['max'])
    except:
        pass

    st.subheader('Volatility Index for Emerging Markets')
    try:
        plot, VIX = vix_emerg()
        st.plotly_chart(plot, use_container_width=True)

        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index EM:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index EM:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index EM:    ', VIX['max'])
    except:
        pass

    st.subheader("Volatility Index for China")
    try:
        plot, VIX = vix_china()

        st.plotly_chart(plot, use_container_width=True)
        coll11, coll12, coll13 = st.columns(3)
        with coll11:
            st.write('Current Volatility Index China:    ', VIX['current'])
        with coll12:
            st.write('Min Volatility Index China:    ', VIX['min'])
        with coll13:
            st.write('Max Volatility Index China:    ', VIX['max'])
    except:
        pass

    st.subheader("Fair Value")
    try:
            fair, real = fair_cost()
            st.write("Fair Value S&P500:    ", fair)
            st.write("Real Value S&P50:     ", real)
    except:
        pass
