import pandas as pd
import numpy as np



# ____________________________  РАСЧЕТ ПОКАЗАТЕЛЕЙ КАЧЕСТВА ___________________________________________________



# считаем рост выручки______________________________________________________________


def quality(work_table, year_len, sector_margin_company, sector_company):
    print('РАСЧЕТ ПОКАЗАТЕЛЕЙ КАЧЕСТВА (оценка)')

    Total_rating = []

    pr_revenue_growth = -work_table['revenue'][year_len] + work_table['revenue'][0]

    revenue_growth_DMITR = (pr_revenue_growth / abs(
        work_table['revenue'][year_len]) * 100) / year_len

    # print(f'revenue_growth_DMITR {revenue_growth_DMITR}')

    if revenue_growth_DMITR >= 5:
        revenue_growth_rating = 1
    elif 5 > revenue_growth_DMITR >= 2:
        revenue_growth_rating = 0
    else:
        revenue_growth_rating = -1

    Total_rating.append(revenue_growth_rating)

    # считаем Использование прибыли______________________________________________________________
    average_dividend_payout_percentage = np.mean(work_table['dividendYield'][0:])

    SHAREHOLDERS_EQUITY = (work_table['totalAssets'] - work_table['totalLiabilities'])

    pr_SHAREHOLDERS_EQUITY = - SHAREHOLDERS_EQUITY[year_len] + SHAREHOLDERS_EQUITY[0]
    SHAREHOLDERS_EQUITY_growth_DMITR = (pr_SHAREHOLDERS_EQUITY / abs(
        SHAREHOLDERS_EQUITY[year_len]) * 100) / len(work_table['totalAssets'])

    use_of_profit_DMITR = round((average_dividend_payout_percentage + SHAREHOLDERS_EQUITY_growth_DMITR), 2)

    # print(f'use_of_profit_DMITR {use_of_profit_DMITR}')

    if use_of_profit_DMITR < 3:
        use_of_profit_rating = -2
    elif 6 >= use_of_profit_DMITR >= 3:
        use_of_profit_rating = 0
    else:
        use_of_profit_rating = 2

    Total_rating.append(use_of_profit_rating)

    # Считаем маржу Валовой прибыли_________________________________________________________________

    GROSS_MAR = work_table['grossProfit'] / work_table['revenue']

    gross_margin = round(np.mean(GROSS_MAR[0:]), 2) * 100

    # print('*' * 50)
    # print(f'gross_margin {gross_margin}')

    gross_Margin_sector = pd.DataFrame(sector_margin_company[sector_company]['Gross Margin'])
    qattile_gross_Margin = gross_Margin_sector.describe()

    if gross_margin <= qattile_gross_Margin.loc['25%'][0]:
        gross_margin_ratingz = -1

    elif gross_margin >= qattile_gross_Margin.loc['75%'][0]:
        gross_margin_ratingz = 1

    else:
        gross_margin_ratingz = 0

    Total_rating.append(gross_margin_ratingz)

    # Считаем Динамику валовой маржи_________________________________________________________________

    if GROSS_MAR.std() < 0.1:
        gross_margin_dynamics = 'Рост'
        gross_margin_dynamics_rating = 1
    else:
        gross_margin_dynamics = 'Падение'
        gross_margin_dynamics_rating = 0

    Total_rating.append(gross_margin_dynamics_rating)

    # print(f'operating_margin_dynamics {gross_margin_dynamics}')

    # Считаем Операционную маржу_________________________________________________________________

    OPERATING_MAR = work_table['operatingIncome'] / work_table['revenue']

    operating_margin = round(np.mean(OPERATING_MAR[0:year_len]), 2) * 100

    # print(f'operating_margin {operating_margin}')

    operating_Margin_sector = pd.DataFrame(sector_margin_company[sector_company]['Operating Margin'])
    qattile_operating_Margin = operating_Margin_sector.describe()

    if operating_margin <= qattile_operating_Margin.loc['25%'][0]:
        operating_margin_rating = -1

    elif operating_margin >= qattile_operating_Margin.loc['75%'][0]:
        operating_margin_rating = 1

    else:
        operating_margin_rating = 0

    Total_rating.append(operating_margin_rating)

    # Считаем Динамику операционной маржи_________________________________________________________________

    if OPERATING_MAR.std() < 0.1:
        operating_margin_dynamics = 'Рост'
        operating_margin_dynamics_rating = 1
    else:
        operating_margin_dynamics = 'Падение'
        operating_margin_dynamics_rating = 0

    Total_rating.append(operating_margin_dynamics_rating)

    # print(f'operating_margin_dynamics {operating_margin_dynamics}')

    # Считаем Маржу чистой прибыли_________________________________________________________________

    NET_PROFIT_MAR = work_table['netIncome'] / work_table['revenue']

    net_profit_margin = round(np.mean(NET_PROFIT_MAR[0:]), 2) * 100

    # print(f'net_profit_margin {net_profit_margin}')

    net_Profit_Margin_sector = pd.DataFrame(sector_margin_company[sector_company]['Net Profit Margin'])
    qattile_net_Profit_Margin = net_Profit_Margin_sector.describe()

    if net_profit_margin <= qattile_net_Profit_Margin.loc['25%'][0]:
        net_profit_margin_rating = -1
    elif net_profit_margin >= qattile_net_Profit_Margin.loc['75%'][0]:
        net_profit_margin_rating = 1
    else:
        net_profit_margin_rating = 0

    Total_rating.append(net_profit_margin_rating)

    # Считаем Динамику маржи чистой прибыли_________________________________________________________________

    if NET_PROFIT_MAR.std() < 0.1:
        net_profit_margin_dynamics = 'Рост'
        net_profit_margin_dynamics_rating = 1
    else:
        net_profit_margin_dynamics = 'Падение'
        net_profit_margin_dynamics_rating = 0

    Total_rating.append(net_profit_margin_dynamics_rating)

    # print(f'net_profit_margin_dynamics {net_profit_margin_dynamics}')
    # print('*' * 50)

    # Считаем Рост EPS_________________________________________________________________

    pr_EPS_growth = -work_table['eps'][year_len] + work_table['eps'][0]
    EPS_growth_DMITR = (pr_EPS_growth / abs(work_table['eps'][year_len]) * 100) / len(
        work_table['eps'])

    # print(f'EPS_growth_DMITR {EPS_growth_DMITR}')

    if EPS_growth_DMITR < 0.06:
        EPS_growth_rating = -2
    elif 0.1 >= EPS_growth_DMITR >= 0.06:
        EPS_growth_rating = 0
    else:
        EPS_growth_rating = 2

    Total_rating.append(EPS_growth_rating)

    # Считаем Уровень долга_________________________________________________________________

    debt_level = round(np.mean(work_table['totalLiabilities'] / work_table['totalAssets']), 2)

    # print(f'debt_level {debt_level}')

    if debt_level < 0.5:
        debt_level_rating = 1
    elif 0.7 >= debt_level >= 0.5:
        debt_level_rating = 0
    else:
        debt_level_rating = -1

    Total_rating.append(debt_level_rating)

    # ___________________________________ ROE_DuPont _______________________________________________

    Profitability = []
    TechnicalEfficiency = []
    FinancialStructure = []

    for year in range(0, year_len):
        last_year = year + 1
        Profitability.append(work_table['netIncome'][year] / work_table['revenue'][year])
        TechnicalEfficiency.append(work_table['revenue'][year] / (
                (work_table['totalAssets'][year] + work_table['totalAssets'].replace(np.nan, 0)[last_year]) / 2))
        FinancialStructure.append(
            ((work_table['totalAssets'][year] + work_table['totalAssets'].replace(np.nan, 0)[last_year]) / 2) /
            ((work_table['totalStockholdersEquity'][year] +
              work_table['totalStockholdersEquity'].replace(np.nan, 0)[last_year]) / 2))

    table = pd.DataFrame()
    table['Profitability'] = Profitability
    table['TechnicalEfficiency'] = TechnicalEfficiency
    table['FinancialStructure'] = FinancialStructure

    table['ROE_DuPont'] = table['FinancialStructure'] * table['TechnicalEfficiency'] * \
                          table['Profitability']

    ROE_DuPont = round(table['ROE_DuPont'].mean(axis=0), 5)

    print(f'ROE_DuPont_average {ROE_DuPont}')
    # print(decomp)

    # print(f'ROE_DuPont_average {ROE_DuPont}')
    # print(decomp)

    if ROE_DuPont < 0.08:
        ROE_average_rating = -2
    elif 0.14 > ROE_DuPont > 0.08:
        ROE_average_rating = 0
    else:
        ROE_average_rating = 2

    Total_rating.append(ROE_average_rating)

    # Считаем ROA_________________________________________________________________

    ROA_average = round(np.mean(work_table['netIncome'] / work_table['totalAssets']), 2)

    # print(f'ROA_average {ROA_average}')

    if ROA_average < 0.06:
        ROA_average_rating = -1
    elif 0.08 > ROA_average > 0.06:
        ROA_average_rating = 0
    else:
        ROA_average_rating = 1

    Total_rating.append(ROA_average_rating)

    # Получаем общую оценку_________________________________________________________________

    # print(Total_rating)
    # print(f'Общая оценка = {sum(Total_rating)}')
    # print('*' * 50)
    # print(f'''Среднее по сектору = {median_by_Industry[sector_company[0]]}''')
    # print('*' * 50)

    # print(f'''Квантили маржи Валовой прибыли {qattile_gross_Margin}''')
    # print(f'''Квантили Операционной маржи {qattile_operating_Margin}''')
    # print(f'''Квантили маржи чистой прибыли {qattile_net_Profit_Margin}''')
    # print(symbols_all)

    return sum(
        Total_rating), gross_margin, operating_margin, net_profit_margin, revenue_growth_DMITR, use_of_profit_DMITR, EPS_growth_DMITR, debt_level, ROE_DuPont, ROA_average, \
           table['ROE_DuPont'], GROSS_MAR, OPERATING_MAR, NET_PROFIT_MAR