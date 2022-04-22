import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def FCF_plot(work_table, FCF_List_plot):
 pre_plot_date = pd.to_datetime(work_table['date'], format='%Y-%m-%d').dt.year

 ttm = ['TTM', '']

 ttm_df = pd.DataFrame(ttm)

 date_ttm = pre_plot_date[::-1].append(ttm_df).reset_index(drop=True)
 plot_date_1 = (date_ttm[1:11][::-1])
 plot_date = []

 for i in range(0, 10):
  plot_date.append(str(plot_date_1.iloc[i]))

 print(type(date_ttm))
 #     plot_date = pd.to_datetime(work_table['date'], format='%Y-%m-%d').dt.year

 #     netIncome_plot = plt.plot(plot_date[::-1], work_table['revenue'][::-1], label='Revenue', color='b')

 next_year_index = pre_plot_date[0] + 1

 freeCash = work_table['freeCashFlow'][::-1]
 ca2atement = freeCash.append(FCF_List_plot).reset_index(drop=True)
 freeCashFlow_plus_next_year5 = ca2atement.reset_index(drop=True)
 future_5year = pd.DataFrame(
  [f'{next_year_index}', f'{next_year_index + 1}', f'{next_year_index + 2}', f'{next_year_index + 3}',
   f'{next_year_index + 4}'])
 future_5year_date = date_ttm[1:11].append(future_5year).reset_index(drop=True)

 future_5year_datezzzzzzzzzzz = []

 for j in range(0, 15):
  future_5year_datezzzzzzzzzzz.append(str(future_5year_date[0][j]))

 plt.figure(figsize=(14, 8))
 freeCashFlow_plus_next_year5
 freeCashFlow_plus_year5 = plt.plot(future_5year_datezzzzzzzzzzz, freeCashFlow_plus_next_year5, label='Free Cash Flow',
                                    color='b')
 plt.scatter(future_5year_datezzzzzzzzzzz, freeCashFlow_plus_next_year5, color='r', s=40)
 #     plt.setp(netIncome_plot, linewidth=3, alpha=1)
 plt.xlabel('Dates')
 plt.ylabel('Free Cash Flow')
 plt.legend(loc='lower right', title='Legend')
 plt.title('FCF in Last 10 Years')

 plt.grid(color='black', which='major', linewidth=1.1)
 plt.minorticks_on()
 plt.grid(color='black', linestyle=':', linewidth=0.8, which='minor')

 plt.axvline(x=future_5year_datezzzzzzzzzzz[-6], ls='--', linewidth=3, color='tomato')

 plt.tick_params(axis='both',  # Применяем параметры к обеим осям
                 which='major',  # Применяем параметры к основным делениям
                 width=4,  # Ширина делений
                 color='black',  # Цвет делений
                 pad=5,  # Расстояние между черточкой и ее подписью
                 labelsize=10,  # Размер подписи
                 labelcolor='green',  # Цвет подписи
                 labelbottom=True,  # Рисуем подписи снизу
                 labelleft=True,  # слева
                 labelrotation=0)  # Поворот подписей

 return plt.show()