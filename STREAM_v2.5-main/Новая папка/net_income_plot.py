import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def net_income_plot(work_table):
 plot_date = pd.to_datetime(work_table['date'], format='%Y-%m-%d').dt.year
 plt.figure(figsize=(14, 8))

 netIncome_plot = plt.plot(plot_date[::-1], work_table['netIncome'][::-1], label='Net Income', color='b')
 plt.scatter(plot_date[::-1], work_table['netIncome'][::-1], color='r', s=40)
 plt.setp(netIncome_plot, linewidth=3, alpha=1)
 plt.xlabel('Dates')
 plt.ylabel('Net Income')
 plt.legend(loc='lower right', title='Legend')
 plt.title('Net Income in Last 10 Years')

 plt.grid(color='black', which='major', linewidth=1.1)
 plt.minorticks_on()
 plt.grid(color='black', linestyle=':', linewidth=0.8, which='minor')

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

