import pandas as pd

data=pd.read_excel('./generated_sp.xlsx')

condition = data['SP(Sec/SPI)'] > 0.5

data1 = data[condition]

condition1 = data1['预测类别'] == '高'

data2 = data1[condition1]

data2_sorted = data2.sort_values(by='预测酶活', ascending=False)

data2_sorted.to_excel('./generated_sp_filtered.xlsx')