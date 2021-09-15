import pandas as pd

xls = pd.ExcelFile('Klimaat_2050_WH.xlsm')
file_P = pd.read_excel(xls, 'Neerslag', header=1, skiprows=1)
file_EV = pd.read_excel(xls, 'Verdamping', header=0)
file_EV['Datum'] = pd.to_datetime(file_EV['Datum'], format='%Y-%m-%d')
file_EV.index = file_EV.loc[:, 'Datum']
file_EV = file_EV.drop(columns='Datum')

file_P['Datum'] = pd.to_datetime(file_P['datumtijd'], format='%Y-%m-%d %H:%M')
file_P = file_P.groupby(file_P['Datum'].dt.floor('d')).sum()

frames = [file_EV, file_P]
data0 = pd.concat(frames, axis=1)

data0.to_csv("Climate_2050WH.csv")
