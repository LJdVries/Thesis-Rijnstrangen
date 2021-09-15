import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('ggplot')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 16,
          'axes.labelsize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)


def read_Rhine(date_start=1960, date_end=2009):
    datestart = str(date_start)
    dateend = str(date_end)

    # 2050 Stoom WH time series: 1911-oct-02 until 2011-oct-31
    data0 = pd.read_csv('Waterstand_Stoom2050(WH).csv', sep=';', decimal=',', header=0, names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  # Read dataset
    data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
    data0.index = data0.loc[:, 'Date']
    data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
    data0['WL'] = data0['WL'].fillna(-99)
    data0['WL'] = data0['WL'].astype(float)
    data0['WL'] = data0['WL'].replace(-99)

    data_stoom = pd.DataFrame(data=data0.loc[datestart:dateend, 'WL'])
    df_average_stoom = data_stoom.groupby([data_stoom.index.month, data_stoom.index.day]).mean()
    df_descending_stoom = data_stoom.sort_values('WL', ascending=False)
    df_descending_stoom.index = np.linspace(0, len(df_descending_stoom.WL), len(df_descending_stoom.WL))/len(df_descending_stoom.WL)*100

    # 2050 Rust GL time series: 1911-oct-02 until 2011-oct-31
    data0 = pd.read_csv('Waterstand_Rust2050(GL).csv', sep=';', decimal=',', header=0, names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  # Read dataset
    data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
    data0.index = data0.loc[:, 'Date']
    data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
    data0['WL'] = data0['WL'].fillna(-99)
    data0['WL'] = data0['WL'].astype(float)
    data0['WL'] = data0['WL'].replace(-99)

    data_rust = pd.DataFrame(data=data0.loc[datestart:dateend, 'WL'])
    df_average_rust = data_rust.groupby([data_rust.index.month, data_rust.index.day]).mean()
    df_descending_rust = data_rust.sort_values('WL', ascending=False)
    df_descending_rust.index = np.linspace(0, len(df_descending_rust.WL), len(df_descending_rust.WL))/len(df_descending_rust.WL)*100

    # Reference (2017) time series: 1911-oct-02 until 2011-oct-31
    data0 = pd.read_csv('Waterstand_Ref2017.csv', sep=';', decimal=',', header=0, names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  # Read dataset
    data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
    data0.index = data0.loc[:, 'Date']
    data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
    data0['WL'] = data0['WL'].fillna(-99)
    data0['WL'] = data0['WL'].astype(float)
    data0['WL'] = data0['WL'].replace(-99)

    data_ref = pd.DataFrame(data=data0.loc[datestart:dateend, 'WL'])
    df_average_ref = data_ref.groupby([data_ref.index.month, data_ref.index.day]).mean()
    df_descending_ref = data_ref.sort_values('WL', ascending=False)
    df_descending_ref.index = np.linspace(0, len(df_descending_ref.WL), len(df_descending_ref.WL))/len(df_descending_ref.WL)*100

    # Historical time series: 1824-nov-12 until 2021-mrt-14
    data0 = pd.read_csv('Waterstand_Historical_Lobith.DagGem.csv', sep=';', names=['Time', 'WL'])      # Read dataset
    data0['Date'] = pd.DatetimeIndex(data0.Time).normalize()                                # Time-field to date format (year-month-day)
    data0.index = data0.loc[:, 'Date']                                                      # Set index column
    data0 = data0.drop(columns=['Date', 'Time'])                                            # Drop columns that are (now) irrelevant
    data0['WL'] = data0['WL'].replace(-99)                                                  # Replace -99 of water levels with np.nan

    data_hist = pd.DataFrame(data=data0.loc[datestart:dateend, 'WL'])
    # print(data_hist['WL'].min())
    # print(data_hist['WL'].max())
    df_average_hist = data_hist.groupby([data_hist.index.month, data_hist.index.day]).mean()
    df_descending_hist = data_hist.sort_values('WL', ascending=False)
    df_descending_hist.index = np.linspace(0, len(df_descending_hist.WL), len(df_descending_hist.WL))/len(df_descending_hist.WL)*100


    ## PLOT THE TIME SERIES
    plt.figure(figsize=(16, 4))

    plt.plot(data_stoom.WL, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(data_rust.WL, '-', label='2050 G$_L$', color='olivedrab')
    plt.plot(data_ref.WL, '-', label='Reference', color='royalblue')
    plt.plot(data_hist.WL, '-', label='Historical', color='darkslategrey')
    plt.fill_between(data_hist.index, 10, 15.5, color='black', alpha=0.2, label='Weir height range')

    leg1 = plt.legend(loc='upper right')
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Water level [m +NAP]')
    # plt.gcf().autofmt_xdate()
    plt.show()


    ## PLOT THE FLOW DURATION CURVES
    plt.figure(figsize=(15, 10))
    plt.fill_between(np.linspace(0, 100, 100), 10.0, 15.5, color='black', alpha=0.15, label='Weir height range')
    # plt.fill_between(np.linspace(0, 2, 2), 13.5, 15.5, color='black', alpha=0.15, label='Weir height range')

    plt.plot(df_descending_stoom.WL, '-', label='2050 W$_H$', color='firebrick', linewidth=3)
    # plt.plot(df_descending_stoom.loc[0:2, 'WL'], '-', label='2050 W$_H$', color='firebrick', linewidth=4)
    plt.plot(df_descending_rust.WL, '-', label='2050 G$_L$', color='olivedrab', linewidth=3)
    # plt.plot(df_descending_rust.loc[0:2, 'WL'], '-', label='2050 G$_L$', color='olivedrab', linewidth=4)
    plt.plot(df_descending_ref.WL, '-', label='Reference', color='royalblue', linewidth=3)
    # plt.plot(df_descending_ref.loc[0:2, 'WL'], '-', label='Reference', color='royalblue', linewidth=4)
    plt.plot(df_descending_hist.WL, '-', label='Historical', color='darkslategrey', linewidth=3)
    # plt.plot(df_descending_hist.loc[0:2, 'WL'], '-', label='Historical', color='darkslategrey', linewidth=4)

    leg1 = plt.legend(loc='lower left')
    plt.ylabel('Water level Rhine [m +NAP]')
    plt.xlabel('Percentage of time water level is exceeded')
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)
    plt.show()


    ## PLOT THE AVERAGES
    plt.figure(figsize=(16, 8))

    avg_stoom = df_average_stoom['WL'].values
    avg_stoom = np.delete(avg_stoom, 58)
    avg_rust = df_average_rust['WL'].values
    avg_rust = np.delete(avg_rust, 58)
    avg_ref = df_average_ref['WL'].values
    avg_ref = np.delete(avg_ref, 58)
    avg_hist = df_average_hist['WL'].values
    avg_hist = np.delete(avg_hist, 59)
    avg_hist = np.delete(avg_hist, 364)

    plt.plot(np.linspace(1, 364, 364), avg_stoom, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(np.linspace(1, 364, 364), avg_rust, '-', label='2050 G$_L$', color='olivedrab')
    plt.plot(np.linspace(1, 364, 364), avg_ref, '-', label='Reference', color='royalblue')
    plt.plot(np.linspace(1, 364, 364), avg_hist, '-', label='Historical', color='darkslategrey')
    plt.fill_between(np.linspace(1, 364, 364), 10, 15.5, color='black', alpha=0.2, label='Weir height range')

    leg1 = plt.legend(loc='upper right', prop={'size': 16})
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Water level [m +NAP]')
    plt.gcf().autofmt_xdate()
    plt.show()
# read_Rhine(date_start=1960, date_end=2009)


def read_precipitation(date_start=1960, date_end=2009):
    datestart = str(date_start)
    dateend = str(date_end)

    ## Climate 2050 WH  -- 1906-01-01 until 2014-12-31 : Climate_2050WH.csv
    df = pd.read_csv('Climate_2050WH.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum)
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G_center']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                          'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    data_WH = pd.DataFrame(data=df.loc[datestart:dateend, 'P'])
    data_average_WH = data_WH.groupby([data_WH.index.month, data_WH.index.day]).mean()
    df_descending_WH = data_WH.sort_values('P', ascending=False)
    df_descending_WH.index = np.linspace(0, len(df_descending_WH.P), len(df_descending_WH.P)) / len(df_descending_WH.P) * 100

    ## Reference (2014) -- 1906-01-01 until 2014-12-31 : Climate_Ref2014.csv
    df = pd.read_csv('Climate_Ref2014.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum, format='%m/%d/%Y')
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G', 'H', 'H+', 'L'])

    data_ref = pd.DataFrame(data=df.loc[datestart:dateend, 'P'])
    data_average_ref = data_ref.groupby([data_ref.index.month, data_ref.index.day]).mean()
    df_descending_ref = data_ref.sort_values('P', ascending=False)
    df_descending_ref.index = np.linspace(0, len(df_descending_ref.P), len(df_descending_ref.P)) / len(df_descending_ref.P) * 100

    ## Climate 2050 GL  -- 1906-01-01 until 2014-12-31 : Climate_2050GL.csv
    df = pd.read_csv('Climate_2050GL.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum)
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G_center']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                          'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    data_GL = pd.DataFrame(data=df.loc[datestart:dateend, 'P'])
    data_average_GL = data_GL.groupby([data_GL.index.month, data_GL.index.day]).mean()
    df_descending_GL = data_GL.sort_values('P', ascending=False)
    df_descending_GL.index = np.linspace(0, len(df_descending_GL.P), len(df_descending_GL.P)) / len(df_descending_GL.P) * 100

    ## Historical: 1957-jul-01 until 2021-jun-21 (present)
    df = pd.read_csv('Climate_Historical_DeBilt.txt', sep=',', skipinitialspace=True)
    df['Date'] = pd.to_datetime(df.YYYYMMDD, format='%Y%m%d')
    df.index = df.loc[:, 'Date']
    df = df.drop(columns=['STN', 'YYYYMMDD', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FHXH', 'FHN', 'FHNH', 'FXX', 'FXXH', 'TG', 'TN',
                          'TNH', 'TX', 'TXH', 'T10N', 'T10NH', 'SQ', 'SP', 'Q', 'DR', 'RHX', 'RHXH', 'PG', 'PX', 'PXH', 'PN',
                          'PNH', 'VVN', 'VVNH', 'VVX', 'VVXH', 'NG', 'UG', 'UX', 'UN', 'UXH', 'UNH', 'Date'])
    df.columns = ['P', 'EV']
    df = df.loc[str(date_start):str(date_end)]
    df.P = df.P.replace(-1, 0)    # -1 was the value for P < 0.05mm, now this is set to 0mm
    df.P = df.P*0.1               # etmaalsom van de neerslag (from 0.1 mm to mm)
    df.EV = df.EV*0.1             # referentiegewasverdamping Makkink (from 0.1 mm to mm)

    data_hist = pd.DataFrame(data=df.loc[datestart:dateend, 'P'])
    data_average_hist = data_hist.groupby([data_hist.index.month, data_hist.index.day]).mean()
    df_descending_hist = data_hist.sort_values('P', ascending=False)
    df_descending_hist.index = np.linspace(0, len(df_descending_hist.P), len(df_descending_hist.P)) / len(df_descending_hist.P) * 100


    ## PLOT THE TIME SERIES
    plt.figure(figsize=(16, 4))

    plt.plot(data_WH.P, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(data_ref.P, '-', label='Reference', color='royalblue')
    plt.plot(data_GL.P, '-', label='2050 G$_L$', color='olivedrab')
    plt.plot(data_hist.P, '-', label='Historical', color='darkslategrey')

    leg1 = plt.legend(loc='upper right', prop={'size': 16})
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Precipitation [mm/day]')
    # plt.gcf().autofmt_xdate()
    plt.show()


    ## PLOT THE FLOW DURATION CURVES
    plt.figure(figsize=(15, 10))

    plt.plot(df_descending_WH.P, '-', label='2050 W$_H$', color='firebrick', linewidth=3)
    # plt.plot(df_descending_WH.loc[30:100, 'P'], '-', label='2050 W$_H$', color='firebrick', linewidth=4)
    plt.plot(df_descending_GL.P, '-', label='2050 G$_L$', color='olivedrab', linewidth=3)
    # plt.plot(df_descending_GL.loc[30:100, 'P'], '-', label='2050 G$_L$', color='olivedrab', linewidth=4)
    plt.plot(df_descending_ref.P, '-', label='Reference', color='royalblue', linewidth=3)
    # plt.plot(df_descending_ref.loc[30:100, 'P'], '-', label='Reference', color='royalblue', linewidth=4)
    plt.plot(df_descending_hist.P, '-', label='Historical', color='darkslategrey', linewidth=3)
    # plt.plot(df_descending_hist.loc[30:100, 'P'], '-', label='Historical', color='darkslategrey', linewidth=4)

    # plt.yscale("log")
    leg1 = plt.legend(loc='upper right')
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Percentage of time precipitation is exceeded')
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)
    plt.show()


    ## PLOT THE AVERAGES
    plt.figure(figsize=(16, 8))

    avg_WH = data_average_WH['P'].values
    avg_ref = data_average_ref['P'].values
    avg_GL = data_average_GL['P'].values
    avg_hist = data_average_hist['P'].values

    plt.plot(np.linspace(1, 366, 366), avg_WH, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(np.linspace(1, 366, 366), avg_ref, '-', label='Reference', color='olivedrab')
    plt.plot(np.linspace(1, 366, 366), avg_GL, '-', label='2050 G$_L$', color='royalblue')
    plt.plot(np.linspace(1, 366, 366), avg_hist, '-', label='Historical', color='darkslategrey')

    leg1 = plt.legend(loc='upper right', prop={'size': 16})
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Precipitation [mm/day]')
    plt.gcf().autofmt_xdate()
    plt.show()
# read_precipitation(date_start=1960, date_end=2009)


def read_evaporation(date_start=1960, date_end=2009):
    datestart = str(date_start)
    dateend = str(date_end)

    ## Climate 2050 WH  -- 1906-01-01 until 2014-12-31 : Climate_2050WH.csv
    df = pd.read_csv('Climate_2050WH.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum)
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G_center']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                          'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    data_WH = pd.DataFrame(data=df.loc[datestart:dateend, 'EV'])
    data_average_WH = data_WH.groupby([data_WH.index.month, data_WH.index.day]).mean()
    df_descending_WH = data_WH.sort_values('EV', ascending=False)
    df_descending_WH.index = np.linspace(0, len(df_descending_WH.EV), len(df_descending_WH.EV)) / len(df_descending_WH.EV) * 100

    ## Reference (2014) -- 1906-01-01 until 2014-12-31 : Climate_Ref2014.csv
    df = pd.read_csv('Climate_Ref2014.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum, format='%m/%d/%Y')
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G', 'H', 'H+', 'L'])

    data_ref = pd.DataFrame(data=df.loc[datestart:dateend, 'EV'])
    data_average_ref = data_ref.groupby([data_ref.index.month, data_ref.index.day]).mean()
    df_descending_ref = data_ref.sort_values('EV', ascending=False)
    df_descending_ref.index = np.linspace(0, len(df_descending_ref.EV), len(df_descending_ref.EV)) / len(df_descending_ref.EV) * 100

    ## Climate 2050 GL  -- 1906-01-01 until 2014-12-31 : Climate_2050GL.csv
    df = pd.read_csv('Climate_2050GL.csv', sep=',', header=0)
    df['Datum'] = pd.to_datetime(df.Datum)
    df.index = df.loc[:, 'Datum']
    df = df.loc[str(date_start):str(date_end)]
    df['P'] = df['G_center']
    df['EV'] = df['Makkink (mm)']
    df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                          'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    data_GL = pd.DataFrame(data=df.loc[datestart:dateend, 'EV'])
    data_average_GL = data_GL.groupby([data_GL.index.month, data_GL.index.day]).mean()
    df_descending_GL = data_GL.sort_values('EV', ascending=False)
    df_descending_GL.index = np.linspace(0, len(df_descending_GL.EV), len(df_descending_GL.EV)) / len(df_descending_GL.EV) * 100

    ## Historical: 1957-jul-01 until 2021-jun-21 (present)
    df = pd.read_csv('Climate_Historical_DeBilt.txt', sep=',', skipinitialspace=True)
    df['Date'] = pd.to_datetime(df.YYYYMMDD, format='%Y%m%d')
    df.index = df.loc[:, 'Date']
    df = df.drop(columns=['STN', 'YYYYMMDD', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FHXH', 'FHN', 'FHNH', 'FXX', 'FXXH', 'TG', 'TN',
                          'TNH', 'TX', 'TXH', 'T10N', 'T10NH', 'SQ', 'SP', 'Q', 'DR', 'RHX', 'RHXH', 'PG', 'PX', 'PXH', 'PN',
                          'PNH', 'VVN', 'VVNH', 'VVX', 'VVXH', 'NG', 'UG', 'UX', 'UN', 'UXH', 'UNH', 'Date'])
    df.columns = ['P', 'EV']
    df = df.loc[str(date_start):str(date_end)]
    df.P = df.P.replace(-1, 0)    # -1 was the value for P < 0.05mm, now this is set to 0mm
    df.P = df.P*0.1               # etmaalsom van de neerslag (from 0.1 mm to mm)
    df.EV = df.EV*0.1             # referentiegewasverdamping Makkink (from 0.1 mm to mm)

    data_hist = pd.DataFrame(data=df.loc[datestart:dateend, 'EV'])
    data_average_hist = data_hist.groupby([data_hist.index.month, data_hist.index.day]).mean()
    df_descending_hist = data_hist.sort_values('EV', ascending=False)
    df_descending_hist.index = np.linspace(0, len(df_descending_hist.EV), len(df_descending_hist.EV)) / len(df_descending_hist.EV) * 100


    ## PLOT THE TIME SERIES
    plt.figure(figsize=(16, 4))

    plt.plot(data_WH.EV, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(data_ref.EV, '-', label='Reference', color='royalblue')
    plt.plot(data_GL.EV, '-', label='2050 G$_L$', color='olivedrab')
    plt.plot(data_hist.EV, '-', label='Historical', color='darkslategrey')

    leg1 = plt.legend(loc='upper right', prop={'size': 16})
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Evaporation [mm/day]')
    # plt.gcf().autofmt_xdate()
    plt.show()


    ## PLOT THE FLOW DURATION CURVES
    plt.figure(figsize=(15, 10))

    plt.plot(df_descending_WH.EV, '-', label='2050 W$_H$', color='firebrick', linewidth=3)
    # plt.plot(df_descending_WH.loc[0:1, 'EV'], '-', label='2050 W$_H$', color='firebrick', linewidth=4)
    plt.plot(df_descending_GL.EV, '-', label='2050 G$_L$', color='olivedrab', linewidth=3)
    # plt.plot(df_descending_GL.loc[0:1, 'EV], '-', label='2050 G$_L$', color='olivedrab', linewidth=4)
    plt.plot(df_descending_ref.EV, '-', label='Reference', color='royalblue', linewidth=3)
    # plt.plot(df_descending_ref.loc[0:1, 'EV'], '-', label='Reference', color='royalblue', linewidth=4)
    plt.plot(df_descending_hist.EV, '-', label='Historical', color='darkslategrey', linewidth=3)
    # plt.plot(df_descending_hist.loc[0:1, 'EV'], '-', label='Historical', color='darkslategrey', linewidth=4)

    leg1 = plt.legend(loc='upper right')
    plt.ylabel('Evaporation [mm/day]')
    plt.xlabel('Percentage of time evaporation is exceeded')
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)
    plt.show()


    ## PLOT THE AVERAGES
    plt.figure(figsize=(16, 8))

    avg_WH = data_average_WH['EV'].values
    avg_ref = data_average_ref['EV'].values
    avg_GL = data_average_GL['EV'].values
    avg_hist = data_average_hist['EV'].values

    plt.plot(np.linspace(1, 366, 366), avg_WH, '-', label='2050 W$_H$', color='firebrick')
    plt.plot(np.linspace(1, 366, 366), avg_ref, '-', label='Reference', color='olivedrab')
    plt.plot(np.linspace(1, 366, 366), avg_GL, '-', label='2050 G$_L$', color='royalblue')
    plt.plot(np.linspace(1, 366, 366), avg_hist, '-', label='Historical', color='darkslategrey')

    leg1 = plt.legend(loc='upper right', prop={'size': 16})
    leg1.get_lines()[0].set_linewidth(6)
    leg1.get_lines()[1].set_linewidth(6)
    leg1.get_lines()[2].set_linewidth(6)
    leg1.get_lines()[3].set_linewidth(6)

    plt.ylabel('Evaporation [mm/day]')
    plt.xlabel('Day number')
    # plt.gcf().autofmt_xdate()
    plt.show()
# read_evaporation(date_start=2005, date_end=2005)
