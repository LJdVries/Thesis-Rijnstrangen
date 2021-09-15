import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.linalg import expm, sqrtm
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')
import matplotlib.pylab as pylab


data_lookup = pd.read_csv('Vol&Area_vs_depth_CSV.csv', header=0, sep=',')
def lookup_V(V):
    V = V * 1000000
    if V <= data_lookup.loc[0, 'Volume (Mm3)']:
        h = data_lookup.loc[0, 'Water level (m +NAP)']
        A_wet = data_lookup.loc[0, 'Area (m2)']
    elif V >= data_lookup.loc[22, 'Volume (Mm3)']:
        h = data_lookup.loc[22, 'Water level (m +NAP)']
        A_wet = data_lookup.loc[22, 'Area (m2)']
    else:
        for i in np.arange(22, 0, -1):
            if V <= data_lookup.loc[i, 'Volume (Mm3)']:
                h = (data_lookup.loc[i, 'Water level (m +NAP)'] + data_lookup.loc[i - 1, 'Water level (m +NAP)']) / 2
                A_wet = (data_lookup.loc[i, 'Area (m2)'] + data_lookup.loc[i - 1, 'Area (m2)']) / 2
            else:
                break
    return h, A_wet


def lookup_h(h):
    if h <= data_lookup.loc[0, 'Water level (m +NAP)']:
        V = data_lookup.loc[0, 'Volume (Mm3)']
        A_wet = data_lookup.loc[0, 'Area (m2)']
    elif h >= data_lookup.loc[22, 'Water level (m +NAP)']:
        V = data_lookup.loc[22, 'Volume (Mm3)']
        A_wet = data_lookup.loc[22, 'Area (m2)']
    else:
        for i in np.arange(22, 0, -1):
            if h <= data_lookup.loc[i, 'Water level (m +NAP)']:
                V = (data_lookup.loc[i, 'Volume (Mm3)'] + data_lookup.loc[i-1, 'Volume (Mm3)'])/2
                A_wet = (data_lookup.loc[i, 'Area (m2)'] + data_lookup.loc[i-1, 'Area (m2)'])/2
            else:
                break
    V = V / 1000000
    return V, A_wet


def read_Rhine(scenario, date_start, date_end):
    ## options for scenario: "Historical", "Reference", "2050GL" and "2050WH"
    ## date_start is starting year from which Rhine data is read
    ## date_end is last year included in the data (up to and including)

    if scenario == 'Historical':
        ## Historical time series: 1824-nov-12 until 2021-mrt-14
        data0 = pd.read_csv('Waterstand_Historical_Lobith.DagGem.csv', sep=';', names=['Time', 'WL'])      ## Read dataset
        data0['Date'] = pd.DatetimeIndex(data0.Time).normalize()                                ## Time-field to date format (year-month-day)
        data0.index = data0.loc[:, 'Date']                                                      ## Set index column
        data0 = data0.drop(columns=['Date', 'Time'])                                            ## Drop columns that are (now) irrelevant
        data0['WL'] = data0['WL'].replace(-99)                                                  ## Replace -99 of water levels with np.nan

    if scenario == 'Reference':
        ## Reference (2017) time series: 1911-oct-02 until 2011-oct-31
        data0 = pd.read_csv('Waterstand_Ref2017.csv', sep=';', decimal=',', header=0, names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  ## Read dataset
        data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
        data0.index = data0.loc[:, 'Date']
        data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
        ## data0['WL'] = data0['WL'].replace(',', '.')
        data0['WL'] = data0['WL'].fillna(-99)
        data0['WL'] = data0['WL'].astype(float)
        data0['WL'] = data0['WL'].replace(-99)

    if scenario == '2050GL':
        ## 2050 Rust GL time series: 1911-oct-02 until 2011-oct-31
        data0 = pd.read_csv('Waterstand_Rust2050(GL).csv', sep=';', decimal=',', header=0,
                            names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  ## Read dataset
        data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
        data0.index = data0.loc[:, 'Date']
        data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
        data0['WL'] = data0['WL'].fillna(-99)
        data0['WL'] = data0['WL'].astype(float)
        data0['WL'] = data0['WL'].replace(-99)

    if scenario == '2050WH':
        ## 2050 Stoom WH time series: 1911-oct-02 until 2011-oct-31
        data0 = pd.read_csv('Waterstand_Stoom2050(WH).csv', sep=';', decimal=',', header=0, names=['Time', 'loc', 'parm', 'scen', 'unit', 'WL'])  ## Read dataset
        data0['Date'] = pd.to_datetime(data0['Time'], format='%d-%m-%Y')
        data0.index = data0.loc[:, 'Date']
        data0 = data0.drop(columns=['Date', 'Time', 'loc', 'parm', 'scen', 'unit'])
        ## data0['WL'] = data0['WL'].replace(',', '.')
        data0['WL'] = data0['WL'].fillna(-99)
        data0['WL'] = data0['WL'].astype(float)
        data0['WL'] = data0['WL'].replace(-99)

    ## From the original data (data0) select the part to work with:
    data = pd.DataFrame(data=data0.loc[str(date_start):str(date_end)])

    return data


def read_climate(scenario, date_start, date_end):
    ## options for scenario: "Historical", "Reference", "2050GL" and "2050WH"
    ## date_start is starting year from which Rhine data is read
    ## date_end is last year included in the data (up to and including)

    if scenario == 'Historical':
        ## Historical: 1957-jul-01 until 2021-jun-21 (present)
        df = pd.read_csv('Climate_Historical_DeBilt.txt', sep=',', skipinitialspace=True)
        df['Date'] = pd.to_datetime(df.YYYYMMDD, format='%Y%m%d')
        df.index = df.loc[:, 'Date']
        df = df.drop(columns=['STN', 'YYYYMMDD', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FHXH', 'FHN', 'FHNH', 'FXX', 'FXXH', 'TG', 'TN',
                              'TNH', 'TX', 'TXH', 'T10N', 'T10NH', 'SQ', 'SP', 'Q', 'DR', 'RHX', 'RHXH', 'PG', 'PX', 'PXH', 'PN',
                              'PNH', 'VVN', 'VVNH', 'VVX', 'VVXH', 'NG', 'UG', 'UX', 'UN', 'UXH', 'UNH', 'Date'])
        df.columns = ['P', 'EV']
        df = df.loc[str(date_start):str(date_end)]
        df.P = df.P.replace(-1, 0)    ## -1 was the value for P < 0.05mm, now this is set to 0mm
        df.P = df.P*0.1               ## etmaalsom van de neerslag (from 0.1 mm to mm)
        df.EV = df.EV*0.1             ## referentiegewasverdamping Makkink (from 0.1 mm to mm)

    if scenario == 'Reference':
        ## Reference (2014): 1906-01-01 until 2014-12-31
        df = pd.read_csv('Climate_Ref2014.csv', sep=',', header=0)
        df['Datum'] = pd.to_datetime(df.Datum, format='%m/%d/%Y')
        df.index = df.loc[:, 'Datum']
        df = df.loc[str(date_start):str(date_end)]
        df['P'] = df['G']
        df['EV'] = df['Makkink (mm)']
        df = df.drop(columns=['Datum', 'Makkink (mm)', 'G', 'H', 'H+', 'L'])

    if scenario == '2050GL':
        ## Climate 2050 GL: 1906-01-01 until 2014-12-31
        df = pd.read_csv('Climate_2050GL.csv', sep=',', header=0)
        df['Datum'] = pd.to_datetime(df.Datum)
        df.index = df.loc[:, 'Datum']
        df = df.loc[str(date_start):str(date_end)]
        df['P'] = df['G_center']
        df['EV'] = df['Makkink (mm)']
        df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                              'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    if scenario == '2050WH':
        ## Climate 2050 WH: 1906-01-01 until 2014-12-31
        df = pd.read_csv('Climate_2050WH.csv', sep=',', header=0)
        df['Datum'] = pd.to_datetime(df.Datum)
        df.index = df.loc[:, 'Datum']
        df = df.loc[str(date_start):str(date_end)]
        df['P'] = df['G_center']
        df['EV'] = df['Makkink (mm)']
        df = df.drop(columns=['Datum', 'Makkink (mm)', 'G_lower', 'G_center', 'G_upper', 'H_lower', 'H_center', 'H_upper',
                              'Hplus_lower', 'Hplus_center', 'Hplus_upper', 'L_lower', 'L_center', 'L_upper'])

    return df


def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False):
    ## options for h_Rijnstrangen: value, most likely between 10.0 and 15.5 m +NAP
    ## Q_GWleft and Q_GWright respectively are the groundwater extraction on the left (southwest) and right (northeast) of the Rijnstrangen
    ## plot_phi, plot_q and plot_s can be True or False, dependent on if the heads (phi), groundwater flow (q) or seepage (s) have to be plotted

    A_wet = lookup_h(h_Rijnstrangen)[1]
    fraction = A_wet/A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left
    c_var = 10000

    ## c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    ## MULTI LAYER SYSTEM
    # c = np.array([[1, 650, c_var, 80, c_var, 250, 225],
    #               [1.0e4, 3.2e4, 3.2e4, 3.2e4, 3.2e4, 1.0e4, 5.0e3],
    #               [1.0e2, 5.0e2, 1.0e3, 5.0e2, 5.0e2, 5.0e2, 5.0e2]])

    ## ONE LAYER SYSTEM
    c = np.array([[1, 650, c_var, 80, c_var, 250, 225]])

    ## T, (nLay by nSec) matrix of transmissivity values
    ## MULTI LAYER SYSTEM
    # kD = np.array([[250, 750, 750, 750, 750, 750, 750],
    #                [500, 100, 50, 50, 50, 50, 250],
    #                [400, 400, 500, 500, 500, 500, 500]])

    ## ONE LAYER SYSTEM
    kD = np.array([[250, 750, 750, 750, 750, 750, 750]])

    ## h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    heads = np.array([9.2, 9.8, 9.8, h_Rijnstrangen, 9.7, 9.7, 9.1])

    ## Q, (nNod by nSec) matrix of nodal injections [L2/T]
    ## MULTI LAYER SYSTEM
    # Q = np.array([[0,  Q_GWleft,  0, 0,  Q_GWright,  0],
    #               [0,  0,  0,  0,  0,  0],
    #               [0,  0,  0,  0,  0,  0]])

    ## ONE LAYER SYSTEM
    Q = np.array([[0, Q_GWleft, 0, 0, Q_GWright, 0]])

    ## x, (nNod by 1) vector of coordinates of intersection points except +/-inf
    x = np.array([-3100, 0, x_left, x_right, 2000, 3000])

    ## X, vector of points where values will be computed
    X = np.arange(-3700, 5510, 10)

    nLay = len(kD[:, 0])
    nSec = len(kD[0, :])

    ## include the outer sections to infinity
    a = np.zeros((nLay, 1))
    Q = np.concatenate((a, Q, a), axis=1)
    x = np.append(x, math.inf)
    x = np.append(-math.inf, x)
    Nx = len(x)
    H = np.ones((nLay, 1)) * heads

    ## Mid-section points are used to compute relative coordinates within sections
    xMidSec = 0.5 * (x[:-1] + x[1:])
    xMidSec[0] = x[1]
    xMidSec[-1] = x[-2]

    ## System matrices for all sections
    A = np.zeros((nLay, nLay, nSec))
    for iSec in range(nSec):
        a = 1 / (kD[:, iSec] * c[:, iSec])
        p = np.append(c[1:nLay, iSec], math.inf)
        b = 1 / (kD[:, iSec] * p)
        A[:, :, iSec] = np.diag(a + b) - np.diag(a[1:nLay], -1) - np.diag(b[0:nLay - 1], 1)

    ## Generating and filling the coefficient matrix C
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  ## coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  ## right hand side vector

    nNod = nSec - 1
    for i in range(nNod):
        ## i is section N, also the left node number of the section
        ## j is right the node number of the section
        ## ii and jj point to the position within the total coefficient matrix
        j = i + 1
        ii = 2 * nLay * (i)
        jj = ii + nLay
        C[ii:jj, ii:jj] = +expm(-(x[j] - xMidSec[i]) * sqrtm(A[:, :, i]))
        C[ii:jj, ii + nLay:jj + nLay] = +expm(+(x[j] - xMidSec[i]) * sqrtm(A[:, :, i]))
        C[ii:jj, ii + 2 * nLay: jj + 2 * nLay] = -expm(-(x[j] - xMidSec[j]) * sqrtm(A[:, :, j]))
        C[ii:jj, ii + 3 * nLay: jj + 3 * nLay] = -expm(+(x[j] - xMidSec[j]) * sqrtm(A[:, :, j]))
        R[ii:jj] = np.vstack(-H[:, i] + H[:, j])

        C[ii + nLay:jj + nLay, ii:jj] = np.matmul(np.matmul(-np.diag(kD[:, i]), sqrtm(A[:, :, i])),
                                                  expm(-(x[j] - xMidSec[i]) * sqrtm(A[:, :, i])))
        C[ii + nLay:jj + nLay, ii + nLay:jj + nLay] = np.matmul(np.matmul(+np.diag(kD[:, i]), sqrtm(A[:, :, i])),
                                                                expm(+(x[j] - xMidSec[i]) * sqrtm(A[:, :, i])))
        C[ii + nLay:jj + nLay, ii + 2 * nLay:jj + 2 * nLay] = np.matmul(
            np.matmul(+np.diag(kD[:, j]), sqrtm(A[:, :, j])), expm(-(x[j] - xMidSec[j]) * sqrtm(A[:, :, j])))
        C[ii + nLay:jj + nLay, ii + 3 * nLay:jj + 3 * nLay] = np.matmul(
            np.matmul(-np.diag(kD[:, j]), sqrtm(A[:, :, j])), expm(+(x[j] - xMidSec[j]) * sqrtm(A[:, :, j])))
        R[ii + nLay:jj + nLay] = np.vstack(Q[:, j])

    ## Solve the system, using all layers and leaving out the outer column as they have no freedom, because the sections extend to infinity
    COEF = np.vstack(spsolve((C[:, nLay:-nLay]), R))
    COEF = np.concatenate((np.zeros((nLay, 1)), COEF, np.zeros((nLay, 1))))

    ## output:
    ## phi [H] = computed heads, a (nLay by length(X)) matrix
    ## q [L2/T] = computed flows, a (nLay by length(X)) matrix
    ## s [L/T] = downward positive seepage rate through top of each layer, a nLay by length(X) matrix
    phi = np.zeros((nLay, len(X)))
    q = np.zeros((nLay, len(X)))
    s = np.zeros((nLay, len(X)))

    for i in range(len(X)):
        iSec = np.nonzero(np.logical_and(X[i] > x[:-1], X[i] <= x[1:]))
        iSec = iSec[0]
        iSec = iSec[0]
        k = 2 * nLay * (iSec)
        l = k + nLay

        C1 = np.matmul(expm(-(X[i] - xMidSec[iSec]) * sqrtm(A[:, :, iSec])), COEF[k:l])
        C2 = np.matmul(expm(+(X[i] - xMidSec[iSec]) * sqrtm(A[:, :, iSec])), COEF[k + nLay:l + nLay])
        C3 = np.matmul(sqrtm(A[:, :, iSec]), C1)
        C4 = np.matmul(sqrtm(A[:, :, iSec]), C2)

        phi[:, i] = np.hstack(C1) + np.hstack(C2) + (H[:, iSec])
        q[:, i] = np.hstack(np.matmul(np.diag(kD[:, iSec]), (C3 - C4)))

        sNet = np.matmul(np.matmul(np.diag(kD[:, iSec]), sqrtm(A[:, :, iSec])), (C3 + C4))
        s[nLay - 1, i] = sNet[nLay - 1]
        for iLay in np.arange(nLay - 2, -1, -1):
            s[iLay, i] = sNet[iLay] + s[iLay + 1, i]

    ## MULTI LAYER SYSTEM
    # qsum = q[0, :] + q[1, :] + q[2, :]

    ## ONE LAYER SYSTEM
    qsum = q[0, :]

    qleft_Rijnstr = qsum[371]                                   ## Q left of Rijnstrangen
    qright_Rijnstr = qsum[570]                                  ## Q right of Rijnstrangen
    qtot = (qright_Rijnstr - qleft_Rijnstr) * 12000 / 1000000   ## gives q_total out of Rijnstrangen in Mm3/d

    Q_extracted = (-Q_GWleft - Q_GWright) * 12000 / 1000000

    qleft_region= qsum[370]         ## Q right of Rijnstrangen coming from surrounding
    qright_region = qsum[571]       ## Q left of Rijnstrangen coming from surrounding
    # qtot_region = (qleft_region - qright_region) * 12000 / 1000000  ## gives q_total coming from surrounding region in Mm3/d

    if qleft_region > 0 and qright_region < 0:
        Q_perc_Rijnstr = qtot / Q_extracted
    if qleft_region > 0 and qright_region > 0:
        Q_perc_Rijnstr = (qtot - (qright_region*12000/1000000)) / Q_extracted
    if qleft_region < 0 and qright_region < 0:
        Q_perc_Rijnstr = (qtot - (-qleft_region * 12000 / 1000000)) / Q_extracted
    if qleft_region < 0 and qright_region > 0:
        Q_perc_Rijnstr = 1
    Q_perc_region = 1 - Q_perc_Rijnstr

    if plot_phi is True:
        plt.figure(figsize=(15, 10))

        plt.axvline(-3700, c='dimgrey', linestyle=':')
        plt.axvline(-3100, c='dimgrey', linestyle=':')
        plt.axvline(0, c='dimgrey', linestyle=':', label='Fixed section separation')
        plt.axvline(x_left, c='darkgrey', linestyle=':', label='Flexible section separation')
        plt.axvline(x_right, c='darkgrey', linestyle=':')
        plt.axvline(2000, c='dimgrey', linestyle=':')
        plt.axvline(3000, c='dimgrey', linestyle=':')
        plt.axvline(5500, c='dimgrey', linestyle=':')

        # v = phi[0,:].min()
        v = 8.9

        marge = v*0.005
        plt.plot([-3700, -3100], [v - marge, v - marge], c='grey')
        plt.plot([0, 2000], [v - marge, v - marge], c='grey')
        plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)
        plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)

        plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation', c='darkred', linestyle='--', linewidth=2)
        ## ONLY FOR MULTY LAYER SYSTEM:
        # plt.plot(X, phi[1, :], label='Average head in second aquifer (modelled)', c='royalblue', linestyle='-.', linewidth=2)
        # plt.plot(X, phi[2, :], label='Average head in third aquifer (modelled)', c='darkgreen', linestyle=':', linewidth=2)

        # plt.title('Heads in aquifers', fontsize=22)
        plt.xlabel('Distance along cross-section [m]', size=14)
        plt.ylabel('Head [m]', size=14)
        leg = plt.legend(fontsize=14, loc='best')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)
        leg.get_lines()[2].set_linewidth(4)
        plt.show()

    if plot_q is True:
        ## To go from m2/d to Mm3/y:
        # qleft_region = qleft_region*12000*365/1000000
        # qleft_Rijnstr = qleft_Rijnstr*12000*365/1000000
        # qright_region = qright_region*12000*365/1000000
        # qright_Rijnstr = qright_Rijnstr*12000*365/1000000
        # q = q*12000*365/1000000

        plt.figure(figsize=(15, 8))
        plt.axhline(0, linewidth=2, c='white')
        plt.plot([0, 0, 2000, 2000], [qleft_Rijnstr, qleft_region, qright_Rijnstr, qright_region], 'o', c='black')
        plt.text(-400, qleft_Rijnstr, round(qleft_Rijnstr, 1), size=12)
        plt.text(2100, qright_Rijnstr, round(qright_Rijnstr, 1), size=12)
        plt.text(100, qleft_region, round(qleft_region, 1), size=12)
        plt.text(1600, qright_region, round(qright_region, 1), size=12)
        plt.axvline(0, c='dimgrey', linestyle=':', label='Fixed section separation')
        plt.axvline(x_left, c='darkgrey', linestyle=':', label='Flexible section separation')
        plt.axvline(x_right, c='darkgrey', linestyle=':')
        plt.axvline(2000, c='dimgrey', linestyle=':')
        plt.axvline(3000, c='dimgrey', linestyle=':')
        plt.axvline(5500, c='dimgrey', linestyle=':')
        plt.axvline(-3700, c='dimgrey', linestyle=':')
        plt.axvline(-3100, c='dimgrey', linestyle=':')
        v = qsum[:].min() + 0.08*qsum[:].min()
        marge = v*0.02
        plt.plot([0, 2000], [v + marge, v + marge], c='grey')
        plt.plot([-3700, -3100], [v + marge, v + marge], c='grey')
        plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)
        plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)

        # perc_Rijnstr = str(round(Q_perc_Rijnstr, 2))
        # perc_region = str(round(Q_perc_region, 2))
        # plt.text(-3500, max(q[0, :]+2*marge), 'Fraction from Rijnstrangen: %s \n Fraction from region: %s' %(perc_Rijnstr, perc_region))

        plt.plot(X, q[0, :], label='q in first aquifer', c='darkred', linestyle='--')
        ## ONLY FOR MULTY LAYER SYSTEM:
        # plt.plot(X, q[1, :], label='q in second aquifer', c='royalblue', linestyle='-.')
        # plt.plot(X, q[2, :], label='q in third aquifer', c='darkgreen', linestyle=':')
        # plt.plot(X, qsum, label='Total q', c='black', linestyle='-', linewidth=2)

        plt.suptitle('Groundwater flux', fontsize=22, y=0.95)
        plt.title('positive = northward; negative = southward', fontsize=18)
        plt.xlabel('Distance along cross-section [m]', size=14)
        plt.ylabel('Flux [m$^2$/d]', size=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        leg = plt.legend(fontsize=14, loc='best')
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)
        leg.get_lines()[2].set_linewidth(4)
        ## ONLY FOR MULTY LAYER SYSTEM:
        # leg.get_lines()[3].set_linewidth(3.5)
        # leg.get_lines()[4].set_linewidth(3.5)
        # leg.get_lines()[5].set_linewidth(3.5)
        plt.show()

    if plot_s is True:
        plt.figure(figsize=(15, 10))
        plt.axvline(0, c='dimgrey', linestyle=':', label='Fixed section separation')
        plt.axvline(x_left, c='darkgrey', linestyle=':', label='Flexible section separation')
        plt.axvline(x_right, c='darkgrey', linestyle=':')
        plt.axvline(2000, c='dimgrey', linestyle=':')
        plt.axvline(3000, c='dimgrey', linestyle=':')
        plt.axvline(3000, c='dimgrey', linestyle=':')
        plt.axvline(-3700, c='dimgrey', linestyle=':')
        plt.axvline(-3100, c='dimgrey', linestyle=':')
        v = s[0,:].min() -0.005
        marge = -0.001
        plt.plot([0, 2000], [v + marge, v + marge], c='grey')
        plt.plot([-3700, -3100], [v + marge, v + marge], c='grey')
        plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)
        plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)

        plt.plot(X, s[0, :], label='Seepage from first aquifer', c='darkred', linestyle='--', linewidth=2)
        ## ONLY FOR MULTI LAYER SYSTEM
        # plt.plot(X, s[1, :], label='Seepage from second aquifer', c='royalblue', linestyle='-.', linewidth=2)
        # plt.plot(X, s[2, :], label='Seepage from third aquifer', c='darkgreen', linestyle=':', linewidth=2)

        plt.title('Seepage', fontsize=22)
        plt.xlabel('Distance along cross-section [m]')
        plt.ylabel('Seepage [m/d]')
        leg = plt.legend(fontsize=14, loc='best')
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)
        leg.get_lines()[2].set_linewidth(3.5)
        ## ONLY FOR MULTI LAYER SYSTEM
        # leg.get_lines()[3].set_linewidth(3.5)
        # leg.get_lines()[4].set_linewidth(3.5)
        plt.show()

    return qtot, qsum, x_left, x_right, Q_perc_region


def volume_Rijnstrangen(scenario, V_max, weir_width, weir_height, date_from, date_to, Q_GWleft, Q_GWright):
    ## options for scenario: "Historical", "Reference", "2050GL" and "2050WH"
    ## all other input values are defined in the parameter section below

    df_volume = read_climate(scenario=scenario, date_start=date_from, date_end=date_to)

    df_volume['P'] = df_volume.P / 1000 * A_tot / 1000000  ## mm/day to m/day to m3/day to Mm3/day
    df_volume['P_cumulative'] = df_volume.P * 0

    df_volume['WL'] = read_Rhine(scenario=scenario, date_start=date_from, date_end=date_to).WL

    df_volume['Q'] = df_volume.P * 0
    df_volume['Q_cumulative'] = df_volume.Q * 0

    df_volume['Outflow'] = df_volume.Q * 0
    df_volume['Outflow_cumulative'] = df_volume.Q * 0

    df_volume['E0'] = df_volume.Q * 0
    df_volume['E0_cumulative'] = df_volume.Q * 0

    df_volume['V'] = df_volume.Q * 0
    df_volume['V_cumulative'] = df_volume.Q * 0

    df_volume['h'] = df_volume.Q * 0

    df_volume['GW'] = df_volume.Q * 0
    df_volume['GW_cumulative'] = df_volume.Q * 0

    df_volume['perc_reg'] = df_volume.Q * 0

    gw_dict = {}        ## this is a dictionary
    perc_dict = {}

    min_h_rijnstrangen = 10.125
    max_h_rijnstrangen = 15.375
    step_size_h_rijnstrangen = 0.25

    gw_dict[10] = groundwater(10, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[0]
    perc_dict[10] = groundwater(10, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
    gw_dict[15] = groundwater(10, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[0]
    perc_dict[15] = groundwater(10, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]

    for h_rijnstrangen in np.arange(min_h_rijnstrangen, max_h_rijnstrangen + step_size_h_rijnstrangen, step_size_h_rijnstrangen):
        gw_dict[h_rijnstrangen] = groundwater(h_rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[0]
        perc_dict[h_rijnstrangen] = groundwater(h_rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]

    V = V_start
    E0_cumulative = 0
    GW_cumulative = 0
    P_cumulative = 0
    Outflow_cumulative = 0
    Q_cumulative = 0
    Q_daycount = 0

    for index, row in df_volume.iterrows():
        h = lookup_V(V)[0]
        row['h'] = h
        A_wet = lookup_V(V)[1]

        ## Makkink evaporation times correction factor gives evaporation in mm/day, to m/day, times m2 gives m3/day, to Mm3/day
        E0 = ((row['EV'] * c_EV_land / 1000 * (A_tot - A_wet)) + (row['EV'] * c_EV_water / 1000 * A_wet)) / 1000000
        row['E0'] = E0
        E0_cumulative += E0
        row['E0_cumulative'] = E0_cumulative

        P_cumulative += row['P']
        row['P_cumulative'] = P_cumulative

        if row['WL'] > weir_height and row['WL'] > row['h']:
            if V >= V_max:
                Q = 0
            else:
                h1 = row['WL'] - weir_height    ## h1 = upstream water level = water level Rhine
                h3 = row['h'] - weir_height     ## h3 = downstream water level = water level Rijnstrangen
                ## For free flowing weir (volkomen overlaat):
                if h3 <= 2/3*h1:
                    Q_old = 1.7 * discharge_coeff_free * weir_width * h1**(3/2)    ## m3/sec
                    Q = Q_old * 60 * 60 * 24 / 1000000                             ## Mm3/day
                ## For submerged weir flow (onvolkomen overlaat):
                if h3 > 2/3*h1:
                    Q_old = discharge_coeff_subm * weir_width * h3 * np.sqrt(2*9.81*(h1 - h3))   ## m3/sec
                    Q = Q_old * 60 * 60 * 24 / 1000000               ## Mm3/day

            row['Q'] = Q                                         ## store Q in dataframe

        GW = gw_dict[h]
        row['GW'] = GW
        GW_cumulative += GW
        row['GW_cumulative'] = GW_cumulative

        perc_reg = perc_dict[h]
        row['perc_reg'] = perc_reg

        V += row['Q'] - E0 + row['P'] - GW

        if row['Q'] > 0:
                h_new = lookup_V(V)[0]
                if h_new > row['WL']:
                    h_cor = row['WL']
                    V_cor = lookup_h(h_cor)[0]
                    Q_cor = row['Q'] - (V - V_cor)
                    if Q_cor > 0:
                        row['Q'] = row['Q'] - (V - V_cor)
                        V = V_cor

        if V >= V_max:
            row['Q'] = row['Q'] - (V - V_max)
            V = V_max

        if V <= 0:
            V = 0

        if row['Q'] < 0:
            row['Outflow'] = -row['Q']
            row['Q'] = 0
            Outflow_cumulative += row['Outflow']
        row['Outflow_cumulative'] = Outflow_cumulative

        if row['Q'] > 0:
            Q_daycount += row['Q']/row['Q']

        Q_cumulative += row['Q']
        row['Q_cumulative'] = Q_cumulative

        row['V'] = V

    # print('Precipitation mean:', round(np.mean(df_volume.P*1000000/A_tot*1000),3), 'mm/day')
    # print('Evaporation mean:', round(np.mean(df_volume.E0 * 1000000 / A_tot * 1000), 3), 'mm/day')
    # print('Inflow mean:', round(np.mean(df_volume.Q * 1000000 / A_tot * 1000), 3), 'mm/day')
    # print('Groundwater mean:', round(np.mean(df_volume.GW * 1000000 / A_tot * 1000), 3), 'mm/day')
    # print('Of which', round((Q_GWright + Q_GWleft) * 12000 / A_tot * 1000, 3), 'mm/day is extracted for water use')
    # print('Outflow mean:', round(np.mean(df_volume.Outflow * 1000000 / A_tot * 1000), 3), 'mm/day')
    # print(Q_daycount)
    # print('mean volume:', df_volume['V'].mean())
    return df_volume


def plot_volume_Rijnstrangen(scenario, date_from, date_to, Q_GWleft, Q_GWright, plottype):
    ## options for scenario: "Historical", "Reference", "2050GL" and "2050WH"
    ## options for plottype: "normal", "cumulative" and "multiple"
    ## all other input values are defined in the parameter section below

    if plottype == 'normal':
        df_volume = volume_Rijnstrangen(scenario=scenario, V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

        params = {'legend.fontsize': 14,
                  'axes.labelsize': 14,
                  'xtick.labelsize': 14,
                  'ytick.labelsize': 14}
        pylab.rcParams.update(params)

        ## plot water levels:
        plt.figure()
        plt.plot(df_volume.h)
        plt.title('Water level')

        ## plot percentage of extracted gw coming from surroundings
        plt.figure()
        plt.plot(df_volume.perc_reg)
        plt.ylim(-0.05, 1.05)
        plt.title('Fraction of extracted water \n coming from surrounding Region')

        ## plot whole water balance:
        # fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 0.5, 2]})
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

        # axs[1].plot(df_volume.perc_reg, color='purple')
        # axs[1].set_ylim(-0.05, 1.05)
        # axs[1].set_ylabel('Fraction from \n surroundings')

        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
        axs[0].plot(df_volume.V, color='red', label='Volume')
        axs[0].axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
        leg = axs[0].legend(loc='lower right')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(4)

        axs[1].set_ylabel('Fluxes [mm/day]')
        axs[1].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
        axs[1].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
        axs[1].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 10, color='blue',
                    label='Q$_{Rhine,net}$ / 10')
        axs[1].plot(-df_volume.GW * 1000000 / A_tot * 1000, color='black', label='Q$_{Groundwater}$')
        # axs[1].set_ylim(-12, 50)

        leg = axs[1].legend(loc='upper right')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(6)
        leg.get_lines()[2].set_linewidth(6)
        leg.get_lines()[3].set_linewidth(6)

        plt.show()

    if plottype == 'cumulative':
        df_volume = volume_Rijnstrangen(scenario=scenario, V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        plt.figure()
        plt.plot(df_volume.P_cumulative, color='lightblue', label='Q$_{Precipitation}$')
        plt.fill_between(df_volume.index, df_volume.P_cumulative, 0, color='lightblue', alpha=0.3)
        plt.plot((df_volume.Q_cumulative - df_volume.Outflow_cumulative) + df_volume.P_cumulative, color='blue', label='Q$_{Rhine,net}$')
        plt.fill_between(df_volume.index, (df_volume.Q_cumulative - df_volume.Outflow_cumulative) + df_volume.P_cumulative, df_volume.P_cumulative, color='blue', alpha=0.3)
        plt.plot(-df_volume.E0_cumulative, color='darkgreen', label='Q$_{Evaporation}$')
        plt.fill_between(df_volume.index, -df_volume.E0_cumulative, 0, color='darkgreen', alpha=0.3)
        plt.plot(-df_volume.GW_cumulative - df_volume.E0_cumulative, color='black', label='Q$_{Groundwater}$')
        plt.fill_between(df_volume.index, -df_volume.E0_cumulative, -df_volume.GW_cumulative - df_volume.E0_cumulative, color='black', alpha=0.3)
        plt.plot((df_volume.V) * 10, ':', c='red', label='Resulting volume Rijnstrangen $\cdot$ 10')
        plt.legend(loc='best')
        plt.ylabel('Volume [Mm$^3$]')
        leg = plt.legend()
        leg.get_lines()[0].set_linewidth(5)
        leg.get_lines()[1].set_linewidth(5)
        leg.get_lines()[2].set_linewidth(5)
        leg.get_lines()[3].set_linewidth(5)
        leg.get_lines()[4].set_linewidth(4)
        plt.show()

    if plottype == 'multiple':
        params = {'legend.fontsize': 14,
                  'axes.labelsize': 14,
                  'xtick.labelsize': 14,
                  'ytick.labelsize': 14}
        pylab.rcParams.update(params)

        df_volume_hist = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_volume_ref = volume_Rijnstrangen(scenario='Reference', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_volume_2050GL = volume_Rijnstrangen(scenario='2050GL', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_volume_2050WH = volume_Rijnstrangen(scenario='2050WH', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=date_from, date_to=date_to, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

        ## plot water levels:
        plt.figure()
        plt.plot(df_volume_hist.h, color='darkslategrey')
        plt.plot(df_volume_ref.h, color='royalblue')
        plt.plot(df_volume_2050GL.h, color='olivedrab')
        plt.plot(df_volume_2050WH.h, color='firebrick')
        plt.title('Water level')

        ## plot percentage of extracted gw coming from surroundings
        plt.figure()
        plt.plot(df_volume_hist.perc_reg, '.--', color='darkslategrey')
        plt.plot(df_volume_ref.perc_reg, '.--', color='royalblue')
        plt.plot(df_volume_2050GL.perc_reg, '.--', color='olivedrab')
        plt.plot(df_volume_2050WH.perc_reg, '.--', color='firebrick')
        plt.ylim(-0.05, 1.05)
        plt.title('Fraction of extracted water \n coming from surrounding Region')

        ## plot WB only
        plt.figure(figsize=(16,4))
        plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
        plt.plot(df_volume_hist.V, color='darkslategrey', label='Historical')
        plt.plot(df_volume_ref.V, color='royalblue', label='Reference')
        plt.plot(df_volume_2050GL.V, color='olivedrab', label='2050 G$_L$')
        plt.plot(df_volume_2050WH.V, color='firebrick', label='2050 W$_H$')
        plt.ylim(-3, 63)
        plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
        leg = plt.legend(loc='lower right')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(6)
        leg.get_lines()[2].set_linewidth(6)
        leg.get_lines()[3].set_linewidth(6)
        leg.get_lines()[4].set_linewidth(3)

        ## plot WB-duration line only
        df_descending_hist = df_volume_hist.sort_values('V', ascending=False)
        df_descending_hist.index = np.linspace(0, len(df_descending_hist.V), len(df_descending_hist.V)) / len(df_descending_hist.V) * 100

        df_descending_ref = df_volume_ref.sort_values('V', ascending=False)
        df_descending_ref.index = np.linspace(0, len(df_descending_ref.V), len(df_descending_ref.V)) / len(df_descending_ref.V) * 100

        df_descending_2050GL = df_volume_2050GL.sort_values('V', ascending=False)
        df_descending_2050GL.index = np.linspace(0, len(df_descending_2050GL.V), len(df_descending_2050GL.V)) / len(df_descending_2050GL.V) * 100

        df_descending_2050WH = df_volume_2050WH.sort_values('V', ascending=False)
        df_descending_2050WH.index = np.linspace(0, len(df_descending_2050WH.V), len(df_descending_2050WH.V)) / len(df_descending_2050WH.V) * 100

        plt.figure(figsize=(13, 8))
        plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
        plt.xlabel('Percentage of time volume is exceeded')
        plt.plot(df_descending_hist.V, color='darkslategrey', label='Historical', linewidth=2)
        plt.plot(df_descending_ref.V, color='royalblue', label='Reference', linewidth=2)
        plt.plot(df_descending_2050GL.V, color='olivedrab', label='2050 G$_L$', linewidth=2)
        plt.plot(df_descending_2050WH.V, color='firebrick', label='2050 W$_H$', linewidth=2)
        plt.ylim(-3, 62)
        plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
        leg = plt.legend(loc='best')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(6)
        leg.get_lines()[2].set_linewidth(6)
        leg.get_lines()[3].set_linewidth(6)
        leg.get_lines()[4].set_linewidth(3)

        ## plot whole water balance with fluxes:
        fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})
        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
        axs[0].plot(df_volume_hist.V, color='darkslategrey', label='Historical')
        axs[0].plot(df_volume_ref.V, color='royalblue', label='Reference')
        axs[0].plot(df_volume_2050GL.V, color='olivedrab', label='2050 G$_L$')
        axs[0].plot(df_volume_2050WH.V, color='firebrick', label='2050 W$_H$')
        axs[0].set_ylim(-3, 63)
        axs[0].axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
        leg = axs[0].legend(loc='lower right')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(6)
        leg.get_lines()[2].set_linewidth(6)
        leg.get_lines()[3].set_linewidth(6)
        leg.get_lines()[4].set_linewidth(3)

        axs[1].set_ylabel('Fluxes [mm/day]')
        axs[1].plot(df_volume_hist.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
        axs[1].plot(-df_volume_hist.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
        axs[1].plot((df_volume_hist.Q - df_volume_hist.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue', label='Q$_{Rhine,net}$ / 50')
        axs[1].plot(-df_volume_hist.GW * 1000000 / A_tot * 1000, color='black', label='Q$_{Groundwater}$')

        leg = axs[1].legend(loc='upper right')
        leg.get_lines()[0].set_linewidth(6)
        leg.get_lines()[1].set_linewidth(6)
        leg.get_lines()[2].set_linewidth(6)
        leg.get_lines()[3].set_linewidth(6)

        plt.show()
    return


def plot_volume_narratives(V_max, Q_GW, date_from, date_to):
    df_volume = volume_Rijnstrangen(scenario='2050GL', V_max=V_max, weir_width=weir_width, weir_height=weir_height,
                                    date_from=date_from, date_to=date_to, Q_GWleft=Q_GW, Q_GWright=Q_GW)

    params = {'legend.fontsize': 14,
              'axes.labelsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)

    ## plot water levels:
    plt.figure()
    plt.plot(df_volume.h)
    plt.title('Water level')

    ## plot percentage of extracted gw coming from surroundings
    plt.figure()
    plt.plot(df_volume.perc_reg)
    plt.ylim(-0.05, 1.05)
    plt.title('Fraction of extracted water \n coming from surrounding Region')

    ## plot whole water balance:
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 2]})
    # print('Vmax:', V_max, 'Quse:', 2*Q_GW*365*12000, 'Average percentage from region:', round(np.mean(df_volume.perc_reg),4)*100)

    axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    axs[0].plot(df_volume.V, color='red', label='Volume')
    axs[0].axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
    leg = axs[0].legend(loc='lower right')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(4)

    axs[1].plot(df_volume.perc_reg, color='purple')
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_ylabel('Fraction from \n surroundings')

    axs[2].set_ylabel('Fluxes [mm/day]')
    axs[2].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
    axs[2].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
    axs[2].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue',
                label='Q$_{Rhine,net}$ / 50')
    axs[2].plot(-df_volume.GW * 1000000 / A_tot * 1000, color='black', label='Q$_{Groundwater}$')

    leg = axs[2].legend(loc='upper right')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(6)
    leg.get_lines()[2].set_linewidth(6)
    leg.get_lines()[3].set_linewidth(6)

    plt.show()


## PARAMETERS (don't change):
A_tot = 28646820
discharge_coeff_free = 1
discharge_coeff_subm = 1
c_EV_land = 0.85
c_EV_water = 1.25


## VARIABLES (change :-) ):
## time span
start = 1960
end = 2009

## weir configuration
weir_width = 500                ## 500 = neutral
weir_height = 12.5              ## 12.5 = neutral

## basin
V_max = 58                      ## 58 = neutral = corresponding to water level of 14.0 m +NAP
V_start = 22300000 / 1000000    ## 22.3 = neutral = corresponding to water level of 12.5 m +NAP

## water use
Q_GWleft = -25000000/365/12000  ## -25
Q_GWright = -25000000/365/12000 ## -25
# Q_GWleft = 0
# Q_GWright = 0


# groundwater(h_Rijnstrangen=11.0, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=True, plot_s=False)
# plot_volume_Rijnstrangen(scenario='Historical', plottype='multiple', date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
# plot_volume_narratives(V_max, Q_GW=Q_GWleft, date_from=start, date_to=end)


## This function plots the groundwater flow for multiple Quse values.
def GW_multiple_Quse_plot(h_Rijnstrangen):
    for q in np.arange(50000000, 0, -12500000):
        Q_GW = -q / 365 / 12000
        qsum = groundwater(h_Rijnstrangen=h_Rijnstrangen, Q_GWleft=Q_GW, Q_GWright=Q_GW, plot_phi=False, plot_q=False, plot_s=False)[1]
        X = np.arange(-3700, 5510, 10)
        plt.plot(X, qsum, label='GW flow for Q$_{use}$ = 2 x ' + str(Q_GW*12000*365/1000000) + ' Mm$^3$/year', linestyle='-', linewidth=2)

    for q in np.arange(100000000, 0, -50000000):
        Q_GW = -q / 365 / 12000
        qsum = groundwater(h_Rijnstrangen=h_Rijnstrangen, Q_GWleft=Q_GW, Q_GWright=0, plot_phi=False, plot_q=False, plot_s=False)[1]
        X = np.arange(-3700, 5510, 10)
        plt.plot(X, qsum, label='GW flow for Q$_{use, left}$ = ' + str(Q_GW*12000*365/1000000) + ' Mm$^3$/year', linestyle='-', linewidth=2)

    for q in np.arange(50000000, 0, -25000000):
        Q_GW = -q / 365 / 12000
        qsum = groundwater(h_Rijnstrangen=h_Rijnstrangen, Q_GWleft=0, Q_GWright=Q_GW, plot_phi=False, plot_q=False, plot_s=False)[1]
        X = np.arange(-3700, 5510, 10)
        plt.plot(X, qsum, label='GW flow for Q$_{use, right}$ = ' + str(Q_GW*12000*365/1000000) + ' Mm$^3$/year', linestyle='-', linewidth=2)

    for q in np.arange(0, -1, -1):
        Q_GW = -q / 365 / 12000
        qsum = groundwater(h_Rijnstrangen=h_Rijnstrangen, Q_GWleft=0, Q_GWright=Q_GW, plot_phi=False, plot_q=False, plot_s=False)[1]
        X = np.arange(-3700, 5510, 10)
        plt.plot(X, qsum, label='GW flow for Q$_{use}$ = ' + str(Q_GW*12000*365/1000000) + ' Mm$^3$/year', linestyle='-', linewidth=2)

    plt.axvline(0, c='dimgrey', linestyle=':', label='Section separation')
    plt.axvline(2000, c='dimgrey', linestyle=':')
    plt.axvline(3000, c='dimgrey', linestyle=':')
    plt.axvline(5500, c='dimgrey', linestyle=':')
    plt.axvline(-3700, c='dimgrey', linestyle=':')
    plt.axvline(-3100, c='dimgrey', linestyle=':')
    v = -10
    marge = v * 0.01
    plt.plot([0, 2000], [v + marge, v + marge], c='grey')
    plt.plot([-3700, -3100], [v + marge, v + marge], c='grey')
    plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)
    plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)

    plt.suptitle('Groundwater flux', fontsize=22, y=0.95)
    plt.title('h$_{Rijnstrangen}$ = ' + str(h_Rijnstrangen) + ' m+NAP; positive = northward; negative = southward', fontsize=18)
    plt.xlabel('Distance along cross-section [m]', size=14)
    plt.ylabel('Flux [m2/d]', size=14)
    plt.legend(fontsize=14, loc='best')

    plt.show()
# GW_multiple_Quse_plot(12.5)


## This function plots the water balance for multiple Quse values.
def WB_multiple_Quse_plot():
    params = {'legend.fontsize': 14,
              'axes.labelsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)


    ## With fluxes:
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})

    df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

    axs[1].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
    axs[1].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
    axs[1].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue',
                label='Q$_{Rhine,net}$ / 50')

    for q in np.arange(50000000, -1, -10000000):
        Q_GW = -q/365/12000
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GW, Q_GWright=Q_GW)
        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]', size=14)
        axs[0].plot(df_volume.V, label = 'Q$_{use}$ = 2 x ' + str(q/1000000) + ' Mm$^3$/y')
        axs[1].plot(-df_volume.GW * 1000000 / A_tot * 1000, label='Q$_{GW}$ for Q$_{use}$ = 2 x ' + str(q/1000000) + ' Mm$^3$/y')

    leg1 = axs[0].legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)
    axs[1].set_ylabel('Fluxes [mm/day]', size=14)
    axs[0].set_ylim(-3, 63)

    leg = plt.legend(loc='upper right', fontsize=14)
    leg.get_lines()[0].set_linewidth(4)
    leg.get_lines()[1].set_linewidth(4)
    leg.get_lines()[2].set_linewidth(4)
    leg.get_lines()[3].set_linewidth(4)
    leg.get_lines()[4].set_linewidth(4)
    leg.get_lines()[5].set_linewidth(4)
    leg.get_lines()[6].set_linewidth(4)
    leg.get_lines()[7].set_linewidth(4)
    leg.get_lines()[8].set_linewidth(4)


    ## Without fluxes:
    plt.figure(figsize=(16, 4))
    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.ylim(-3, 63)
    for q in np.arange(50000000, -1, -10000000):
        Q_GW = -q/365/12000
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GW, Q_GWright=Q_GW)
        plt.plot(df_volume.V, label = 'Q$_{use}$ = 2 x ' + str(q/1000000) + ' Mm$^3$/y')

    leg1 = plt.legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)


    ## Duration line
    plt.figure(figsize=(13, 8))
    for q in np.arange(50000000, -1, -10000000):
        Q_GW = -q/365/12000
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GW, Q_GWright=Q_GW)
        df_descending = df_volume.sort_values('V', ascending=False)
        df_descending.index = np.linspace(0, len(df_descending.V), len(df_descending.V)) / len(df_descending.V) * 100
        plt.plot(df_descending.V, label='Q$_{use}$ = 2 x ' + str(q/1000000) + ' Mm$^3$/y', linewidth=2)


    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.xlabel('Percentage of time volume is exceeded')
    plt.ylim(-3, 62)
    plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
    leg = plt.legend(loc='best')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(6)
    leg.get_lines()[2].set_linewidth(6)
    leg.get_lines()[3].set_linewidth(6)
    leg.get_lines()[4].set_linewidth(6)
    leg.get_lines()[5].set_linewidth(6)
    leg.get_lines()[6].set_linewidth(3)


    plt.show()
# WB_multiple_Quse_plot()


## This funtion plots the water balance for multiple Vmax values.
def WB_multiple_Vmax_plot():
    params = {'legend.fontsize': 18,
              'axes.labelsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18}
    pylab.rcParams.update(params)


    ## With fluxes:
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})

    df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

    axs[1].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
    axs[1].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
    axs[1].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue', label='Q$_{Rhine,net}$ / 50')

    for Vmax in np.arange(100, 0, -20):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=Vmax, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]', size=14)
        axs[0].plot(df_volume.V, label = 'V$_{max}$ = ' + str(Vmax) + ' Mm$^3$')
        axs[1].plot(-df_volume.GW * 1000000 / A_tot * 1000, label='Q$_{GW}$ for V$_{max}$ = ' + str(Vmax) + ' Mm$^3$')

    leg1 = axs[0].legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    axs[1].set_ylabel('Fluxes [mm/day]', size=14)
    # axs[1].set_ylim(-12, 50)
    axs[0].set_ylim(-5, 105)

    leg = plt.legend(loc='upper right', fontsize=14)
    leg.get_lines()[0].set_linewidth(4)
    leg.get_lines()[1].set_linewidth(4)
    leg.get_lines()[2].set_linewidth(4)
    leg.get_lines()[3].set_linewidth(4)
    leg.get_lines()[4].set_linewidth(4)
    leg.get_lines()[5].set_linewidth(4)
    leg.get_lines()[6].set_linewidth(4)
    leg.get_lines()[7].set_linewidth(4)


    ## Without fluxes:
    plt.figure(figsize=(16, 4))
    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    for Vmax in np.arange(100, 0, -20):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=Vmax, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        plt.plot(df_volume.V, label = 'V$_{max}$ = ' + str(Vmax) + ' Mm$^3$')

    leg1 = plt.legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)


    ## Duration line
    plt.figure(figsize=(13, 8))
    for Vmax in np.arange(100, 0, -20):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=Vmax, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_descending = df_volume.sort_values('V', ascending=False)
        df_descending.index = np.linspace(0, len(df_descending.V), len(df_descending.V)) / len(df_descending.V) * 100
        plt.plot(df_descending.loc[0:100, 'V'], label='V$_{max}$ = ' + str(Vmax) + ' Mm$^3$', linewidth=3)

    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.xlabel('Percentage of time volume is exceeded')
    plt.ylim(-3, 62)
    plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
    leg = plt.legend(loc='best')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(6)
    leg.get_lines()[2].set_linewidth(6)
    leg.get_lines()[3].set_linewidth(6)
    leg.get_lines()[4].set_linewidth(6)
    leg.get_lines()[5].set_linewidth(6)
    leg.get_lines()[6].set_linewidth(3)

    plt.show()
# WB_multiple_Vmax_plot()


## This function plots the water balance for multiple weir widths.
def WB_multiple_width_plot():
    params = {'legend.fontsize': 14,
              'axes.labelsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)


    ## with fluxes:
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})

    df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

    axs[1].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
    axs[1].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
    axs[1].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue', label='Q$_{Rhine,net}$ / 50')
    axs[1].plot(-df_volume.GW * 1000000 / A_tot * 1000, color='black', label='Q$_{groundwater}$')

    for width in np.arange(1200, 0, -200):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]', size=14)
        axs[0].plot(df_volume.V, label = 'Weir width = ' + str(width) + ' m')

    leg1 = axs[0].legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)
    axs[1].set_ylabel('Fluxes [mm/day]', size=14)
    # axs[1].set_ylim(-12, 50)
    axs[0].set_ylim(-3, 63)

    leg = plt.legend(loc='upper right', fontsize=14)
    leg.get_lines()[0].set_linewidth(4)
    leg.get_lines()[1].set_linewidth(4)
    leg.get_lines()[2].set_linewidth(4)
    leg.get_lines()[3].set_linewidth(4)


    ## Without fluxes:
    plt.figure(figsize=(16, 4))
    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.ylim(-3, 63)
    for width in [250, 200, 150, 100, 50]:
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        plt.plot(df_volume.V, label = 'Weir width = ' + str(width) + ' m')

    leg1 = plt.legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)


    ## Duration line
    plt.figure(figsize=(13, 8))
    for width in [500, 250, 100, 50, 25, 10]:
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_descending = df_volume.sort_values('V', ascending=False)
        df_descending.index = np.linspace(0, len(df_descending.V), len(df_descending.V)) / len(df_descending.V) * 100
        plt.plot(df_descending.loc[0:100, 'V'], label='Weir width = ' + str(width) + ' m', linewidth=3)

    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.xlabel('Percentage of time volume is exceeded')
    plt.ylim(-3, 62)
    plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
    leg = plt.legend(loc='best')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(6)
    leg.get_lines()[2].set_linewidth(6)
    leg.get_lines()[3].set_linewidth(6)
    leg.get_lines()[4].set_linewidth(6)
    leg.get_lines()[5].set_linewidth(6)
    leg.get_lines()[6].set_linewidth(3)

    plt.show()
# WB_multiple_width_plot()


## This function plots the water balance for multiple weir heights.
def WB_multiple_height_plot():
    params = {'legend.fontsize': 14,
              'axes.labelsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)

    ## With fluxes:
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})

    df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=weir_height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)

    axs[1].plot(df_volume.P * 1000000 / A_tot * 1000, color='lightblue', label='Q$_{Precipitation}$')
    axs[1].plot(-df_volume.E0 * 1000000 / A_tot * 1000, color='darkgreen', label='Q$_{Evaporation}$')
    axs[1].plot((df_volume.Q - df_volume.Outflow) * 1000000 / A_tot * 1000 / 50, color='blue', label='Q$_{Rhine,net}$ / 50')
    axs[1].plot(-df_volume.GW * 1000000 / A_tot * 1000, color='black', label='Q$_{groundwater}$')

    for height in np.arange(15, 9, -1):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        axs[0].set_ylabel('Volume in \n Rijnstrangen [Mm$^3$]', size=14)
        axs[0].plot(df_volume.V, label = 'Weir height = ' + str(height) + ' m')

    leg1 = axs[0].legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)
    axs[1].set_ylabel('Fluxes [mm/day]', size=14)
    axs[0].set_ylim(-3, 63)

    leg = plt.legend(loc='upper right', fontsize=14)
    leg.get_lines()[0].set_linewidth(4)
    leg.get_lines()[1].set_linewidth(4)
    leg.get_lines()[2].set_linewidth(4)
    leg.get_lines()[3].set_linewidth(4)

    ## Without fluxes:
    plt.figure(figsize=(16, 4))
    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.ylim(-3, 63)
    for height in np.arange(15, 9, -1):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        plt.plot(df_volume.V, label = 'Weir height = ' + str(height) + ' m')

    leg1 = plt.legend(loc='lower right', fontsize=14)
    leg1.get_lines()[0].set_linewidth(4)
    leg1.get_lines()[1].set_linewidth(4)
    leg1.get_lines()[2].set_linewidth(4)
    leg1.get_lines()[3].set_linewidth(4)
    leg1.get_lines()[4].set_linewidth(4)
    leg1.get_lines()[5].set_linewidth(4)

    ## Duration line
    plt.figure(figsize=(13, 8))
    for height in np.arange(15, 9, -1):
        df_volume = volume_Rijnstrangen(scenario='Historical', V_max=V_max, weir_width=weir_width, weir_height=height, date_from=start, date_to=end, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright)
        df_descending = df_volume.sort_values('V', ascending=False)
        df_descending.index = np.linspace(0, len(df_descending.V), len(df_descending.V)) / len(df_descending.V) * 100
        plt.plot(df_descending.loc[0:100, 'V'], label='Weir height = ' + str(height) + ' m', linewidth=3)

    plt.ylabel('Volume in \n Rijnstrangen [Mm$^3$]')
    plt.xlabel('Percentage of time volume is exceeded')
    plt.ylim(-3, 62)
    plt.axhline(y=V_max, xmin=0.05, xmax=0.95, color='darksalmon', linestyle='--', label='Max. volume')
    leg = plt.legend(loc='best')
    leg.get_lines()[0].set_linewidth(6)
    leg.get_lines()[1].set_linewidth(6)
    leg.get_lines()[2].set_linewidth(6)
    leg.get_lines()[3].set_linewidth(6)
    leg.get_lines()[4].set_linewidth(6)
    leg.get_lines()[5].set_linewidth(6)
    leg.get_lines()[6].set_linewidth(3)


    plt.show()
# WB_multiple_height_plot()


## This function prints or plots the percentages of the extraction originating from the region around the Rijnstrangen.
def function_percentages():
    ## PRINT
    for V_max in [0.56, 1.21, 3.6, 7.66, 13.9, 22.32, 32.9, 45.12, 58.4, 72.1, 86.0, 100]:
        for Q_GW in [-2500000/365/12000, -5000000/365/12000, -10000000/365/12000, -25000000/365/12000, -50000000/365/12000]:
            plot_volume_narratives(V_max, Q_GW, date_from=start, date_to=end)  ## comment everything except the "print" part in the plot_volume_narratives function

    ## PLOT
    # V_max = 7.7
    # Q_GW = -2000000/365/12000
    # print(V_max, Q_GW)
    # for width in [500, 400, 300, 200, 100]:
    #     for height in [15, 14, 13, 12, 11, 10]:
    #         weir_width = width
    #         weir_height = height
    #         plot_volume_narratives(V_max, Q_GW, start,end)
    #
    # V_max = 22.3
    # Q_GW = -25000000/365/12000
    # print(V_max, Q_GW)
    # for width in [500, 400, 300, 200, 100]:
    #     for height in [15, 14, 13, 12, 11, 10]:
    #         weir_width = width
    #         weir_height = height
    #         plot_volume_narratives(V_max, Q_GW, start,end)
    #
    # V_max = 72
    # Q_GW = -57000000/365/12000
    # print(V_max, Q_GW)
    # for width in [500, 400, 300, 200, 100]:
    #     for height in [15, 14, 13, 12, 11, 10]:
    #         weir_width = width
    #         weir_height = height
    #         plot_volume_narratives(V_max, Q_GW, start,end)
# function_percentages()


## This function plots the relation between the percentage extracted from the region and the water level for various extraction volumes.
def Perc_multiple_Quse_plot():
    params = {'legend.fontsize': 14,
              'axes.labelsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14}
    pylab.rcParams.update(params)

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -10000000 / 365 / 12000
        Q_GWright = -10000000 / 365 / 12000
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'x:', markersize=10, color='dodgerblue', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = -10 Mm$^3$/year, Q$_{right}$ = -10 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='blue', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = -10 Mm$^3$/year, Q$_{right}$ = -10 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -20000000 / 365 / 12000
        Q_GWright = 0
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, '*:', markersize=10, color='dodgerblue', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = -20 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='green', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = -20 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = 0
        Q_GWright = -20000000 / 365 / 12000
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'o:', markersize=10, color='dodgerblue', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -20 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='dodgerblue', label='Q$_{tot}$ = -20 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -20 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -25000000 / 365 / 12000
        Q_GWright = -25000000 / 365 / 12000
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'x:', markersize=10, color='purple', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -25 Mm$^3$/year, Q$_{right}$ = -25 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='darkorange', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -25 Mm$^3$/year, Q$_{right}$ = -25 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = 0
        Q_GWright = -50000000 / 365 / 12000
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, '*:', markersize=10, color='purple', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -50 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='red', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -50 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -50000000 / 365 / 12000
        Q_GWright = 0
        perc_reg = groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'o:', markersize=10, color='purple', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -50 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='purple', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -50 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -50000000 / 365 / 12000
        Q_GWright = -50000000 / 365 / 12000
        perc_reg = \
        groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'x:', markersize=10, color='darkorange',
             label='Q$_{tot}$ = -100 Mm$^3$/year, Q$_{left}$ = -50 Mm$^3$/year, Q$_{right}$ = -50 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='darkorange', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -25 Mm$^3$/year, Q$_{right}$ = -25 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = 0
        Q_GWright = -100000000 / 365 / 12000
        perc_reg = \
        groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, '*:', markersize=10, color='darkorange',
             label='Q$_{tot}$ = -100 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -100 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='red', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = 0.0 Mm$^3$/year, Q$_{right}$ = -50 Mm$^3$/year')

    perc_list = []
    h_list = []
    for h in np.arange(9.75, 15.51, 0.25):
        Q_GWleft = -100000000 / 365 / 12000
        Q_GWright = 0
        perc_reg = \
        groundwater(h_Rijnstrangen=h, Q_GWleft=Q_GWleft, Q_GWright=Q_GWright, plot_phi=False, plot_q=False, plot_s=False)[4]
        perc_list.append(perc_reg)
        h_list.append(h)
    plt.plot(h_list, perc_list, 'o:', markersize=10, color='darkorange',
             label='Q$_{tot}$ = -100 Mm$^3$/year, Q$_{left}$ = -100 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')
    # plt.plot(15, 0, 'o', color='purple', label='Q$_{tot}$ = -50 Mm$^3$/year, Q$_{left}$ = -50 Mm$^3$/year, Q$_{right}$ = 0.0 Mm$^3$/year')

    plt.xlabel('h in Rijnstrangen [m]')
    plt.ylabel('Fraction extracted water from region (not coming from reservoir)')
    plt.legend(loc='best')

    plt.show()
# Perc_multiple_Quse_plot()