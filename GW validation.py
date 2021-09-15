import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.linalg import expm, sqrtm
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')
import random


data_lookup = pd.read_csv('Vol&Area_vs_depth_CSV.csv', header=0, sep=',')
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


def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright):
    A_wet = lookup_h(h_Rijnstrangen)[1]
    A_tot = 28646820
    fraction = A_wet / A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left

    ## Q, (nNod by nSec) matrix of nodal injections [L2/T]
    Q = np.array([[0, Q_GWleft, 0, 0, Q_GWright, 0]])

    ## x, (nNod by 1) vector of coordinates of intersection points except +/-inf
    x = np.array([-3100, 0, x_left, x_right, 2000, 3000])

    ## X, vector of points where values will be computed
    X = np.arange(-3700, 5510, 10)

    nLay = len(kD[:, 0])
    nSec = len(kD[0, :])

    # include the outer sections to infinity
    a = np.zeros((nLay, 1))
    Q = np.concatenate((a, Q, a), axis=1)
    x = np.append(x, math.inf)
    x = np.append(-math.inf, x)
    Nx = len(x)
    H = np.ones((nLay, 1)) * heads

    # Mid-section points are used to compute relative coordinates within sections
    xMidSec = 0.5 * (x[:-1] + x[1:])
    xMidSec[0] = x[1]
    xMidSec[-1] = x[-2]

    # System matrices for all sections
    A = np.zeros((nLay, nLay, nSec))
    for iSec in range(nSec):
        a = 1 / (kD[:, iSec] * c[:, iSec])
        p = np.append(c[1:nLay, iSec], math.inf)
        b = 1 / (kD[:, iSec] * p)
        A[:, :, iSec] = np.diag(a + b) - np.diag(a[1:nLay], -1) - np.diag(b[0:nLay - 1], 1)

    # Generating and filling the coefficient matrix C
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  # coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  # right hand side vector

    nNod = nSec - 1
    for i in range(nNod):
        # i is section N, also the left node number of the section
        # j is right the node number of the section
        # ii and jj point to the position within the total coefficient matrix
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

    # Solve the system, using all layers and leaving out the outer column as they have no freedom, because the sections extend to infinity
    COEF = np.vstack(spsolve((C[:, nLay:-nLay]), R))
    COEF = np.concatenate((np.zeros((nLay, 1)), COEF, np.zeros((nLay, 1))))

    # output:
    # phi [H] = computed heads, a (nLay by length(X)) matrix
    # q [L2/T] = computed flows, a (nLay by length(X)) matrix
    # s [L/T] = downward positive seepage rate through top of each layer, a nLay by length(X) matrix
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

    # Multi layer system:
    # qsum = q[0, :] + q[1, :] + q[2, :]

    # One layer system:
    qsum = q[0, :]

    qleft_Rijnstr = qsum[371]                                   # Q left of Rijnstrangen
    qright_Rijnstr = qsum[570]                                  # Q right of Rijnstrangen
    qtot = (qright_Rijnstr - qleft_Rijnstr) * 12000 / 1000000   # gives q_total out of Rijnstrangen in Mm3/d

    return X, phi, x_left, x_right, qtot

plt.figure(figsize=(15, 10))
b = []

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(650, 1690)*0.01
    h3 = random.randint(1000, 1070)*0.01
    h3right = random.randint(850, 1150)*0.01
    h4 = random.randint(850, 1150)*0.01
    h5 = random.randint(750, 1050)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(700, 1600)*0.01
    h3 = random.randint(1000, 1070)*0.01
    h3right = random.randint(850, 1150)*0.01
    h4 = random.randint(850, 1150)*0.01
    h5 = random.randint(750, 1050)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(750, 1550)*0.01
    h3 = random.randint(1010, 1060)*0.01
    h3right = random.randint(875, 1125)*0.01
    h4 = random.randint(875, 1125)*0.01
    h5 = random.randint(750, 1050)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(800, 1500)*0.01
    h3 = random.randint(1010, 1050)*0.01
    h3right = random.randint(900, 1100)*0.01
    h4 = random.randint(900, 1100)*0.01
    h5 = random.randint(800, 1000)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(750, 1400)*0.01
    h3 = random.randint(1010, 1060)*0.01
    h3right = random.randint(875, 1125)*0.01
    h4 = random.randint(875, 1125)*0.01
    h5 = random.randint(750, 1050)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

for i in range(100):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,5000)*0.01
    c2 = random.randint(575,675)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(50,125)
    c3right = random.randint(1000, 10000)
    c4 = random.randint(225,300)
    c5 = random.randint(175,250)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(100,500)
    kD2 = random.randint(250,1000)
    kD3left = random.randint(250, 1000)
    kD3 = random.randint(250, 1000)
    kD3right = random.randint(250, 1000)
    kD4 = random.randint(250, 1000)
    kD5 = random.randint(250, 1000)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(750, 1300)*0.01
    h3 = random.randint(1010, 1060)*0.01
    h3right = random.randint(875, 1125)*0.01
    h4 = random.randint(875, 1125)*0.01
    h5 = random.randint(750, 1050)*0.01
    h2 = (h1+h3)/2
    h3left = h2

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0
    Q_GWright = 0
    X, phi, x_left, x_right, a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    plt.plot(X, phi[0, :], c='darksalmon', alpha=0.2, linewidth=2)

    print(i)

c = np.array([[1, 650, 10000, 80, 10000, 250, 225]])
kD = np.array([[250, 750, 750, 750, 750, 750, 750]])
heads = np.array([9.2, 9.8, 9.8, 10.4, 9.7, 9.7, 9.1])
X, phi, x_left, x_right, qtot = groundwater(10.4, 0, 0)

a = 0   # for RMSE calculation
b = 0

plt.plot(-3700, 9.3, 'o', c='black', label='Average head in first aquifer (measured)')  # average value (depth tov maaiveld: 9.05-10.05m) B40D2181
a += (phi[:, 0]-9.3)**2
b += 1
plt.plot(-3700, 9.059, 'o', c='black')  # average value (depth tov maaiveld: 3.82-4.82m) B40D2179
a += (phi[:, 0]-9.059)**2
b += 1
plt.plot(-3700, 9.195, 'o', c='black')  # average value (depth tov maaiveld: 7.47-8.47m) B40D2180
a += (phi[:, 0]-9.195)**2
b += 1
plt.plot(-3700, 9.3, 'o', c='black')  # average value (depth tov maaiveld: 0.93-4.09m) B40D1157
a += (phi[:, 0]-9.3)**2
b += 1
plt.plot(-1750, 9.7, 'o', c='grey', label='Average head in first aquifer (measured), \n data from before 2000')  # average value (depth tov maaiveld: 3.97-4.97m) B40G1186 LETOP weinig meetpunten
a += (phi[:, 195]-9.7)**2
b += 1
plt.plot(-1750, 9.8, 'o', c='grey')  # average value (depth tov maaiveld: 3.28-3.78m) B40G0267 LETOP oud (1975-2000)
a += (phi[:, 195]-9.8)**2
b += 1
plt.plot(-700, 9.9, 'o', c='black')  # average value (depth tov maaiveld: 2.48-3.48m) B40G1055 (ook representatief voor het zooitje peilbuizen daaromheen)
a += (phi[:, 300]-9.9)**2
b += 1
plt.plot(-400, 9.9, 'o', c='black')  # average value (depth tov maaiveld: 1.7-2.7m) B40G1069
a += (phi[:, 330]-9.9)**2
b += 1
plt.plot(-50, 10.07, 'o', c='black')  # average value (depth tov maaiveld: 3.4-4.4m) B40G1166
a += (phi[:, 365]-10.07)**2
b += 1
plt.plot((50, 50, 50), (10.0, 9.94, 10.15), 'o', c='black')  # average value (depth tov maaiveld: 4.2-5.2m) B40G1054/B40G1167/B40G1054
a += (phi[:, 375]-10)**2
a += (phi[:, 375]-9.94)**2
a += (phi[:, 375]-10.15)**2
b += 1
b += 1
b += 1
plt.plot(1000, 10.11, 'o', c='black')  # average value (depth tov maaiveld: 1.6-2.6m/2.5-5m) B40G0214
a += (phi[:, 470]-10.11)**2
b += 1
plt.plot(1000, 10.2, 'o', c='black')  # average value (depth tov maaiveld: 2.42-4.42m) B40G1053 LETOP semi-oud (1994-2008)
a += (phi[:, 470]-10.2)**2
b += 1
plt.plot(1000, 10.15, 'o', c='black')  # average value (depth tov maaiveld: 4-5m) B40G0214 LETOP alleen recente deel met veel metingen meegenomen
a += (phi[:, 470]-10.15)**2
b += 1
plt.plot(2050, 10.07, 'o', c='black')  # average value (depth tov maaiveld: 1.95-2.95m) B40G1066
a += (phi[:, 575]-10.07)**2
b += 1
plt.plot(2050, 10.12, 'o', c='black')  # average value (depth tov maaiveld: 1.95-2.95m) B40G1193
a += (phi[:, 575]-10.12)**2
b += 1
plt.plot(2050, 9.93, 'o', c='black')  # average value (depth tov maaiveld: 1.95-2.95m) B40G1192
a += (phi[:, 575]-9.93)**2
b += 1
plt.plot(2050, 9.74, 'o', c='black')  # average value (depth tov maaiveld: ??-??m) B40G1079 LETOP geen diepte bekend
a += (phi[:, 575]-9.74)**2
b += 1
plt.plot(2700, 9.48, 'o', c='black')  # average value (depth tov maaiveld: 2.9-3.4m) B40G0341
a += (phi[:, 640]-9.48)**2
b += 1
plt.plot(2700, 9.58, 'o', c='black')  # average value (depth tov maaiveld: 2.9-3.4m) B40G1190
a += (phi[:, 640]-9.58)**2
b += 1
plt.plot(4000, 9.03, 'o', c='black')  # average value (depth tov maaiveld: 2.99-3.99m) B40E1491
a += (phi[:, 770]-9.03)**2
b += 1
plt.plot(4000, 9.16, 'o', c='black')  # average value (depth tov maaiveld: 3.01-4.01m) B40E1315
a += (phi[:, 770]-9.16)**2
b += 1
plt.plot(4000, 9.12, 'o', c='black')  # average value (depth tov maaiveld: 3.1-4.1m) B40E1492
a += (phi[:, 770]-9.12)**2
b += 1
plt.plot(4000, 9.27, 'o', c='black')  # average value (depth tov maaiveld: 7.44-10.44m) B40E0254
a += (phi[:, 770]-9.27)**2
b += 1
plt.plot(4000, 9.30, 'o', c='black')  # average value (depth tov maaiveld: 4.65-5.65m) B40E1311
a += (phi[:, 770]-9.03)**2
b += 1

print('RMSE:', np.sqrt(a/b))

plt.axvline(-3700, c='dimgrey', linestyle=':')
plt.axvline(-3100, c='dimgrey', linestyle=':')
plt.axvline(0, c='dimgrey', linestyle=':', label='Fixed section separation')
plt.axvline(x_left, c='darkgrey', linestyle=':', label='Flexible section separation')
plt.axvline(x_right, c='darkgrey', linestyle=':')
plt.axvline(x_right, c='darkgrey', linestyle=':')
plt.axvline(2000, c='dimgrey', linestyle=':')
plt.axvline(3000, c='dimgrey', linestyle=':')
plt.axvline(5500, c='dimgrey', linestyle=':')

v = 7

marge = v * 0.005
plt.plot([-3700, -3100], [v - marge, v - marge], c='grey')
plt.plot([0, 2000], [v - marge, v - marge], c='grey')
plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)
plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)

plt.plot(X, phi[0, :], label='Range of modelled heads in first aquifer', c='darksalmon', alpha=0.5, linewidth=2)
plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation', c='darkred', linestyle='--', linewidth=2)

plt.xlabel('Distance along cross-section [m]')
plt.ylabel('Head [m]')
leg = plt.legend(fontsize=14, loc='best')
leg.get_lines()[0].set_linewidth(2)
leg.get_lines()[1].set_linewidth(2)
leg.get_lines()[2].set_linewidth(4)
leg.get_lines()[3].set_linewidth(4)
leg.get_lines()[4].set_linewidth(4)
leg.get_lines()[5].set_linewidth(4)
plt.show()

plt.show()