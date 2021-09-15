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

import matplotlib.pylab as pylab
params = {'legend.fontsize': 16,
         'axes.labelsize':   15,
         'axes.labelcolor':  'black',
         'axes.titlesize':   16,
         'xtick.labelsize':  13.5,
         'ytick.labelsize':  14}
pylab.rcParams.update(params)


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

    ## Multi layer system:
    # qsum = q[0, :] + q[1, :] + q[2, :]

    ## One layer system:
    qsum = q[0, :]

    qleft_Rijnstr = qsum[371]                                   # Q left of Rijnstrangen
    qright_Rijnstr = qsum[570]                                  # Q right of Rijnstrangen
    qtot = (qright_Rijnstr - qleft_Rijnstr) * 12000 / 1000000   # gives q_total out of Rijnstrangen in Mm3/d

    return qtot

fig, axs = plt.subplots(3, 7, sharey='all', figsize=(18,9))
b = []

for i in range(1000):
    # c, (nLay by nSec) matrix of vertical resistance values of top layer and aquitards
    c1 = random.randint(1,6500)*0.01
    c2 = random.randint(550,700)
    c3left = random.randint(1000, 10000)
    c3 = random.randint(25,150)
    # c3 = 80
    c3right = random.randint(1000, 10000)
    c4 = random.randint(200,325)
    c5 = random.randint(150,275)

    c = np.array([[c1, c2, c3left, c3, c3right, c4, c5]])

    # T, (nLay by nSec) matrix of transmissivity values
    kD1 = random.randint(50,600)
    kD2 = random.randint(100,1200)
    # kD2 = 750
    kD3left = random.randint(100, 1200)
    # kD3left = 750
    kD3 = random.randint(100,1200)
    # kD3 = 750
    kD3right = random.randint(100, 1200)
    # kD3right = 750
    kD4 = random.randint(100,1200)
    # kD4 = 750
    kD5 = random.randint(100,1200)

    kD = np.array([[kD1, kD2, kD3left, kD3, kD3right, kD4, kD5]])

    # h, (1 by nSec) vector of heads on top of each section (surface water level in m+NAP)
    h1 = random.randint(600, 1780)*0.01
    # h1 = 9.2
    h2 = random.randint(800, 1600)*0.01
    # h2 = 9.8
    h3left = random.randint(800, 1600)*0.01
    # h3left = 9.8
    h3 = random.randint(925, 1575)*0.01
    # h3 = 12.5
    h3right = random.randint(850, 1150)*0.01
    # h3right = 9.7
    h4 = random.randint(850, 1150)*0.01
    # h4 = 9.7
    h5 = random.randint(750, 1050)*0.01
    # h5 = 9.1

    # heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])
    heads = np.array([h1, h2, h3left, h3, h3right, h4, h5])

    Q_GWleft = 0 #-50000000 / 365 / 12000
    Q_GWright = 0 #-100000000 / 365 / 12000

    a = groundwater(h3, Q_GWleft, Q_GWright)
    b.append(a)

    axs[0, 0].plot(h1, a, 'b.')
    axs[0, 1].plot(h2, a, 'b.')
    axs[0, 2].plot(h3left, a, 'b.')
    axs[0, 3].plot(h3, a, 'b.')
    axs[0, 4].plot(h3right, a, 'b.')
    axs[0, 5].plot(h4, a, 'b.')
    axs[0, 6].plot(h5, a, 'b.')

    axs[1, 0].plot(c1, a, 'b.')
    axs[1, 1].plot(c2, a, 'b.')
    axs[1, 2].plot(c3left, a, 'b.')
    axs[1, 3].plot(c3, a, 'b.')
    axs[1, 4].plot(c3right, a, 'b.')
    axs[1, 5].plot(c4, a, 'b.')
    axs[1, 6].plot(c5, a, 'b.')

    axs[2, 0].plot(kD1, a, 'b.')
    axs[2, 1].plot(kD2, a, 'b.')
    axs[2, 2].plot(kD3left, a, 'b.')
    axs[2, 3].plot(kD3, a, 'b.')
    axs[2, 4].plot(kD3right, a, 'b.')
    axs[2, 5].plot(kD4, a, 'b.')
    axs[2, 6].plot(kD5, a, 'b.')

    # axs[0, 0].plot(np.sqrt(c1*kD1), a, 'm.')
    # axs[0, 1].plot(np.sqrt(c2*kD2), a, 'm.')
    # axs[0, 2].plot(np.sqrt(c3left*kD3left), a, 'm.')
    # axs[0, 3].plot(np.sqrt(c3*kD3), a, 'm.')
    # axs[0, 5].plot(np.sqrt(c3right*kD3right), a, 'm.')
    # axs[0, 4].plot(np.sqrt(c4*kD4), a, 'm.')
    # axs[0, 6].plot(np.sqrt(c5*kD5), a, 'm.')

    print(i)

# print(b)
max = np.max(b)
print('max:', max)
min = np.min(b)
print('min:', min)

axs[0, 0].set_title('Section 1', fontweight="bold")
axs[0, 0].set(ylabel='Q$_{out}$ [Mm$^3$/day]')
axs[0, 0].set(xlabel='h [m]')
axs[0, 0].plot((6.9, 6.9), (min, max), c='olivedrab')
axs[0, 0].plot((16.9, 16.9), (min, max), c='olivedrab')
axs[0, 1].set_title('Section 2', fontweight="bold")
axs[0, 1].set(xlabel='h [m]')
axs[0, 1].plot((9, 9), (min, max), c='olivedrab')
axs[0, 1].plot((15, 15), (min, max), c='olivedrab')
axs[0, 2].set_title('Section 3a', fontweight="bold")
axs[0, 2].set(xlabel='h [m]')
axs[0, 2].plot((9, 9), (min, max), c='olivedrab')
axs[0, 2].plot((15, 15), (min, max), c='olivedrab')
axs[0, 3].set_title('Section 3', fontweight="bold")
axs[0, 3].set(xlabel='h [m]')
axs[0, 3].plot((10, 10), (min, max), c='olivedrab')
axs[0, 3].plot((15, 15), (min, max), c='olivedrab')
axs[0, 4].set_title('Section 3b', fontweight="bold")
axs[0, 4].set(xlabel='h [m]')
axs[0, 4].plot((9, 9), (min, max), c='olivedrab')
axs[0, 4].plot((11, 11), (min, max), c='olivedrab')
axs[0, 5].set_title('Section 4', fontweight="bold")
axs[0, 5].set(xlabel='h [m]')
axs[0, 5].plot((9, 9), (min, max), c='olivedrab')
axs[0, 5].plot((11, 11), (min, max), c='olivedrab')
axs[0, 6].set_title('Section 5', fontweight="bold")
axs[0, 6].set(xlabel='h [m]')
axs[0, 6].plot((8, 8), (min, max), c='olivedrab')
axs[0, 6].plot((10, 10), (min, max), c='olivedrab')

# axs[0, 0].set_title('Section 1', fontsize=14)
axs[1, 0].set(ylabel='Q$_{out}$ [Mm$^3$/day]')
axs[1, 0].set(xlabel='c$_{top}$ [days]')
axs[1, 0].plot((0.01, 0.01), (min, max), c='olivedrab')
axs[1, 0].plot((50, 50), (min, max), c='olivedrab')
# axs[0, 1].set_title('Section 2', fontsize=14)
axs[1, 1].set(xlabel='c$_{top}$ [days]')
axs[1, 1].plot((575, 575), (min, max), c='olivedrab')
axs[1, 1].plot((675, 675), (min, max), c='olivedrab')
# axs[0, 2].set_title('Section 3', fontsize=14)
axs[1, 2].set(xlabel='c$_{top}$ [days]')
axs[1, 2].plot((1000, 1000), (min, max), c='olivedrab', linestyle=':')
axs[1, 2].plot((10000, 10000), (min, max), c='olivedrab', linestyle=':')
# axs[0, 3].set_title('Section 4 left', fontsize=14)
axs[1, 3].set(xlabel='c$_{top}$ [days]')
axs[1, 3].plot((50, 50), (min, max), c='olivedrab')
axs[1, 3].plot((125, 125), (min, max), c='olivedrab')
# axs[0, 4].set_title('Section 4', fontsize=14)
axs[1, 4].set(xlabel='c$_{top}$ [days]')
axs[1, 4].plot((1000, 1000), (min, max), c='olivedrab', linestyle=':')
axs[1, 4].plot((10000, 10000), (min, max), c='olivedrab', linestyle=':')
# axs[0, 5].set_title('Section 4 right', fontsize=14)
axs[1, 5].set(xlabel='c$_{top}$ [days]')
axs[1, 5].plot((225, 225), (min, max), c='olivedrab')
axs[1, 5].plot((300, 300), (min, max), c='olivedrab')
# axs[0, 6].set_title('Section 5', fontsize=14)
axs[1, 6].set(xlabel='c$_{top}$ [days]')
axs[1, 6].plot((175, 175), (min, max), c='olivedrab')
axs[1, 6].plot((250, 250), (min, max), c='olivedrab')

axs[2, 0].set(ylabel='Q$_{out}$ [Mm$^3$/day]')
axs[2, 0].set(xlabel='kD [m$^2$/d]')
axs[2, 0].plot((100, 100), (min, max), c='olivedrab')
axs[2, 0].plot((500, 500), (min, max), c='olivedrab')
axs[2, 1].set(xlabel='kD [m$^2$/d]')
axs[2, 1].plot((250, 250), (min, max), c='olivedrab')
axs[2, 1].plot((1000, 1000), (min, max), c='olivedrab')
axs[2, 2].set(xlabel='kD [m$^2$/d]')
axs[2, 2].plot((250, 250), (min, max), c='olivedrab')
axs[2, 2].plot((1000, 1000), (min, max), c='olivedrab')
axs[2, 3].set(xlabel='kD [m$^2$/d]')
axs[2, 3].plot((250, 250), (min, max), c='olivedrab')
axs[2, 3].plot((1000, 1000), (min, max), c='olivedrab')
axs[2, 4].set(xlabel='kD [m$^2$/d]')
axs[2, 4].plot((250, 250), (min, max), c='olivedrab')
axs[2, 4].plot((1000, 1000), (min, max), c='olivedrab')
axs[2, 5].set(xlabel='kD [m$^2$/d]')
axs[2, 5].plot((250, 250), (min, max), c='olivedrab')
axs[2, 5].plot((1000, 1000), (min, max), c='olivedrab')
axs[2, 6].set(xlabel='kD [m$^2$/d]')
axs[2, 6].plot((250, 250), (min, max), c='olivedrab')
axs[2, 6].plot((1000, 1000), (min, max), c='olivedrab', label='Likely parameter range')

# axs[0, 0].set(ylabel='Q$_{out}$ [Mm$^3$/day]')
# axs[0, 0].set(xlabel='Leakage factor')
# axs[0, 1].set(xlabel='Leakage factor')
# axs[0, 2].set(xlabel='Leakage factor')
# axs[0, 3].set(xlabel='Leakage factor')
# axs[0, 4].set(xlabel='Leakage factor')
# axs[0, 5].set(xlabel='Leakage factor')
# axs[0, 6].set(xlabel='Leakage factor')

leg = axs[2, 6].legend(bbox_to_anchor=(1.0, -0.3), frameon=False)
leg.get_lines()[0].set_linewidth(4)

fig.subplots_adjust(hspace=0.4)

plt.show()