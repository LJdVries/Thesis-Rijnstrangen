import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.linalg import expm, sqrtm
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')


## PARAMETERS:
A_tot = 28646820


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


## NORMAL GW CALCULATION (1 LAYER, 7 SECTIONS)
def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_phi_ranges=False, plot_q=True, plot_s=False):
    A_wet = lookup_h(h_Rijnstrangen)[1]
    fraction = A_wet/A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left
    c_var = 10000

    c = np.array([[1, 650, c_var, 80, c_var, 250, 225]])

    kD = np.array([[250, 750, 750, 750, 750, 750, 750]])

    heads = np.array([9.2, 9.8, 9.8, h_Rijnstrangen, 9.7, 9.7, 9.1])

    Q = np.array([[0, Q_GWleft, 0, 0, Q_GWright, 0]])

    x = np.array([-3100, 0, x_left, x_right, 2000, 3000])

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
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  # coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  # right hand side vector

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

        marge = v*0.002
        plt.plot([-3700, -3100], [v - marge, v - marge], c='grey')
        plt.plot([0, 2000], [v - marge, v - marge], c='grey')
        plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)
        plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)

        plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation (1 layer, 7 sections)', c='darkred', linestyle='--', linewidth=2.5)

    if plot_q is True:

        plt.figure(figsize=(15, 10))
        plt.axhline(0, linewidth=2, c='white')
        plt.axvline(0, c='dimgrey', linestyle=':', label='Fixed section separation')
        plt.axvline(x_left, c='darkgrey', linestyle=':', label='Flexible section separation')
        plt.axvline(x_right, c='darkgrey', linestyle=':')
        plt.axvline(2000, c='dimgrey', linestyle=':')
        plt.axvline(3000, c='dimgrey', linestyle=':')
        plt.axvline(5500, c='dimgrey', linestyle=':')
        plt.axvline(-3700, c='dimgrey', linestyle=':')
        plt.axvline(-3100, c='dimgrey', linestyle=':')

        plt.plot(X, q[0, :], label='Total groundwater flux (1 layer, 7 sections)', c='darkred', linestyle='--', linewidth=2.5)
groundwater(h_Rijnstrangen=13, Q_GWleft=0, Q_GWright=0, plot_phi=True, plot_phi_ranges=False, plot_q=False, plot_s=False)


## GW CALCULATION FOR 1 LAYER, 9 SECTIONS
def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_phi_ranges=False, plot_q=True, plot_s=False):
    A_wet = lookup_h(h_Rijnstrangen)[1]
    fraction = A_wet/A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left
    c_var = 10000

    c = np.array([[675, 1, 650, c_var, 80, c_var, 250, 225, 225]])

    kD = np.array([[500, 250, 750, 750, 750, 750, 750, 750, 750]])

    heads = np.array([9.5, 9.2, 9.8, 9.8, h_Rijnstrangen, 9.7, 9.7, 9.1, 8.8])

    Q = np.array([[0, 0, Q_GWleft, 0, 0, Q_GWright, 0, 0]])

    x = np.array([-3700, -3100, 0, x_left, x_right, 2000, 3000, 5500])

    X = np.arange(-8000, 8010, 10)

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
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  # coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  # right hand side vector

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

    if plot_phi is True:
        plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation (1 layer, 9 sections)', c='darkgreen', linestyle='--', linewidth=2.5)

    if plot_q is True:
        plt.plot(X, q[0, :], label='Total groundwater flux (1 layer, 9 sections)', c='darkgreen', linestyle='--', linewidth=2.5)
groundwater(h_Rijnstrangen=13, Q_GWleft=0, Q_GWright=0, plot_phi=True, plot_phi_ranges=False, plot_q=False, plot_s=False)


## GW CALCULATION FOR 3 LAYERS, 7 SECTIONS
def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_phi_ranges=False, plot_q=True, plot_s=False):
    A_wet = lookup_h(h_Rijnstrangen)[1]
    fraction = A_wet/A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left
    c_var = 10000

    c = np.array([[1, 650, c_var, 80, c_var, 250, 225],
                  [1.0e4, 3.2e4, 3.2e4, 3.2e4, 3.2e4, 1.0e4, 5.0e3],
                  [1.0e2, 5.0e2, 1.0e3, 5.0e2, 5.0e2, 5.0e2, 5.0e2]])

    kD = np.array([[250, 750, 750, 750, 750, 750, 750],
                   [500, 100, 50, 50, 50, 50, 250],
                   [400, 400, 500, 500, 500, 500, 500]])

    heads = np.array([9.2, 9.8, 9.8, h_Rijnstrangen, 9.7, 9.7, 9.1])

    Q = np.array([[0,  Q_GWleft,  0, 0,  Q_GWright,  0],
                  [0,  0,  0,  0,  0,  0],
                  [0,  0,  0,  0,  0,  0]])

    x = np.array([-3100, 0, x_left, x_right, 2000, 3000])

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
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  # coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  # right hand side vector

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

    qsum = q[0, :] + q[1, :] + q[2, :]

    if plot_phi is True:
        plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation (3 layers, 7 sections)', c='royalblue', linestyle='--', linewidth=2.5)
        # plt.plot(X, phi[1, :], label='Modelled head in second aquifer \n for average situation (3 layers, 7 sections)', c='royalblue', linestyle='-.', linewidth=1.5)
        # plt.plot(X, phi[2, :], label='Modelled head in third aquifer \n for average situation (3 layers, 7 sections)', c='royalblue', linestyle=':', linewidth=1.5)

    if plot_q is True:
        plt.plot(X, qsum, label='Total groundwater flux (3 layers, 7 sections)', c='royalblue', linestyle='--', linewidth=2.5)
        plt.plot(X, q[0, :], label='q in first aquifer (3 layers, 7 sections)', c='royalblue', linestyle='--', linewidth=1.5)
        plt.plot(X, q[1, :], label='q in second aquifer (3 layers, 7 sections)', c='royalblue', linestyle='-.', linewidth=1.5)
        plt.plot(X, q[2, :], label='q in third aquifer (3 layers, 7 sections)', c='royalblue', linestyle=':', linewidth=1.5)

        v = qsum[:].min() + 0.08*qsum[:].min()
        marge = v*0.02
        plt.plot([0, 2000], [v + marge, v + marge], c='grey')
        plt.plot([-3700, -3100], [v + marge, v + marge], c='grey')
        plt.text(1000, v, 'Rijnstrangen', color='grey', horizontalalignment='center', fontsize=14)
        plt.text(-3400, v, 'Rhine', color='grey', horizontalalignment='center', fontsize=14)
groundwater(h_Rijnstrangen=13, Q_GWleft=0, Q_GWright=0, plot_phi=True, plot_phi_ranges=False, plot_q=False, plot_s=False)


## GW CALCULATION FOR 3 LAYERS, 9 SECTIONS
def groundwater(h_Rijnstrangen, Q_GWleft, Q_GWright, plot_phi=False, plot_phi_ranges=False, plot_q=True, plot_s=False):
    A_wet = lookup_h(h_Rijnstrangen)[1]
    fraction = A_wet/A_tot
    x_left = (2000 - (2000 * fraction)) / 2
    x_right = 2000 - x_left
    c_var = 10000

    c = np.array([[675, 1, 650, c_var, 80, c_var, 250, 225, 225],
                  [3.2e4, 1.0e4, 3.2e4, 3.2e4, 3.2e4, 3.2e4, 1.0e4, 5.0e3, 5.0e3],
                  [1, 1.0e2, 5.0e2, 1.0e3, 5.0e2, 5.0e2, 5.0e2, 5.0e2, 5.0e2]])

    kD = np.array([[500, 250, 750, 750, 750, 750, 750, 750, 750],
                   [250, 500, 100, 50, 50, 50, 50, 250, 250],
                   [400, 400, 400, 500, 500, 500, 500, 500, 500]])

    heads = np.array([9.5, 9.2, 9.8, 9.8, h_Rijnstrangen, 9.7, 9.7, 9.1, 8.8])

    Q = np.array([[0, 0,  Q_GWleft,  0, 0,  Q_GWright,  0, 0],
                  [0, 0,  0,  0,  0,  0,  0, 0],
                  [0, 0,  0,  0,  0,  0,  0, 0]])

    x = np.array([-3700, -3100, 0, x_left, x_right, 2000, 3000, 5500])

    X = np.arange(-8000, 8010, 10)

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
    C = np.zeros((nLay * (2 * (Nx - 2)), nLay * (2 * (Nx - 2) + 2)))  # coefficient matrix
    R = np.zeros((nLay * (2 * (Nx - 2)), 1))  # right hand side vector

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

    qsum = q[0, :] + q[1, :] + q[2, :]

    if plot_phi is True:
        plt.plot(X, phi[0, :], label='Modelled head in first aquifer \n for average situation (3 layers, 9 sections)', c='darkorange', linestyle='--', linewidth=2.5)
        # plt.plot(X, phi[1, :], label='Modelled head in second aquifer \n for average situation (3 layers, 7 sections)', c='darkorange', linestyle='-.', linewidth=1.5)
        # plt.plot(X, phi[2, :], label='Modelled head in third aquifer \n for average situation (3 layers, 7 sections)', c='darkorange', linestyle=':', linewidth=1.5)

        plt.xlabel('Distance along cross-section [m]', size=14)
        plt.ylabel('Head [m +NAP]', size=14)
        leg = plt.legend(fontsize=14, loc='best')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)
        leg.get_lines()[2].set_linewidth(4)
        leg.get_lines()[3].set_linewidth(4)
        leg.get_lines()[4].set_linewidth(4)
        leg.get_lines()[5].set_linewidth(4)
        plt.show()

    if plot_q is True:
        plt.plot(X, qsum, label='Total groundwater flux (3 layers, 9 sections)', c='darkorange', linestyle='--', linewidth=2.5)
        plt.plot(X, q[0, :], label='q in first aquifer (3 layers, 9 sections)', c='darkorange', linestyle='--', linewidth=1.5)
        plt.plot(X, q[1, :], label='q in second aquifer (3 layers, 9 sections)', c='darkorange', linestyle='-.', linewidth=1.5)
        plt.plot(X, q[2, :], label='q in third aquifer (3 layers, 9 sections)', c='darkorange', linestyle=':', linewidth=1.5)

        plt.xlabel('Distance along cross-section [m]', size=14)
        plt.ylabel('Flux [m$^2$/d]', size=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        leg = plt.legend(fontsize=14, loc='best')
        leg.get_lines()[0].set_linewidth(2)
        leg.get_lines()[1].set_linewidth(2)
        leg.get_lines()[2].set_linewidth(4)
        leg.get_lines()[3].set_linewidth(4)
        leg.get_lines()[4].set_linewidth(4)
        leg.get_lines()[5].set_linewidth(2.5)
        leg.get_lines()[6].set_linewidth(2.5)
        leg.get_lines()[7].set_linewidth(2.5)
        leg.get_lines()[8].set_linewidth(4)
        leg.get_lines()[9].set_linewidth(2.5)
        leg.get_lines()[10].set_linewidth(2.5)
        leg.get_lines()[11].set_linewidth(2.5)

        plt.show()
groundwater(h_Rijnstrangen=13, Q_GWleft=0, Q_GWright=0, plot_phi=True, plot_phi_ranges=False, plot_q=False, plot_s=False)
