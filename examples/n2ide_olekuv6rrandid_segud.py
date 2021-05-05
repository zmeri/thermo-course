'''
Näide olekuvõrrandite kasutamisest Pythonis
'''
import numpy as np
import pandas as pd
import CoolProp.CoolProp as cp # dokumentatsioon: http://www.coolprop.org/index.html
from pcsaft import pcsaft_p, pcsaft_den, flashTQ # dokumentatsioon: https://pcsaft.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
from scipy.optimize import minimize

R = 8.314462618 # m^3 Pa K^-1 mol^-1

def mixing_vdW1f(a, b, x, c):
    '''
    Calculate mixture parameters for cubic equations of state using the van
    der Waals one fluid mixing rule and classical combination rule

    Parameters
    ----------
    a : ndarray, shape (n,)
        Parameter *a* for each of the components
    b : ndarray, shape (n,)
        Parameter *b* for each of the components
    x : ndarray, shape (n,)
        Mole fractions of each component
    c : dict or list of dicts
        Parameters for the cubic equation of state

    Returns
    -------
    a_mix : float
        Parameter *a* for the mixture
    b_mix : float
        Parameter *b* for the mixture
    '''
    a2d = a.reshape(-1,1)
    if 'kij' in c:
        aij = np.sqrt(np.dot(a2d, a2d.transpose())) * (1 - c['kij'])
    else:
        aij = np.sqrt(np.dot(a2d, a2d.transpose()))

    b_tile = np.tile(b, (b.shape[0], 1))
    if 'lij' in c:
        bij = (b_tile + b_tile.transpose()) / 2 * (1 - c['lij'])
    else:
        bij = (b_tile + b_tile.transpose()) / 2

    x2d = x.reshape(-1,1)
    a_mix = np.sum(x2d * x2d.transpose() * aij)
    b_mix = np.sum(x2d * x2d.transpose() * bij)
    return a_mix, b_mix


def soave_redlich_kwong(t, v, x, c):
    '''
    Calculate pressure using the SRK equation of state

    Parameters
    ----------
    t : float
        Temperature (K)
    v : float
        Molar volume (m^3 mol^-1)
    x : ndarray, shape (n,)
        Mole fractions of each component
    c : dict
        Parameters for the SRK equation of state

    Returns
    -------
    p : float
        Pressure (Pa)
    '''
    alpha = (1 + (0.48 + 1.574 * c['acentric'] - 0.176 * c['acentric']**2) \
            * (1 - (t / c['t_crit'])**0.5))**2
    ai = alpha * (0.42748 * R**2 * c['t_crit']**2 / c['p_crit'])
    bi = 0.08664 * R * c['t_crit'] / c['p_crit']

    if x.shape[0] > 1:
        a, b = mixing_vdW1f(ai, bi, x, c)
    else:
        a = ai
        b = bi

    p = R * t / (v - b) - a / (v * (v + b))
    return p


def peng_robinson(t, v, x, c):
    '''
    Calculate pressure using the Peng-Robinson equation of state

    Parameters
    ----------
    t : float
        Temperature (K)
    v : float
        Molar volume (m^3 mol^-1)
    x : ndarray, shape (n,)
        Mole fractions of each component
    c : dict or list of dicts
        Parameters for the Peng-Robinson equation of state

    Returns
    -------
    p : float
        Pressure (Pa)
    '''
    alpha = (1 + (0.37464 + 1.54226 * c['acentric'] - 0.26992 * c['acentric']**2) \
            * (1 - (t / c['t_crit'])**0.5))**2
    ai = alpha * (0.45724 * R**2 * c['t_crit']**2 / c['p_crit'])
    bi = 0.07780 * R * c['t_crit'] / c['p_crit']

    if x.shape[0] > 1:
        a, b = mixing_vdW1f(ai, bi, x, c)
    else:
        a = ai
        b = bi

    p = R * t / (v - b) - a / (v**2 + 2*b*v - b**2)
    return p


def srk_cost(den, p, t, x, c):
    ''' Cost function used to solve for the molar density '''
    v = 1/den
    p_calc = soave_redlich_kwong(t, v, x, c)
    cost = ((p_calc - p) / p * 100)**2
    return cost


def pr_cost(den, p, t, x, c):
    ''' Cost function used to solve for the molar density '''
    v = 1/den
    p_calc = peng_robinson(t, v, x, c)
    cost = ((p_calc - p) / p * 100)**2
    return cost


def cubic_den_solver(eos, den_guess, p, t, x, c):
    '''
    Solves for the molar density using a cubic equation of state

    Parameters
    ----------
    eos : string
        Equation of state to use. Options are soave_redlich_kwong and peng_robinson.
    den_guess: float
        Initial guess for the molar density (mol m^-3)
    p : float
        Pressure (Pa)
    t : float
        Temperature (K)
    c : dict
        Fluid constants for the cubic equation of state

    Returns
    -------
    den : float
        Molar density (mol m^-3)
    '''
    if eos == 'soave_redlich_kwong':
        result = minimize(srk_cost, den_guess, args=(p, t, x, c))
    elif eos == 'peng_robinson':
        result = minimize(pr_cost, den_guess, args=(p, t, x, c))
    else:
        raise ValueError('{} is not one of the implemented cubic equations of state'.format(eos))

    den = result.x
    return den


# heksaan + benseen ------------------------------
t = np.linspace(360, 473, 30) # K
p = 100000 # Pa
x = np.asarray([0.3, 0.7]) # moolosad

den_pcsaft = np.zeros_like(t)
m = np.asarray([3.0576, 2.4653])
sigma = np.asarray([3.7983, 3.6478])
eps_k = np.asarray([236.77, 287.35])
kij_pcsaft = np.asarray([[0, 0.0128],
                        [0.0128, 0]])
param_pcsaft = {'m':m, 's':sigma, 'e':eps_k, 'k_ij': kij_pcsaft}
for i in range(t.shape[0]):
    den_pcsaft[i] = pcsaft_den(t[i], p, x, param_pcsaft, phase='vap')

t_crit = np.asarray([507.82, 562.02])
p_crit = np.asarray([3044115.328359688, 4894000.0])
acentric = np.asarray([0.3003189315498438, 0.2108369732700151])
kij_cubic = np.asarray([[0, 0.0093],
                        [0.0093, 0]])
param_cubic = {'t_crit':t_crit, 'p_crit':p_crit, 'acentric':acentric, 'kij':kij_cubic}
den_pr = np.zeros_like(t)
for i in range(t.shape[0]):
    den_pr[i] = cubic_den_solver('peng_robinson', den_pcsaft[i], p, t[i], x, param_cubic)

# tee joonist
plt.figure()
plt.plot(t, den_pr, label='PR')
plt.plot(t, den_pcsaft, label='PC-SAFT')
title_text = 'Heksaan + benseen rõhul {} bar'.format(p/100000)
plt.title(title_text)
plt.xlabel('Temperatuur (K)')
plt.ylabel('Molaarne tihedus (mol m$^{-3}$)')
plt.legend(frameon=False)
plt.savefig('olekuvõrrandid_heksaan_benseen', dpi=400)
plt.show()
