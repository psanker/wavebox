# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import matplotlib.pyplot as plt

import decimal as dec
import gc
import math
import os
import shutil
import sys

from astropy import units as u
from astropy import constants as const

from matplotlib import animation
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D

from numba import jit

from scipy.misc import factorial
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.fftpack import fftshift
from scipy import linalg as lin
from scipy import optimize as opt
from scipy import stats

import sympy as sp

sp.init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. Constants
#############################################################

PI     = np.pi
TWO_PI = 2.*PI

kB   = const.k_B
R    = const.R
NA   = const.N_A
G    = const.G
g    = const.g0
mu    = 1.317e25 * G
R_E  = 6371 * 1000
h    = const.h
hbar = const.hbar
c    = const.c
m_e  = const.m_e
m_n  = const.m_n
m_p  = const.m_p
R_H  = 2.18e-18 * u.J

#############################################################
# 3. General Functions
#############################################################

def progress_meter(per, LEN=50):
    val = per * LEN

    bar = '['

    for i in range(LEN):
        if i <= val:
            bar += '#'
        else:
            bar += '_'

    return (bar + '] %2.0f%%' % (per * 100))

#############################################################
# 4. Data & Globals
#############################################################

data_N = 40
data_m = m_e.value
data_L = 0.1e-3

x_samples = 10000
x_spacing = (2. * data_L) / x_samples

data_X = np.linspace(0, x_spacing * x_samples, x_samples)
data_T = np.linspace(0, 200e-6, 500)

data_K = fftshift(fftfreq(x_samples, x_spacing))

CACHE  = {
    'psiT': {},
    'psikT': {}
}

#############################################################
# 5. Lab-Specific Functions
#############################################################

def terminate():
    # Any memory clearing logic and stuff

    global CACHE
    purge_tree(CACHE)
    CACHE = None

def purge_tree(d):
    for k, v in d.items():
        if isinstance(d[k], dict):
            purge_tree(d[k])
        else:
            d[k] = None

@jit
def coeff(N, L):
    if 'coeff' in CACHE:
        return CACHE['coeff']

    ret   = np.empty(N)
    sqrt2 = np.sqrt(2) / PI

    for n in np.arange(1, N + 1):
        if n == 2:
            foo = PI / 2.
        else:
            foo = ((n - 2.)**(-1.)) * np.sin(((n - 2.) * PI) / 2.)

        bar = ((n + 2.)**(-1.)) * np.sin(((n + 2.) * PI) / 2.)

        ret[n - 1] = sqrt2 * (foo - bar)

    CACHE['coeff'] = ret
    return ret

@jit
def energies(t, N, L, m):
    ret = np.empty(N, dtype=complex)
    hb  = hbar.value

    c   = coeff(N, L)
    val = (2. * m)**(-1.) * (hb * PI / L)**(2.)

    for n in np.arange(1, N + 1):
        E = val * (n**2.)

        re = c[n - 1] * np.cos(-1. * (E / hb) * t)
        im = c[n - 1] * np.sin(-1. * (E / hb) * t)
        ret[n - 1] = re + im*1j

    return ret

@jit
def basis(x, N, L):
    key = 'xbasis'

    if key in CACHE:
        return CACHE[key]

    ret = np.empty((len(x), N))

    # Nested for loop; sue me
    for i in range(len(x)):
        for n in np.arange(1, N + 1):
            ret[i, n - 1] = np.sqrt(1. / L) * np.sin(((PI * n) / (2. * L)) * x[i])

    CACHE[key] = ret
    return ret

@jit
def k_basis(x, N, L):
    key = 'kbasis'

    if key in CACHE:
        return CACHE[key]

    # transpose so each row corresponds to the x-values for one energy eigenstate
    xbasis  = basis(x, N, L).T
    kbasisT = np.empty((N, len(x)), dtype=complex)

    for i in range(len(xbasis)):
        kbasisT[i] = fftshift(fft(xbasis[i]))

    ret        = kbasisT.T
    CACHE[key] = ret

    return ret

@jit
def psi_T(t, x, N, L, m):
    key = t

    if key in CACHE['psiT']:
        return CACHE['psiT'][key]

    ret = np.empty(len(x), dtype=complex)

    E   = energies(t, N, L, m)
    b   = basis(x, N, L)

    for i in range(len(x)):
        ret[i] = np.dot(b[i], E)

    CACHE['psiT'][key] = ret
    return ret

@jit
def kpsi_T(t, x, N, L, m):
    key = t

    if key in CACHE['psikT']:
        return CACHE['psikT'][key]

    ret = np.empty(len(x), dtype=complex)

    E   = energies(t, N, L, m)
    b   = k_basis(x, N, L)

    for i in range(len(x)):
        ret[i] = np.dot(b[i], E)

    CACHE['psikT'][key] = ret
    return ret

@jit
def probability(wav, delta=x_spacing):
    return np.real(np.multiply(np.conjugate(wav), wav) * delta)

# Interactivity below here ---------

def get_basis():
    return basis(data_X, data_N, data_L)

def get_kbasis():
    return k_basis(data_X, data_N, data_L)

# Get the coefficients of the wavefunction
def get_coeff():
    return coeff(data_N, data_L)

# Check the probability normalization condition
def get_probcheck():
    an   = coeff(data_N, data_L)
    spec = np.empty(len(an))

    for i in range(len(an)):
        spec[i] = an[i]**(2.)

    return np.sum(spec)

# Show which eigenstates show up
def plot_probspec():
    an   = coeff(data_N, data_L)
    spec = np.empty(len(an))

    for i in range(len(an)):
        spec[i] = an[i]**(2.)

    plt.figure()

    n = np.arange(1, data_N + 1)
    plt.plot(n, spec, 'ro')

# Plot the expected postion over time
def plot_expvalue():
    expect = np.empty(len(data_T))

    # Draw the frames
    print('Building...')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        prob = probability(psi_T(data_T[i], data_X, data_N, data_L, data_m))

        expect[i] = np.sum(np.multiply(data_X, prob))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(data_T, expect, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Position ($m$)')
    plt.grid()

# Plot the uncertainty in postion over time
def plot_deviation():
    dev = np.empty(len(data_T))

    # Now draw the frames
    print('Building...')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        prob = probability(psi_T(data_T[i], data_X, data_N, data_L, data_m))

        exp  = np.sum(np.multiply(data_X, prob))
        exp2 = np.sum(np.multiply(data_X**2., prob))

        dev[i] = np.sqrt(exp2 - exp**(2.))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(data_T, dev, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Uncertainty ($m$)')

def plot_expp():
    expect = np.empty(len(data_T))

    print('Building...')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        prob = probability(kpsi_T(data_T[i], data_X, data_N, data_L, data_m), delta=(x_spacing / x_samples))

        expect[i] = hbar.value * np.sum(np.multiply(data_K, prob))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(data_T, expect, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Momentum ($kg\\cdot\\frac{m}{s}$)')
    plt.grid()

def plot_devp():
    dev = np.empty(len(data_T))

    print('Building...')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        prob  = probability(kpsi_T(data_T[i], data_X, data_N, data_L, data_m), delta=(x_spacing / x_samples))

        exp   = hbar.value * np.sum(np.multiply(data_K, prob))
        exp2  = hbar.value * np.sum(np.multiply(data_K**2., prob))

        dev[i] = np.sqrt(exp2 - exp**(2.))

    print('') #dummy spacer
    fig, ax = plt.subplots()

    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

    ax.plot(data_T, dev, 'r-')

    plt.xlabel('Time ($s$)')
    plt.ylabel('Uncertainty ($kg\\cdot\\frac{m}{s}$)')

# Builds the cache of coefficients and wavefunctions
def run_buildcache():
    print('Coefficients..')
    foo = coeff(data_N, data_L)
    foo = None

    print('Position basis..')
    foo = basis(data_X, data_N, data_L)
    foo = None

    print('Wavevector basis..')
    foo = k_basis(data_X, data_N, data_L)
    foo = None

    print('Position Wavefunctions..')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        foo = psi_T(data_T[i], data_X, data_N, data_L, data_m)
        foo = None

    print('\nk-space Wavefunctions..')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        foo = kpsi_T(data_T[i], data_X, data_N, data_L, data_m)
        foo = None

    print('\nCache built.')

# Renders the probability evolving over time
def run_animate():
    # Create cache directory
    directory = 'wavebox'
    cache     = directory + '/dump'

    if not os.path.exists(cache):
        os.makedirs(cache)

    X = data_X
    T = data_T

    # Firstly, build cache if it hasn't already been built
    run_buildcache()

    # Now draw the frames
    print('Frames..')
    for i in range(len(T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(T)))))

        # Get wavefunction
        wav  = psi_T(T[i], X, data_N, data_L, data_m)
        prob = probability(wav)

        # Draw
        fig, ax = plt.subplots()

        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

        ax.plot(X, prob)
        plt.ylim(0, 1e-3)

        plt.xlabel('$x$')
        plt.ylabel('$\\mathcal{P}(x)$')

        plt.savefig('%s/%d.png' % (cache, i))
        plt.clf()
        plt.close()

    print('\nVideo..')
    os.system('ffmpeg -framerate 100 -pattern_type glob -i \'%s/*.png\' -c:v libx264 -pix_fmt yuv420p -preset slower %s/output.mp4' % (cache, directory))

    print('Cleanup..')
    shutil.rmtree(cache)
    plt.close('all') # just in case
    gc.collect()

    print('Done')

# Renders the p probability evolving over time
def run_animp():
    # Create cache directory
    directory = 'wavebox'
    cache     = directory + '/dump'

    if not os.path.exists(cache):
        os.makedirs(cache)

    # Firstly, build cache if it hasn't already been built
    run_buildcache()

    # Now draw the frames
    print('Frames..')
    for i in range(len(data_T)):
        sys.stdout.write('\r%s' % (progress_meter(float(i) / float(len(data_T)))))

        # Get wavefunction
        wav  = kpsi_T(data_T[i], data_X, data_N, data_L, data_m)
        prob = probability(wav, delta=(x_spacing / x_samples))

        # Draw
        fig, ax = plt.subplots()

        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

        ax.plot(hbar.value * data_K, prob)
        plt.ylim(0, 7e-1)
        # plt.xlim(-1e5, 1e5)

        plt.xlabel('$p$')
        plt.ylabel('$\\mathcal{P}(p)$')

        plt.savefig('%s/%d.png' % (cache, i))
        plt.clf()
        plt.close()

    print('\nVideo..')
    os.system('ffmpeg -framerate 100 -pattern_type glob -i \'%s/*.png\' -c:v libx264 -pix_fmt yuv420p -preset slower %s/outputp.mp4' % (cache, directory))

    print('Cleanup..')
    shutil.rmtree(cache)
    plt.close('all') # just in case
    gc.collect()

    print('Done')
