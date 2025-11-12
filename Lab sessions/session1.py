"""
Problem sheet 2 Wednesday 8th and Wednesday 15th Oct
"""

import numpy as np 
import matplotlib.pyplot as plt 
from  scipy.integrate import trapezoid
import pandas as pd

def dft(x, N, T): # N should be a power of two or at least even where possible
    res = np.zeros(N, dtype=complex)
    for p in range(-N//2, N//2): # iterating over p space to find contributions to each p
        for n in range(0,N): # iterating over n's for the summation
            res[p + N//2] += x[n] * np.exp((2j * np.pi * n * p)/N)

    w_arr = (np.array(range(-N//2, N//2)) * 2 * np.pi) / T
    return np.real(res), w_arr
    
def invdft(w, N, T):
    x_rec = np.zeros(N, dtype=complex)
    for p in range(-N//2, N//2):
        for n in range(N):
            x_rec[n] += w[p + N//2] * np.exp((-2j * np.pi * n * p) / N) # sign flip and index change due to spectrum centered on zero
            
    x_rec /= (N) # amplitude normalisation
    t_arr = np.linspace(0, T, N, endpoint=False)
    return np.real(x_rec), t_arr

def gaussian(t,sigma):
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-((t)**2)/(2 * sigma**2)) 

def task1():
    pi = np.pi
    T = 10 * np.pi
    x = np.linspace(0, T, 1000)
    sine = np.sin(x)
    N = len(x)

    freqs, w = dft(sine, N, T)
    plt.plot(x, sine, 'x')
    plt.show()
    
    plt.grid(True)
    plt.plot(w, freqs)
    plt.show()

    inv, x_inv = invdft(freqs, N, T)
    plt.plot(x, sine, 'x')
    plt.plot(x_inv, inv, '.')
    plt.show()


def task2():
    """
    time axis is in ps, therefore the units of the frequency spectrum are per picosecond 10^12 s^-1

    generally, E_j = B_0 j(j+1)
        therefore, B_0 = E_0 = hbar * omega
    """
    laser_data = pd.read_csv('data.txt', delimiter=' ', header=None, names=['t', 'f(t)'])
    laser_data.plot(x='t', y='f(t)')
    plt.show()

    T = max(laser_data['t'])
    N = len(laser_data['f(t)'])

    spectrum, w = dft(x=laser_data['f(t)'], N=N, T=T)
    plt.plot(w, spectrum)
    plt.show()
    return laser_data['t'], spectrum, w

def task3():
    t, init_spectrum, init_w = task2()
    N = len(init_spectrum)
    T = max(t)

    curve = gaussian(t, 0.5) 
    plt.plot(t, curve, "x")
    plt.title("gaussian")
    print(trapezoid(curve, t))
    plt.show()

    gaussian_spect, w = dft(curve, len(curve), max(t))
    delta_w = sum(np.diff(w))/len(np.diff(w))
    plt.plot(w, gaussian_spect)
    plt.title('Spectrum')
    print(trapezoid(gaussian_spect,w))
    plt.show()

    convolved_spect = gaussian_spect * init_spectrum
    plt.plot(init_w, convolved_spect)
    plt.title('convolved spectrum')
    plt.show()

    f_t, t = invdft(convolved_spect, N, T)
    f_t *= (2*np.pi)/(delta_w*N) # accounting for factors lost by computing fft twice

    laser_data = pd.read_csv('data.txt', delimiter=' ', header=None, names=['t', 'f(t)'])
    laser_data.plot(x='t', y='f(t)', label='original')
    plt.plot(t, f_t, label='convolved')
    plt.title('time domain')
    plt.legend(loc='upper right')
    plt.show()

def task4():
    def decay(t, gamma):
        return np.exp(-t/gamma)
    laser_data = pd.read_csv('data.txt', delimiter=' ', header=None, names=['t', 'f(t)'])
    laser_data['alt_f(t)'] = laser_data['f(t)'] * decay(laser_data['t'], 30)
    T = max(laser_data['t'])
    N = len(laser_data['f(t)'])

    laser_data.plot(x='t', y=['f(t)','alt_f(t)'])
    plt.show()

    spectrum, w = dft(x=laser_data['f(t)'], N=N, T=T)
    spectrum_d, w_d = dft(x=laser_data['alt_f(t)'], N=N, T=T)
    plt.plot(w, spectrum, label='normal')
    plt.plot(w_d, spectrum_d, label='decay')
    plt.show()

def lagrange(x_data, y_data, x_eval):
    n = len(x_data)
    P = np.zeros_like(x_eval, dtype=float)
    for j in range(n):
        L_j = np.ones_like(x_eval, dtype=float)
        for k in range(n):
            if j != k:
                L_j *= (x_eval - x_data[k]) / (x_data[j] - x_data[k])
        P += y_data[j] * L_j
    return P  

# part 1 of problem sheet
def problem1(n):
    def function1(x):
        return np.sin(x)
    
    x_data = np.linspace(-np.pi, np.pi, n)
    y_data = function1(x_data)

    plt.plot(x_data, y_data, "x", label='data')

    x_eval = np.linspace(-1.1*np.pi, 1.1*np.pi, 400)
    # x_eval = x_data
    fit = lagrange(x_data, y_data, x_eval)
    plt.plot(x_eval, fit, label='interpolation')
    plt.legend()
    plt.show()


def problem2():
    def trapeziod(f, start, end, step_size, omit=[]):
        area = 0.
        for step in np.arange(start, end, step_size):
            if step in omit: 
                continue
            x_i = f(step)
            x_f = f(step + step_size)

            area += (x_i + x_f) / 2 * step_size
        return area
    

    def simpsons(f, start, end, step_size, omit=[]):
        area = 0.0
        half_step = step_size / 2.0

        for x0 in np.arange(start, end, step_size):
            x1 = x0 + half_step
            x2 = x0 + step_size

            if x2 > end:
                break

            if any(pt in omit for pt in [x0, x1, x2]):
                continue

            f0 = f(x0)
            f1 = f(x1)
            f2 = f(x2)

            area += (step_size / 6.0) * (f0 + 4*f1 + f2)

        return area
    
    def function(x):
        return np.sin(x**2)/x
    
    step_size = 0.001
    I = trapeziod(function, 0, 10, step_size, [0])
    I1 = simpsons(function, 0, 10, step_size, [0])
    print(I)
    print(I1)
            

if __name__ == '__main__':
    problem2()
