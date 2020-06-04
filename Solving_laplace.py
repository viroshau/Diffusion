#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:26:29 2020

@author: viroshaanuthayamoorthy
"""

import numpy as np
from scipy import integrate
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def integrand(x, n, V0): #The integrand of the A_n expression that takes an arbitrary function V_0(x)
    return 2*V0(x)*np.sin(np.pi*n*x)

def Necessary_coefficients(v0,tol,N_dim,title):
    x = np.linspace(0,1,N_dim)
    V = v0(x)
    V_calc = np.zeros(N_dim)
    i = 1
    while (np.linalg.norm((V-V_calc),ord = np.inf) > tol):
        V_calc = np.zeros(N_dim)
        A = np.zeros(i)
        for n in range(1,i+1):
            A[n-1] = quad(integrand, 0, 1, args=(n,v0))[0]
            V_calc += A[n-1]*np.sin(np.pi*n*x)
        if (i == 200):
            break
        i+= 1
    print("The Necessary number of fourier coefficients are: ",i-1)
    
    plt.figure()
    plt.plot(x,np.abs(V_calc - V),color = 'b', label= '$|V_{calc}(x) - V_{0}(x)$| at y = L')
    plt.xlabel('x/L')
    plt.legend()
    plt.title('V(x,y) = ' + title)
    plt.show()
    
    plt.figure()
    plt.plot(x,V,color = 'r', label= '$V_{0}(x)$ at y = L')
    plt.plot(x,V_calc,color = 'b', label= '$V_{calc}(x)$ at y = L')
    plt.xlabel('x/L')
    plt.legend()
    plt.title('V(x,y) = ' + title)
    plt.show()
    return i-1,A

def Entire_potential(tol,v0,N_dim,title):
    xx,yy = np.meshgrid(np.linspace(0,1,N_dim),np.linspace(0,1,N_dim))
    
    n_fourier, A = Necessary_coefficients(v0,tol,N_dim,title)

    Z = np.sinh(np.pi*0*yy)*np.sin(np.pi*0*xx)
        
    # Creates a list for both X and Y. 
    # Used to plot in 3D and in order to calculate the value of the potential at different points in the grid. 
    # Creates a matrix with all the values of the potential at different points (x,y). 
    # This specific command just generates V0, while all the other V(x,y) are left as zeroes.
    for n in range(1,n_fourier+1):
        Z += A[n-1]*np.sinh(np.pi*n*yy)/(np.sinh(np.pi*n))*np.sin(np.pi*n*xx)
   
    # To each coordinate we add on the fourier components. 
    # Since the pre-existing values were 0, the only value at these points are the fourier contributions
    
    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,Z, cmap = 'gist_rainbow')
    plt.contour(xx,yy,v0(xx),zdir = 'y',offset = 1, cmap = 'Dark2')
    plt.title('V(x,y) = ' + title)
    plt.xlabel('x/L')
    plt.ylabel('y/L')
    plt.show()
    
    dy, dx = np.gradient(Z)
    fig2 = plt.figure(figsize=(20, 20))
    fig2, ax2 = plt.subplots()
    skips = 25
    ax2.quiver(xx[::skips,::skips], yy[::skips,::skips], -1*dx[::skips,::skips], -1*dy[::skips,::skips])
    plt.ylim((0.5, 1))
    ax2.set(aspect=1, title='Electric Field')
    plt.xlabel('x/L')
    plt.ylabel('y/L')
    plt.show()
    
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    

def v1(x):
    return np.sin(2*np.pi*x)

def v2(x):
    return np.sin(5*np.pi*x)

def v3(x):
    return -4*x*(x-1)

def v4(x):
    return 1-(x-0.5)**2

def v5(x):
    return (0.5 * (np.sign(x-0.25) + 1)) - (0.5 * (np.sign(x-0.75) + 1))

V1 = Entire_potential(1e-15,v1,1000,"sin(2$\pi$x/L)")
V2 = Entire_potential(1e-15,v2,1000,"sin(5$\pi$x/L)")
V3 = Entire_potential(0.0001,v4,1000,"$1-(x/L-0.5)^2$")
V5 = Entire_potential(0.1,v5,1000,"$\Theta$(x/L - 0.25) - $\Theta$(x/L - 0.75)")