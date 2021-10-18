# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:27:01 2021

@author: wrose
"""

import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

#------------------------------------------------------------------
def model(t, P):
    
    # Define os parametros:
    S = 1.0e3 # Populacao maxima
    k = 0.10e1   # Constant of proportionality

    # Define the Newton's law of cooling:    
    return k*P*(1.0e0 - P/S)

# Define the initial conditions and integration boundaries:
#------------------------------------------------------------------
P_initial = 4e02
S = 1.0e3
k = 0.10e1
time_step = 1e-02
t_upper = 1e01
t_lower = 0e0

initial_conditions = np.array([P_initial]) # [Temperature(t=0) = T_initial]

points_to_plot = RK45(model, t0=t_lower, y0=initial_conditions, first_step = time_step, t_bound=t_upper, vectorized=False)

t_values = []
P_values = []
    
for i in range(10000):
    # get solution step state
    points_to_plot.step()
    t_values.append(points_to_plot.t)
    P_values.append(points_to_plot.y[0])
    # break loop after modeling is finished
    if points_to_plot.status == 'finished':
        break

P_analytical = S/(1.0 + np.exp(-k*np.array(t_values))*1.5e0)

plt.plot(t_values, P_values,'bo-',label='Numerico')
plt.plot(t_values, P_analytical,'r-',label='Analitico')
plt.xlabel('t [s]')
plt.ylabel('Populacao')
plt.legend()
plt.title("Modelo Logistico")
plt.grid(True)
plt.show()