
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
    k = -0.10e1   # Constant of proportionality

    # Define the Newton's law of cooling:    
    return k*P

# Define the initial conditions and integration boundaries:
#------------------------------------------------------------------
P_initial = 5e02
k = -0.10e1
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

P_analytical = P_initial*np.exp(k*np.array(t_values))

plt.plot(t_values, P_values,'bo-',label='Numerico')
plt.plot(t_values, P_analytical,'r-',label='Analitico')
plt.xlabel('t [s]')
plt.ylabel('Populacao')
plt.legend()
plt.title("Crescimento populacional")
plt.grid(True)
plt.show()