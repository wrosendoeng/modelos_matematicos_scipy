# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:27:01 2021

@author: wrose
"""

import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

# Model the calling function [X_dot]:
#------------------------------------------------------------------
def model(t, T):
    
    # Define the ambient temperature:
    T_amb = 3e2 # Temperature in Kelvin
    k = -1.5e1   # Constant of proportionality

    # Define the Newton's law of cooling:    
    return k*(T - T_amb)

# Define the initial conditions and integration boundaries:
#------------------------------------------------------------------
T_initial = 4e02
T_amb = 3e02
k = -1.5e1
time_step = 1e-02
t_upper = 1e01
t_lower = 0e0

initial_conditions = np.array([T_initial]) # [Temperature(t=0) = T_initial]

points_to_plot = RK45(model, t0=t_lower, y0=initial_conditions, first_step = time_step, t_bound=t_upper, vectorized=False)

t_values = []
T_values = []
    
for i in range(10000):
    # get solution step state
    points_to_plot.step()
    t_values.append(points_to_plot.t)
    T_values.append(points_to_plot.y[0])
    # break loop after modeling is finished
    if points_to_plot.status == 'finished':
        break

T_analytical = T_amb + (T_initial - T_amb)*np.exp(k*np.array(t_values))

plt.plot(t_values, T_values,'--',label='Numerical')
plt.plot(t_values, T_analytical,label='Analytical')
plt.xlabel('t [s]')
plt.ylabel('T [K]')
plt.legend()
plt.title("Newton's Law of Cooling")
plt.grid(True)
plt.show()