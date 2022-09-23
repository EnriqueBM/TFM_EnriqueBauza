#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:01:22 2021

@author: quique
"""

from sympy import *
from sympy.geometry import Point
import numpy as np
import random
import graphviz as gp
from matplotlib import pyplot as plt
#import spyder_memory_profiler
#from memory_profiler import profile
import time
import matplotlib.pyplot as plt


######## GLOBAL VARIABLES ###############

x, s, t, r = symbols('x s t r')
scale = 4.0
branch_list = [Mul, Add, sin, Pow, division]
#branch_list = [Mul, Add, sin, exp]
symbols_list = [r, t, x, s]

node_mutation_prob = 0.5
subtree_mutation_rate = 0.05
cross_rate = 0.45
point_rate = 0.15
hoist_rate = 0.1
subtree_rate = 0.15

n_ind = 5000
n_iter = 50

tmt_size = 3
tmt_rates = [0.9, 0.05, 0.05]
tmt_rates_mnm = [0.1, 0.1, 0.8]
mnm_th = 0.01



def objective_mult():
    #xl = Point(x, y, z)
    #xr = Point(0, 0, t)
    #d = xr.distance(xl)
    d = sqrt(r**2 + (t - x)**2)
    #Li = 1000*(exp(-s*t)*exp(-s*d)) / (d**2 * 4*pi)
    Li = (exp(-s*t)*exp(-s*d)) / (d**2 * 4*pi)
    
    return Li

def MC_integration(func, a, b, points):
    T = b - a
    f = lambdify(symbols_list, func, "numpy")
    vals= f(points[0], points[1], points[2], points[3]).flatten()
    summatory = np.sum(vals)
    n_samps = len(obj_vals)
    
    return (T/n_samps) * summatory