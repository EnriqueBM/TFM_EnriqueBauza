#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:07:36 2021

@author: quique
"""

from sympy import *
from sympy.geometry import Point
import numpy as np

x, s, t, r = symbols('x s t r')
symbols_list = [r, t, x]

if __name__ == "__main__":
    
    objective = r**2 + t - 2*x
    
    #samples generation
    x0 = np.arange(0, 3, 1)
    x1 = np.arange(0, 2, 1)
    x2 = np.arange(0, 4, 1)
    x3 = np.arange(0, 1, 1)
    points = np.meshgrid(x0, x1, x2)
    
    #objective function evaluation
    f_obj = lambdify(symbols_list, objective, "numpy")
    obj_vals= f_obj(points[0], points[1], points[2])
    flat_vals = obj_vals.flatten()