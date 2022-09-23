#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 00:38:18 2021

@author: quique
"""

from sympy import *
import numpy as np
import random
import operator

def gen_ind(functions):
    """
    Generate a new individual from the given functions.
    Each individual is a dict which will be transfom into a polynomial.

    Parameters
    ----------
    functions : List
        List of basic functions to generate the individual.

    Returns
    -------
    ind : sympy.Poly
        The individual.

    """
    n = random.randint(1, len(functions) - 1)
    gens =  random.sample(functions, n)
    gens = tuple(gens)
    pol = {}
    n_monom = random.randint(2, 4)
    #print(n_monom)
    idxs = [0, 1, 2]
    for i in range(n_monom):
        monom =  random.choices(idxs, k=len(gens))
        monom = tuple(monom)
        coeff = random.choice([-2, -1, 1, 2])
        pol[monom] = coeff
    #print(pol)
    ind = Poly.from_dict(pol, gens)
    
    return ind

def gen_ind_2(functions):
    """
    Generate a new individual from the given functions.
    Each individual is a dict which will be transfom into a polynomial.

    Parameters
    ----------
    functions : List
        List of basic functions to generate the individual.

    Returns
    -------
    ind : sympy.Poly
        The individual.

    """
    gens = tuple(functions)
    pol = {}
    n_monom = random.randint(2, 4)
    pows = [0, 1, 2]
    prob = 0.5 / (len(pows) - 1)
    ws = [0.5]
    for i in range(len(pows) - 1):
        ws.append(prob)
    for i in range(n_monom):
        monom =  random.choices(pows, weights=ws, k=len(gens))
        monom = tuple(monom)
        coeff = random.choice([-2, -1, 1, 2])
        pol[monom] = coeff
    #print(pol)
    ind = Poly.from_dict(pol, gens)
    
    return ind


def gen_pop(n, functions):
    """
    Generates the initial population

    Parameters
    ----------
    n : int
        Number of individuals.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    pop : List
        Generated initial population.

    """
    pop = []
    for i in range(n):
        ind_i = gen_ind_2(functions)
        pop.append(ind_i)
    obj = gen_ind_2(functions)
        
    return pop, obj

def replace_generator(gens, functions):
    """
    Replace a generator by a new one

    Parameters
    ----------
    gens : tuple
        Current generators for the polynomial.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_gens :tuple
        New generators for the polynomial.

    """
    gens_list = list(gens)
    gen_idx = random.randint(0, len(gens_list))
    changed = False
    while not(changed):
        func = random.choice(functions)
        if not(func in gens_list):
            gens_list[gen_idx] = func
            changed = True
    new_gens = tuple(gens_list)

    return new_gens

def add_generator(gens, functions):
    """
    Add a new geneator to the polynomial

    Parameters
    ----------
    gens : tuple
        Current generators for the polynomial.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_gens :tuple
        New generators for the polynomial.

    """
    gens_list = list(gens)
    added = False
    while not(added):
        func = random.choice(functions)
        if not(func in gens_list):
            gens_list.append(func)
            added = True
    new_gens = tuple(gens_list)

    return new_gens



def mute_key_3(key):
    """
    Mute a monom (key in the dict), increments or decrements
    the power of one of its functions.

    Parameters
    ----------
    key : list
        Original monomial.

    Returns
    -------
    key : list
        New monomial.

    """
    if (random.uniform(0, 1) < 0.5):
        idx = random.randint(0, len(key) - 1)
        while (key[idx] == 0): # No generar nuevas, eso en la otra mutacion
            idx = random.randint(0, len(key) - 1)
        key[idx] += 1 ## ¿Cuanto sumar/restar? ¿Penalizar potencias altas --> Overfitting?
    else:

        if not(all(power <= 1 for power in key)):
            idx = random.randint(0, len(key) - 1)
            while (key[idx] <= 1): # No eliminar existentes, eso en la otra mutacion
                idx = random.randint(0, len(key) - 1)
            key[idx] -= 1 ## ¿Cuanto sumar/restar? ¿Penalizar potencias altas --> Overfitting?
    
    return key

def mute_monoms_3(ind):
    """
    Change one monom of the polynomial with mute_key_3.

    Parameters
    ----------
    ind : Sympy polynomial
        Current individual.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_ind : Sympy polynomial.
        The muted individual

    """
    generators = ind.gens
    ind_dict = ind.as_dict()
    new_ind_dict = {}
    if len(ind_dict.keys()) != 0:
        n_monom = random.randint(1, len(ind_dict.keys()))
        i = 0
        for key in ind_dict.keys():
            new_key = list(key)
            i = i + 1
            if (i == n_monom) and not(all(power == 0 for power in new_key)):
                #print(new_key)
                new_key = mute_key_3(new_key)
                #print(new_key)
            new_key = tuple(new_key)
            new_ind_dict[new_key] = ind_dict[key]
        new_ind = Poly.from_dict(new_ind_dict, generators) 
    else:
        new_ind = ind          
            
    return new_ind

def mute_key_2(key):
    """
    Mute a monom (key in the dict), replaces one of the functions for other of the base.

    Parameters
    ----------
    key : list
        Original monomial.

    Returns
    -------
    key : list
        New monomial.

    """
    idxs = random.sample(range(len(key)), 2)
    aux = key[idxs[0]]
    key[idxs[0]] = key[idxs[1]]
    key[idxs[1]] = aux
    
    return key

def mute_monoms_2(ind):
    """
    Change one monom of the polynomial with mute_key_2.

    Parameters
    ----------
    ind : Sympy polynomial
        Current individual.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_ind : Sympy polynomial.
        The muted individual

    """
    generators = ind.gens
    ind_dict = ind.as_dict()
    new_ind_dict = {}
    if (len(ind_dict.keys()) != 0):
        n_monom = random.randint(1, len(ind_dict.keys()))
        i = 0
        for key in ind_dict.keys():
            new_key = list(key)
            i = i + 1
            if (i == n_monom):
                #print(new_key)
                new_key = mute_key_2(new_key)
                #print(new_key)
            new_key = tuple(new_key)
            new_ind_dict[new_key] = ind_dict[key]
        new_ind = Poly.from_dict(new_ind_dict, generators)   
    else:
        new_ind = ind
            
    return new_ind

def mute_monoms_4(ind):
    """
    Generates a new monom

    Parameters
    ----------
    ind : Sympy polynomial
        Current individual.

    Returns
    -------
    new_ind : Sympy polynomial.
        The muted individual

    """
    generators = ind.gens
    ind_dict = ind.as_dict()
    new_ind_dict = {}
    if (len(ind_dict.keys()) != 0):
        for key in ind_dict.keys():
            new_key = key
            new_ind_dict[new_key] = ind_dict[key]
            
        pows = [0, 1, 2]
        prob = 0.5 / (len(pows) - 1)
        ws = [0.5]
        for i in range(len(pows) - 1):
            ws.append(prob)
        monom =  random.choices(pows, weights=ws, k=len(generators))
        monom = tuple(monom)
        coeff = random.choice([-2, -1, 1, 2])
        new_ind_dict[monom] = coeff
        new_ind = Poly.from_dict(new_ind_dict, generators)   
    else:
        pows = [0, 1, 2]
        prob = 0.5 / (len(pows) - 1)
        ws = [0.5]
        for i in range(len(pows) - 1):
            ws.append(prob)
        monom =  random.choices(pows, weights=ws, k=len(generators))
        monom = tuple(monom)
        coeff = random.choice([-2, -1, 1, 2])
        new_ind_dict[monom] = coeff
        new_ind = Poly.from_dict(new_ind_dict, generators)   
            
    return new_ind

def mute_monoms(ind, functions):
    """
    Change one monom of the polynomial

    Parameters
    ----------
    ind : Sympy polynomial
        Current individual.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_ind : Sympy polynomial.
        The muted individual

    """
    generators = ind.gens
    ind_dict = ind.as_dict()
    new_ind_dict = {}
    muted = False
    rate1 = 0.5
    new_gens = add_generator(generators, functions)
    for key in ind_dict.keys():
        new_key = list(key)
        new_key.append(0)
        if (random.uniform(0, 1) < rate1) and not(muted):
            print(new_key)
            new_key = mute_key_2(new_key)
            print(new_key)
            muted = True
        new_key = tuple(new_key)
        new_ind_dict[new_key] = ind_dict[key]
    new_ind = Poly.from_dict(new_ind_dict, new_gens)           
            
    return new_ind


def mute_coeffs(ind):
    """
    Change one coefficient from the polynomial

    Parameters
    ----------
    ind : Sympy polynomial
        Current individual.
    functions : List
        Functions to generate the individuals.

    Returns
    -------
    new_ind : Sympy polynomial.
        The muted individual

    """
    rate1 = 0.3
    ind_dict = ind.as_dict()
    muted = False
    for key in ind_dict.keys(): #the key is the monom, the value the coeff
        if (random.uniform(0, 1) < rate1) and not(muted):
            coeff_mut = random.randint(-4, 4)
            ind_dict[key] = ind_dict[key] + coeff_mut #mut the coefficient
            muted = True
    new_ind = Poly.from_dict(ind_dict, ind.gens)
    
    return new_ind      

def crossover(parent1, parent2):
    gens = parent1.gens
    prt1_dict = parent1.as_dict()
    prt2_dict = parent2.as_dict()
    ch1 = {}
    ch2 = {}
    ch1_mon = []
    ch2_mon = []
    child_id = 1
    for key in prt1_dict.keys():
        if child_id == 1:
            ch1[key] = prt1_dict[key]
            ch1_mon.append(key)
            child_id = 2
        else:
            ch2[key] = prt1_dict[key]
            ch2_mon.append(key)
            child_id = 1
    child_id = 2
    for key in prt2_dict.keys():
        if child_id == 1:
            if not(key in ch1_mon):    
                ch1[key] = prt2_dict[key]
                child_id = 2
            else:
                ch2[key] = prt2_dict[key]
        else:
            if not(key in ch2_mon):    
                ch2[key] = prt2_dict[key]
                child_id = 1
            else:
                ch1[key] = prt2_dict[key]
            
    child1 = Poly.from_dict(ch1, gens)
    child2 = Poly.from_dict(ch2, gens)
    
    return child1, child2

def selection(population, scores):
    n_pop = len(population)
    scores_idx = np.argsort(scores)
    ord_scores = np.sort(scores)
    select_pop = []
    for idx in scores_idx:
        select_pop.append(population[idx])
    n_select = int(0.8 * n_pop)
    next_pop = select_pop[:n_select]
    
    return next_pop

def fitness(ind, obj_ev, samps):
    f = lambdify(x, ind.as_expr(), "numpy")
    evals = f(samps)
    abs_errors = np.abs(obj_ev - evals)
    fit = np.sum(abs_errors)
    
    return fit

def gen_samples(n_samples, inf, sup):
    samples = []
    step = (sup - inf) / n_samples
    for i in range(n_samples):
        sample = inf + step * i
        samples.append(sample)
    samples = np.array(samples)
    
    return samples
    
        


if __name__ == "__main__":
    
    ## Definimos nuestro espacio de configuraciones (x, sen(x), 1, exp(x)):
    x = symbols('x')
    e1 = x 
    e2 = sin(x)
    #e3 = exp(x)
    #e4 = 1
    e5 = log(x)
    
    
    base = [e1, e2, e5]
    pop = gen_pop(3, base)
    
    #Generamos la poblacion inicial
    n_ind = 10
    pop, objective = gen_pop(n_ind, base)
    obj_ex = objective.as_expr()
    
    
    n_iter = 20000
    samps = gen_samples(5000, 0.01, 5)
    f_obj = lambdify(x, objective.as_expr(), "numpy")
    obj_vals = f_obj(samps)
    
    #Mutation
    """print(pop)
    for i in range(len(pop)):
        #if random.uniform(0, 1) < 0.35:
            pop[i] = mute_monoms_3(pop[i])
        #if random.uniform(0, 1) < 0.35:
            pop[i] = mute_monoms_2(pop[i])
        #if random.uniform(0, 1) < 0.35:
            pop[i] = mute_coeffs(pop[i])
    print(pop)"""
    
    for iter in range(n_iter):
        if (iter % 1000 == 0):
            print(iter)
        #evualte pop to select parents
        scores = np.zeros(len(pop))
        for i in range(len(pop)):
            #print(type(pop[i]))
            scores[i]= fitness(pop[i], obj_vals, samps)
        if (iter % 1000 == 0):
            print(pop)
            print(scores)
            #print(np.linalg.norm(scores))
                
        #Selection
        next_pop = selection(pop, scores)
        n_parents = int(len(pop) / 5)
        parents = next_pop[:n_parents]
        if (iter % 1000 == 0):
            best_ex = next_pop[0].as_expr()
            p = plot(best_ex, obj_ex, show=False)        
            p[1].line_color= 'red'
            p.xlim=(0,5)
            p.show()
        
        #Crossover
        for i in range(int(len(parents)/2)):
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            # c1 = crossover_1(p1, p2)
            next_pop.append(c1)
            next_pop.append(c2)
            
        #Mutation
        for i in range(len(next_pop)):

            if random.uniform(0, 1) < 0.1:
                next_pop[i] = mute_monoms_3(next_pop[i])
            elif random.uniform(0, 1) < 0.1:
                next_pop[i] = mute_coeffs(next_pop[i])
            elif random.uniform(0, 1) < 0.1:
                next_pop[i] = mute_monoms_2(next_pop[i])
            elif random.uniform(0, 1) < 0.01:
                next_pop[i] = mute_monoms_4(next_pop[i])                
        #Update the population
        pop = next_pop.copy()
    for i in range(len(pop)):
        scores[i]= fitness(pop[i], obj_vals, samps)
    print("Poblacion final:")
    print(pop)
    print(scores)