#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:35:59 2021

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
from scipy.special import logsumexp


######## GLOBAL VARIABLES ###############


def division(*args):
    return Mul(args[0], Pow(args[1], -1))

def resta(*args):
    return Add(args[0], Mul(-1, args[1]))

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

def objective_Li():
    xl = Point(2.5, 2.5, 2.5)
    xr = Point(0, 0, x)
    d = xr.distance(xl)
    Li = 1000*(exp(-0.1*x)*exp(-0.1*d)) / (d**2 * 4*pi)
    
    return Li

def gen_leaf():
    float_rate = 1 / (len(symbols_list) + 1)
    if random.uniform(0, 1) < float_rate:
    #if random.uniform(0, 1) < 0.5:
        return N(sympify(scale * random.uniform(-1, 1)), 3)
    else:
        return random.choice(symbols_list)
    
def gen_branch_2(depth, max_args):
    """
    Generates a randon tree with the given depth, can be used to generate the
    full expression or to mute an existing tree.

    Parameters
    ----------
    depth : Int
        Tree structure depth.
    n_args : Int
        Number of arguments for the expressions in the tree.

    Returns
    -------
    TYPE Sympy.expression
        The randomly generated expression.

    """
    func = random.choice(branch_list)
    #Elegimos número de arguemtnos dependiendo de la funcion
    if func == Pow:
        n_args = 2
    elif func == log:
        n_args = 1
    elif arity(func) != None: #NO es Add ni Mul ni Pow, es sin o exp
        n_args = arity(func)
    else: #Add o Mul
        n_args = max_args
    # Generamos el tree    
    if depth == 1:
        largs = [gen_leaf() for auxi in range(n_args)]
        return func(*largs)
    else:
        largs = [gen_branch_2(depth-1, max_args) for auxi in range(n_args)]
        return func(*largs)
    
def gen_branch_3(depth, max_args):
    """
    Generates a randon tree with the given depth, can be used to generate the
    full expression or to mute an existing tree.

    Parameters
    ----------
    depth : Int
        Tree structure depth.
    n_args : Int
        Number of arguments for the expressions in the tree.

    Returns
    -------
    TYPE Sympy.expression
        The randomly generated expression.

    """
    func = random.choice(branch_list)
    #Elegimos número de arguemtnos dependiendo de la funcion
    if func == division:
        n_args = 2
    if func == Pow:
        n_args = 2
    elif arity(func) != None: #NO es Add ni Mul ni Pow, es sin o exp
        n_args = arity(func)
    else: #Add o Mul
        n_args = max_args
    # Generamos el tree    
    if depth == 1:
        largs = [gen_leaf() for auxi in range(n_args)]
        return func(*largs)
    else:
        largs = [gen_branch_3(depth-1, max_args) for auxi in range(n_args)]
        return func(*largs)
    
def plot_tree(ex, n_file):
    gp.Source(dotprint(ex), filename="test" + str(n_file) + ".gv" , format="png").view()
    
def mute_1(expr):
    """
    Substitutes a leaf by a subtree

    Parameters
    ----------
    expr : Sympy.expression
        Expression to mute.

    Returns
    -------
    TYPE Sympy.expression
        Muted expression.

    """
    depth = random.randint(1, 2)
    args = random.randint(2, 3)
    if len(expr.args) != 0: #Por si el individuo se ha simplificado demasiado
        idx = random.choice(range(len(expr.args)))
        if len(expr.args[idx].args) == 0: ## Node before leaf, mute here
            largs = list(expr.args)
            largs[idx] = gen_branch_2(depth, args)
            return expr.func(*largs)
        elif random.uniform(0, 1) < subtree_mutation_rate:
            largs = list(expr.args)
            largs[idx] = gen_branch_2(depth, args)
            return expr.func(*largs)
        else:
            largs = list(expr.args)
            largs[idx] = mute_1(expr.args[idx])
            return expr.func(*largs)
    else:
        return gen_branch_2(2, 2)
            
def mute_2(expr):
    """
    Reduces the depth of a branch
    
    Parameters
    ----------
    expr : Sympy.expression
        Expression to mute.
    
    Returns
    -------
    TYPE Sympy.expression
        Muted expression.
    
    """
    if len(expr.args) != 0: #Si no esta ya muy reducido
        idx_1 = random.choice(range(len(expr.args)))
        i = 0
        while (len(expr.args[idx_1].args) == 0): #Search a 2-depth subtree or return the same
            if (i < 2 * len(expr.args)):
                idx_1 = random.choice(range(len(expr.args)))
                i += 1
            else:
                return expr.func(*expr.args)
              
        idx_2 = random.choice(range(len(expr.args[idx_1].args)))
        if len(expr.args[idx_1].args[idx_2].args) == 0: ## 2 Nodes before leaf
            largs = list(expr.args)
            largs[idx_1] = expr.args[idx_1].args[idx_2]
            return expr.func(*largs)
        else:
            largs = list(expr.args) #Sigue bajando
            largs[idx_1] = mute_2(expr.args[idx_1])
            return expr.func(*largs)
    else:
        return expr
    
def mute_3_old(expr):
    """
    Point mutation. Replaces a node by other function or a leaf.
    
    Parameters
    ----------
    expr : Sympy.expression
        Expression to mute.
    
    Returns
    -------
    TYPE Sympy.expression
        Muted expression.
    
    """
    if len(expr.args) != 0: ## Compruebo que individio > 1 nodo 
        idx = random.choice(range(len(expr.args)))
        if len(expr.args[idx].args) == 0: ## Node before leaf, replace a leaf
            largs = list(expr.args)
            largs[idx] = gen_leaf()
            return expr.func(*largs)
        elif random.uniform(0, 1) < node_mutation_prob: ## Replace the node function
            largs = list(expr.args)
            func = random.choice(branch_list) #Nueva funcion
            if func != Pow:             
                if arity(func) != None: # La nueva no es Add ni Mul (es sin o exp)
                    if (arity(func) != len(expr.args)):
                        largs = random.sample(list(expr.args), arity(func))
                    else: # arity nueva = arity vieja
                        largs = list(expr.args) 
                elif len(expr.args) == 1: #Si sustituimos sin o exp por Add o Mul
                    largs = list(expr.args)
                    largs.append(gen_leaf()) ## Le generamos un segundo argumento
                else:
                    largs = list(expr.args) #Add o Mul por Add o Mul
            elif len(expr.args) == 1: #Pongo potencia por seno o exp
                    largs = list(expr.args)
                    largs.append(gen_leaf())
            else: #pongo potencia por mul, add o pow
                    largs = random.sample(list(expr.args), 2)
            return func(*largs)
        else: ## Pasa a otro nodo
            largs = list(expr.args)
            largs[idx] = mute_3(expr.args[idx])
            return expr.func(*largs)
    else:
        return gen_branch_2(2, 2)
    
    
def mute_3(expr):
    """
    Point mutation. Replaces a node by other function or a leaf.
    
    Parameters
    ----------
    expr : Sympy.expression
        Expression to mute.
    
    Returns
    -------
    TYPE Sympy.expression
        Muted expression.
    
    """   
    if len(expr.args) != 0: ## Compruebo que individio > 1 nodo 
        idx = random.choice(range(len(expr.args)))
        if len(expr.args[idx].args) == 0: ## Node before leaf, replace a leaf
            largs = list(expr.args)
            largs[idx] = gen_leaf()
            return expr.func(*largs)
        elif random.uniform(0, 1) < node_mutation_prob: ## Replace the node function
            largs = list(expr.args)
            func = random.choice(branch_list) #Nueva funcion
            if func == Pow or func == division:    
                if len(expr.args) < 2: #Pongo potencia o division por seno o exp
                        largs = list(expr.args)
                        largs.append(gen_leaf())
                else: #pongo potencia o division por mul, add o pow
                        largs = random.sample(list(expr.args), 2)
            elif func == Add or func == Mul:
                if len(expr.args) == 1:
                    largs = list(expr.args)
                    largs.append(gen_leaf()) ## Le generamos un segundo argumento
                else:
                    largs = list(expr.args)
            elif func == sin or func == exp or func == log:
                if (len(expr.args) > 1):
                        largs = random.sample(list(expr.args), 1)
                else: # arity nueva = arity vieja
                        largs = list(expr.args)
            return func(*largs)
        else: ## Pasa a otro nodo
            largs = list(expr.args)
            largs[idx] = mute_3(expr.args[idx])
            return expr.func(*largs)
    else:
        return gen_branch_2(2, 2)
  
            
    
def crossover(p1, p2, r_cross):
    if random.random() < r_cross:
        if (len(p1.args) != 0) and (len(p2.args) != 0):
            idx_1 = random.choice(range(len(p1.args)))
            idx_2 = random.choice(range(len(p2.args)))
            largs_1 = list(p1.args)
            largs_2 = list(p2.args)
            donned = largs_2[idx_2]
            largs_2[idx_2] = largs_1[idx_1]
            largs_1[idx_1] = donned
            ch1 = p1.func(*largs_1)
            ch2 = p2.func(*largs_2)
        elif (len(p1.args) != 0) and (len(p2.args) == 0):
            idx_1 = random.choice(range(len(p1.args)))
            largs_1 = list(p1.args)
            donned = p2
            ch2 = largs_1[idx_1]
            largs_1[idx_1] = donned
            ch1 = p1.func(*largs_1)
        elif (len(p1.args) == 0) and (len(p2.args) != 0):
            #idx_1 = random.choice(range(len(p1.args)))
            idx_2 = random.choice(range(len(p2.args)))
            largs_2 = list(p2.args)
            ch1 = largs_2[idx_2]
            largs_2[idx_2] = p1
            ch2 = p2.func(*largs_2)
        else:
            ch1 = p1
            ch2 = p2
    else:
        ch1 = p1
        ch2 = p2 
    
    return ch1, ch2

def crossover_gp(p1, p2):
    if (len(p1.args) != 0) and (len(p2.args) != 0):
        idx_1 = random.choice(range(len(p1.args)))
        idx_2 = random.choice(range(len(p2.args)))
        largs_1 = list(p1.args)
        largs_2 = list(p2.args)
        donned = largs_2[idx_2]
        largs_2[idx_2] = largs_1[idx_1]
        largs_1[idx_1] = donned
        ch1 = p1.func(*largs_1)
    elif (len(p1.args) != 0) and (len(p2.args) == 0):
        idx_1 = random.choice(range(len(p1.args)))
        largs_1 = list(p1.args)
        donned = p2
        largs_1[idx_1] = donned
        ch1 = p1.func(*largs_1)
    elif (len(p1.args) == 0) and (len(p2.args) != 0):
        #idx_1 = random.choice(range(len(p1.args)))
        idx_2 = random.choice(range(len(p2.args)))
        largs_2 = list(p2.args)
        ch1 = largs_2[idx_2]
        largs_2[idx_2] = p1
    else:
        ch1 = p1
    
    return ch1

# tournament selection
def tmnt_selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def tmnt_selection_2(pop, scores, rates, k=3):
    scrs = []
    inds = []
    for ix in np.random.randint(0, len(pop), k):
        scrs.append(scores[ix])
        inds.append(pop[ix])
    ord_idxs = np.argsort(scrs)
    r = random.uniform(0, 1)
    if r < rates[0]:
        return inds[ord_idxs[0]]
    elif r < rates[0] + rates[1]:
        return inds[ord_idxs[1]]
    else:
        return inds[ord_idxs[2]]
    
def tmnt_selection_3(pop, scores, rates, k=3):
    scrs = []
    inds = []
    while len(inds) < k:
        ix = np.random.randint(0, len(pop))
        if scores[ix] > 0:
            scrs.append(scores[ix])
            inds.append(pop[ix])
    ord_idxs = np.argsort(scrs)
    r = random.uniform(0, 1)
    if r < rates[0]:
        return inds[ord_idxs[0]]
    elif r < rates[0] + rates[1]:
        return inds[ord_idxs[1]]
    else:
        return inds[ord_idxs[2]]

def evolution(pop, scores, rates, i, srt_idx, k=3):
    if (i < len(pop)/2):
        evolved = pop[srt_idx[i]]
    else:
    # perform torunament
        candidate = tmnt_selection_3(pop, scores, rates, k)
        # perform evolution operation based on each rate
        r = random.uniform(0, 1)
        if (r < cross_rate):
            donnor = tmnt_selection(pop, scores, k)
            evolved = crossover_gp(candidate, donnor) # crossover
        elif (r < cross_rate + hoist_rate):
            evolved = mute_2(candidate) #hoist mutation
        elif (r < cross_rate + hoist_rate + point_rate):
            evolved = mute_3(candidate) #point mutation
        elif (r < cross_rate + hoist_rate + point_rate + subtree_rate):
            evolved = mute_1(candidate) #subtree mutation
        else:
            evolved = candidate #unchanged
    return evolved




def fitness_mse_1d(ind, obj_ev, samps):
    """
    Computes the MSE of an individual for a given set of points to sample

    Parameters
    ----------
    ind : expression
        The ind to evaluate.
    obj_ev : expression
        The objective function.
    samps : np.array
        Set of points to sample.

    Returns
    -------
    mse : float
        DESCRIPTION.

    """
    n_samps = len(samps)
    #f = lambdify(x, ind, "numpy")
    f = lambdify([t, x, y, z], ind, "numpy")
    ################### ¿CATCH WARNINGS AQUI? ###############
    with np.errstate(divide='raise', invalid='raise', over='raise'):
            try:
                evals = f(samps)

                error = obj_ev - evals
                sq_error = np.square(error)
    
                mse = np.abs(np.sum(sq_error) / n_samps)
                if isinstance(evals, np.ndarray):
                    for i in range(len(evals)):
                        if isinstance(evals[i], np.complex128):
                            mse = 1000000
                            break
            except ZeroDivisionError:
                mse = 100000
            except OverflowError:
                mse = 100000
            except FloatingPointError:
                mse = 100000

    
    return mse

#@profile
def fitness_mse(ind, obj_ev, samps, n):
    """
    Computes the MSE of an individual for a given set of points to sample

    Parameters
    ----------
    ind : expression
        The ind to evaluate.
    obj_ev : expression
        The objective function.
    samps : np.array
        Set of points to sample.

    Returns
    -------
    mse : float
        DESCRIPTION.

    """

    ################### ¿CATCH WARNINGS AQUI? ###############
    with np.errstate(divide='raise', invalid='raise', over='raise'):
            try:

                f = lambdify(symbols_list, ind, "numpy")
                evals= f(samps[0], samps[1], samps[2])
                if isinstance(evals, np.ndarray):
                    evals = evals.flatten()

                error = obj_ev - evals
                #sq_error = np.square(error)
                #mae = np.sum(np.abs(error)) / n
                mae = np.mean(np.abs(error))
                #mse = np.abs(np.sum(sq_error) / n)
                if isinstance(evals, np.ndarray):
                    for i in range(len(evals)):
                        if isinstance(evals[i], np.complex128):
                            mae = 10000000
                            break
            except ZeroDivisionError:
                mae = 10000000
            except OverflowError:
                mae = 10000000
            except FloatingPointError:
                mae = 10000000
            except KeyError:
                mae = 10000000

    
    return mae


def fitness_diff(ind, obj_ev, samps, n):
    """
    Computes the MSE of an individual for a given set of points to sample

    Parameters
    ----------
    ind : expression
        The ind to evaluate.
    obj_ev : expression
        The objective function.
    samps : np.array
        Set of points to sample.

    Returns
    -------
    mse : float
        DESCRIPTION.

    """

    ################### ¿CATCH WARNINGS AQUI? ###############
    with np.errstate(divide='raise', invalid='raise', over='raise'):
            try:
                d_ind = diff(ind, t)
                f = lambdify(symbols_list, d_ind, "numpy")
                evals= f(samps[0], samps[1], samps[2], samps[3])
                if isinstance(evals, np.ndarray):
                    evals = evals.flatten()

                error = obj_ev - evals
                #sq_error = np.square(error)
                #mae = np.sum(np.abs(error)) / n
                mae = np.mean(np.abs(error))
                #mse = np.abs(np.sum(sq_error) / n)
                if isinstance(evals, np.ndarray):
                    for i in range(len(evals)):
                        if isinstance(evals[i], np.complex128):
                            mae = 10000000
                            break
            except ZeroDivisionError:
                mae = 10000000
            except OverflowError:
                mae = 10000000
            except FloatingPointError:
                mae = 10000000
            except KeyError:
                mae = 10000000

    
    return mae



def fitness_dict(ind, obj_ev, samps, n, v_dict, n_rep):
    """
    Computes the MSE of an individual for a given set of points to sample

    Parameters
    ----------
    ind : expression
        The ind to evaluate.
    obj_ev : expression
        The objective function.
    samps : np.array
        Set of points to sample.

    Returns
    -------
    mse : float
        DESCRIPTION.

    """
    
    mae = v_dict.get(ind, -2)
    
    if mae == -2:
        ################### ¿CATCH WARNINGS AQUI? ###############
        with np.errstate(divide='raise', invalid='raise', over='raise'):
                try:
    
                    f = lambdify(symbols_list, ind, "numpy")
                    evals= f(samps[0], samps[1], samps[2])
                    if isinstance(evals, np.ndarray):
                        evals = evals.flatten()
    
                    error = np.abs(obj_ev - evals)
                    
                    #sq_error = np.square(error)
                    #mae = np.sum(np.abs(error)) / n
                    mae = np.mean(error)
                    
                    #print(n)
                    #mae = np.abs(np.sum(sq_error) / n)
                    if isinstance(evals, np.ndarray):
                        for i in range(len(evals)):
                            if isinstance(evals[i], np.complex128):
                                mae = -10000000
                                break
                        if not mae == -10000000:
                            #print(mae)
                            hist_aux=plt.hist(error)
                            print(hist_aux)
                            plt.show()
                except ZeroDivisionError:
                    mae = -10000000
                except OverflowError:
                    mae = -10000000
                except FloatingPointError:
                    mae = -10000000
                except KeyError:
                    mae = -10000000
        
        v_dict[ind] = mae
    else:
        n_rep += 1

    
    return mae, n_rep



def fitness_try(ind, obj_ev, samps, n, v_dict, n_rep):
    """
    Computes the MSE of an individual for a given set of points to sample

    Parameters
    ----------
    ind : expression
        The ind to evaluate.
    obj_ev : expression
        The objective function.
    samps : np.array
        Set of points to sample.

    Returns
    -------
    mse : float
        DESCRIPTION.

    """
    
    mae = v_dict.get(ind, -2)
    sorted_error = np.zeros(n)
    if mae == -2:
        ################### ¿CATCH WARNINGS AQUI? ###############
        with np.errstate(divide='raise', invalid='raise', over='raise'):
                try:
    
                    f = lambdify(symbols_list, ind, "numpy")
                    evals= f(samps[0], samps[1], samps[2])
                    if isinstance(evals, np.ndarray):
                        evals = evals.flatten()
    
                    error = np.abs(obj_ev - evals)
                    sorted_error = np.argsort(error)
                    error = np.sort(error)
                    error = error[0:-100]
                    
                    #sq_error = np.square(error)
                    #mae = np.sum(np.abs(error)) / n
                    mae = np.mean(error)
                    
                    #print(n)
                    #mae = np.abs(np.sum(sq_error) / n)
                    if isinstance(evals, np.ndarray):
                        for i in range(len(evals)):
                            if isinstance(evals[i], np.complex128):
                                mae = -10000000
                                break
                        """if not mae == -10000000:
                            #print(mae)
                            hist_aux=plt.hist(error)
                            print(hist_aux)
                            plt.show()"""
                    
                    return mae, sorted_error  
                    
                except ZeroDivisionError:
                    mae = -10000000
                except OverflowError:
                    mae = -10000000
                except FloatingPointError:
                    mae = -10000000
                except KeyError:
                    mae = -10000000
        
        v_dict[ind] = mae
    else:
        n_rep += 1

    
    return mae, sorted_error

def gen_samples(n_samples, inf, sup):
    """
    Generates a set of points to evaluate a function

    Parameters
    ----------
    n_samples : int
        Number of samples.
    inf : TYPE
        Lower limit of the interval to sample.
    sup : TYPE
        Upper limit of the inerval to sample.

    Returns
    -------
    samples : np.array
        Points to sample.

    """
    samples = []
    step = (sup - inf) / n_samples
    for i in range(n_samples):
        sample = inf + step * i
        samples.append(sample)
    samples = np.array(samples)
    
    return samples

def selection(population, scores, percentage):
    """
    Selects the given percentage of best indivuals from the given population.

    Parameters
    ----------
    population : list
        DESCRIPTION.
    scores : TYPE
        DESCRIPTION.
    percentage : float (0 to 1)
        DESCRIPTION.

    Returns
    -------
    next_pop : list
        DESCRIPTION.
    ord_scores : TYPE
        DESCRIPTION.

    """
    n_pop = len(population)
    scores_idx = np.argsort(scores)
    ord_scores = np.sort(scores)
    select_pop = []
    for idx in scores_idx:
        select_pop.append(population[idx])
    n_select = int(percentage * n_pop)
    next_pop = select_pop[:n_select]
    
    return next_pop, ord_scores

def gen_pop(n):
    """
    Generates the initial population

    Parameters
    ----------
    n : int
        Number of individuals.

    Returns
    -------
    pop : List
        Generated initial population.
    obj : Expression
        The objective function

    """
    pop = []
    for i in range(n):
        ind_i = gen_branch_2(3, 4)
        pop.append(ind_i)
    obj = gen_branch_2(4, 2)
        
    return pop, obj

def local_minimum(th, mean0, mean1):
    err = np.abs(mean0 - mean1)
    m = (mean0 + mean1)/2
    p_err= (err / m)
    if p_err < th:
        return True
    else:
        return False
    
def MC_integration(N, a=1.5, b=11.5):
    T = abs(b - a)
    step = T/N
    obj = objective_mult()
    
    
    x0 = np.arange(a, b, step)
    x1 = np.arange(1.5, 2.5, 1/10.)
    x2 = np.arange(0.01, 1.0, 1/10.)
    x3 = np.arange(0.1, 0.2, 1/100)
    
    f_obj = lambdify(symbols_list, obj, "numpy")
    obj_vals= np.zeros(len(x1)*len(x2)*len(x3))
    #Calcular todos los valores de un rayo en un bucle triple y sumarlos
    for idx_1, r_i in enumerate(x1):
        for idx_2, x_i in enumerate(x2):
            for idx_3, s_i in enumerate(x3):
                idx = idx_1 * len(x2)*len(x3) + idx_2 * len(x3) + idx_3
                for t_i in x0:
                    obj_vals[idx] += f_obj(r_i, t_i, x_i, s_i) 
                obj_vals[idx] = step * obj_vals[idx] # MC integration
    
    return obj_vals

def MC_test(N, GT, a=1.5, b=11.5):
    MC_val = MC_integration(a, b, N)

    
    return np.mean(abs(GT - MC_val))

def GP_test(GT, best_ind, a=1.5, b=11.45):
    """Obtengo la integral entre a y b para cada rayo definido por un trío de valores r, s, x"""
    
    """np.seterr(all='raise')
    aprox = best_ind
    aprox_a = N(aprox.subs(t,a), 3)
    aprox_b = N(aprox.subs(t,b), 3)"""
    
    #best_ind = diff(best_ind, t)
    
    x1 = np.arange(1.5, 2.5, 1/10.)
    x2 = np.arange(0.01, 1.0, 1/10.)
    x3 = np.arange(0.1, 0.2, 1/100)
    
    #f_a = lambdify([t,r,s,x], aprox_a, "numpy")
    #f_b = lambdify([t,r,s,x], aprox_b, "numpy")
    
    f_a = lambdify([t,r,s,x], best_ind, "numpy")
    f_b = lambdify([t,r,s,x], best_ind, "numpy")
    vals= np.zeros(len(x1)*len(x2)*len(x3)) # vals es un array con longitud n = numero de rayos
    #print(vals.shape)
    for idx_1, r_i in enumerate(x1):
        for idx_2, x_i in enumerate(x2):
            for idx_3, s_i in enumerate(x3):
                idx = idx_1 * len(x2)*len(x3) + idx_2 * len(x3) + idx_3
                #print(str(np.around(r_i, 3))+" "+str(np.around(s_i, 3))+" "+str(np.around(x_i, 3)))
                #print(str(f_b(np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3)))+\
                #    " - "+str(f_a(np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3))))
                #print(str(f_b(b, np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3)))+\
                #     " - "+str(f_a(a, np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3))))
                    
                #vals[idx] = f_b(np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3)) \
                #    - f_a(np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3))
                vals[idx] = f_b(b, np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3)) \
                    - f_a(a, np.around(r_i, 3), np.around(s_i, 3), np.around(x_i, 3))
                
    
    err = abs(GT - vals) #np.array size n      
    #print(err)
    
    return np.mean(err)

    

            

    


if __name__ == "__main__":
    """expr1 = gen_branch(3, 2)
    plot_tree(expr1, 0)
    expr2 = mute_3(expr1)
    plot_tree(expr2, 1)"""
    
    print("main")
    #Generamos la poblacion inicial
    pop, objective = gen_pop(n_ind)
    objective = objective_mult()
    best_scores = []
    
    #variables for time profiling
    t_total = 0
    t_init = 0
    t_eval = 0
    t_cross = 0
    t_mut = 0
    t_sel = 0
    
    #samples generation
    x0 = np.arange(1.5, 11.5, 1/10.) ## t
    x1 = np.arange(1.5, 2.5, 1/10.) ## r
    x2 = np.arange(0.01, 1, 1/10.) ## x
    x3 = np.arange(0.1, 0.2, 1/100) ## s
    points = np.meshgrid(x0, x1, x2, x3)
    #points = [x0, x1, x2, x3]
    
    #objective function evaluation
    f_obj = lambdify(symbols_list, objective, "numpy")
    obj_vals= f_obj(points[0], points[1], points[2], points[3]).flatten() 
    n_samps = len(obj_vals)
    
    #aux variables
    scores_dict = {}
    ev_rate = tmt_rates
    best_fit = 0.2
    iteration = 0
    errors = []
    m1 = 0.0
    finished = False
    initial_fit = 0
    
    time_file = open("time_file.txt", "w")
    conv_file = open("conv_file.txt", "w")
    
    #while best_fit > 0.1:
    while not(finished):
        t_2 = time.time()
    #for iteration in range(n_iter):
        
        
        #evualte population
        scores = np.zeros(len(pop))
        print(len(pop))
        reps = 0
        for i in range(len(pop)):
            scores[i] = fitness_diff(pop[i], obj_vals, points, n_samps)
            """if not (e_idxs[-1] == 0):
                print(e_idxs[-1])"""
        t_3 = time.time()
        
        #print information
        sorted_scores = np.sort(np.abs(scores))
        sorted_idxs = np.argsort(np.abs(scores))
        best_exp = pop[sorted_idxs[0]]
        if (iteration % 5 == 0):
            best_scores.append(sorted_scores[0])
            print(iteration)
            for j in range(3): 
                print(pop[sorted_idxs[j]])
                print(sorted_scores[j])
        best_fit = sorted_scores[0]
        errors.append(best_fit)
        
        
        ## Stoping criteria
        if iteration == 0:
            initial_fit = best_fit
        if (best_fit / initial_fit) < 0.1 or iteration == 250 :
            finished = True
        
        #Minimum detection
        """if (iteration % 10 == 0):
            e_array = np.array(errors)
            m0 = m1
            m1 = np.mean(e_array)
            if local_minimum(mnm_th, m0, m1):
                ev_rate = tmt_rates_mnm
                print('MINIMUM REACHED')
            else:
                ev_rate = tmt_rates
                print('OUT OF MINIMUM')
            errors.clear()
            scores_dict.clear()"""
        
        t_5 = time.time()
        
        #Evolution:
        next_pop = [evolution(pop, scores, ev_rate, i, sorted_idxs, int(tmt_size)) for i in range(n_ind)]
        #print(len(next_pop))
        ##t_7 = time.time()
                
        #Update the population
        pop = next_pop.copy()
        pop = set(pop)
        pop = list(pop)
        
        t_7 = time.time()

        # Times computation and update
        t_gen = t_7 - t_2;
        print(t_gen)
        #t_eval = (t_3 - t_2) + t_eval
        #t_sel = (t_4 - t_3) + t_sel
        #t_cross = (t_6 - t_5) + t_cross
        #t_mut = (t_7 - t_6) + t_mut
        # = t_eval + t_sel + t_cross + t_mut
        """print(t_eval, t_sel, t_cross, t_mut, t_total)
        print('Eval time: ' + str(t_eval/t_total))
        print('Select time: ' + str(t_sel/t_total))
        print('Cross time: ' + str(t_cross/t_total))
        print('Mut time: ' + str(t_mut/t_total))"""
        #print(reps)
        iteration += 1
        conv = best_fit/initial_fit
        
        
        L_time = [str(iteration), " ", str(t_gen), "\n"]
        time_file.writelines(L_time)
        L_conv = [str(iteration), " ", str(conv), "\n"]
        conv_file.writelines(L_conv)
        
       

    #print("Poblacion final:")
    #print(pop)
    #print(scores)
    time_file.close()
    conv_file.close()
    print(best_exp)
    GT_int = MC_integration(1000)
    for i in range(6):
        n = 2**(i+1)
        print("MC error for "+ str(n) +" samples:" + str(MC_test(n, GT_int)))
    print("Genetic Programming error:" + str(GP_test(GT_int, best_exp)))
    #plt.plot(best_scores[1:])
