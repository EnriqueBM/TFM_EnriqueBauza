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


######## GLOBAL VARIABLES ###############
x, y, z = symbols('x y z')
scale = 4.0
branch_list = [Mul, Add, sin]
symbols_list = [x]
node_mutation_prob = 0.1



def objective_Li():
    x = symbols('x')
    xl = Point(2.5, 2.5, 2.5)
    xr = Point(0, 0, x)
    d = xr.distance(xl)
    Li = 1000*(exp(-0.1*x)*exp(-0.1*d)) / (d**2 * 4*pi)
    
    return Li

def gen_expr (func_list, max_args):
    func = random.choice(branch_list)
    if arity(func) == 1:
        n_args = 1
    
    
    return function, n_args

def gen_leaf():
    if random.uniform(0, 1) < 0.5:
        return sympify(scale * random.uniform(-1, 1))
    else:
        return random.choice(symbols_list)

    
    
def gen_branch(depth, n_args):
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

    if depth == 1:
        largs = [gen_leaf() for auxi in range(n_args)]
        return random.choice(branch_list)(*largs, evaluate=False)
    else:
        largs = [gen_branch(depth-1, n_args) for auxi in range(n_args)]
        return random.choice(branch_list)(*largs, evaluate=False)
    
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
    if arity(func) != None:
        n_args = arity(func)
    else:
        n_args = max_args
    if depth == 1:
        largs = [gen_leaf() for auxi in range(n_args)]
        return func(*largs, evaluate=False)
    else:
        largs = [gen_branch_2(depth-1, max_args) for auxi in range(n_args)]
        return func(*largs, evaluate=False)
    
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
    if len(expr.args) != 0:
        idx = random.choice(range(len(expr.args)))
        if len(expr.args[idx].args) == 0: ## Node before leaf
            largs = list(expr.args)
            largs[idx] = gen_branch(1, 2)
            return expr.func(*largs, evaluate=False)
        else:
            largs = list(expr.args)
            largs[idx] = mute_1(expr.args[idx])
            return expr.func(*largs, evaluate=False)
    else:
        return gen_branch(2, 2)
            
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
    if len(expr.args) != 0:
        idx_1 = random.choice(range(len(expr.args)))
        i = 0
        while (len(expr.args[idx_1].args) == 0):
            if (i < 2 * len(expr.args)):
                idx_1 = random.choice(range(len(expr.args)))
                i += 1
            else:
                return expr.func(*expr.args, evaluate=False)
              
        idx_2 = random.choice(range(len(expr.args[idx_1].args)))
        if len(expr.args[idx_1].args[idx_2].args) == 0: ## 2 Nodes before leaf
            largs = list(expr.args)
            largs[idx_1] = expr.args[idx_1].args[idx_2]
            return expr.func(*largs, evaluate=False)
        else:
            largs = list(expr.args)
            largs[idx_1] = mute_2(expr.args[idx_1])
            return expr.func(*largs, evaluate=False)
    else:
        return expr

def mute_3(expr):
    """
    Point mutation. Replaces a node by other expression or a leaf.
    
    Parameters
    ----------
    expr : Sympy.expression
        Expression to mute.
    
    Returns
    -------
    TYPE Sympy.expression
        Muted expression.
    
    """
    if len(expr.args) != 0:
        idx = random.choice(range(len(expr.args)))
        if len(expr.args[idx].args) == 0: ## Node before leaf, replace the leaf
            largs = list(expr.args)
            largs[idx] = gen_leaf()
            return expr.func(*largs, evaluate=False)
        elif random.uniform(0, 1) < node_mutation_prob: ## Replace the node function
            largs = list(expr.args)
            #largs[idx_1] = mute_2(expr.args[idx_1])
            return random.choice(branch_list)(*largs, evaluate=False)
        else:
            largs = list(expr.args)
            largs[idx] = mute_3(expr.args[idx])
            return expr.func(*largs, evaluate=False)
    else:
        return gen_branch(2, 2)
            
    
def crossover(p1, p2):
    if (len(p1.args) != 0) and (len(p2.args) != 0):
        idx_1 = random.choice(range(len(p1.args)))
        idx_2 = random.choice(range(len(p2.args)))
        largs_1 = list(p1.args)
        largs_2 = list(p2.args)
        donned = largs_2[idx_2]
        largs_2[idx_2] = largs_1[idx_1]
        largs_1[idx_1] = donned
        ch1 = p1.func(*largs_1, evaluate=False)
        ch2 = p2.func(*largs_2, evaluate=False)
    elif (len(p1.args) != 0) and (len(p2.args) == 0):
        idx_1 = random.choice(range(len(p1.args)))
        largs_1 = list(p1.args)
        donned = p2
        ch2 = largs_1[idx_1]
        largs_1[idx_1] = donned
        ch1 = p1.func(*largs_1, evaluate=False)
    elif (len(p1.args) == 0) and (len(p2.args) != 0):
        #idx_1 = random.choice(range(len(p1.args)))
        idx_2 = random.choice(range(len(p2.args)))
        largs_2 = list(p2.args)
        ch1 = largs_2[idx_2]
        largs_2[idx_2] = p1
        ch2 = p2.func(*largs_2, evaluate=False)
    else:
        ch1 = p1
        ch2 = p2
    
    return ch1, ch2
def fitness_mse(ind, obj_ev, samps):
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
    f = lambdify(x, ind, "numpy")
    evals = f(samps)
    error = obj_ev - evals
    sq_error = np.square(error)
    
    mse = np.sum(sq_error) / n_samps
    
    return mse

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
        ind_i = gen_branch(3, 2)
        pop.append(ind_i)
    obj = gen_branch(4, 2)
        
    return pop, obj


    

            

    


if __name__ == "__main__":
    """expr1 = gen_branch(3, 2)
    plot_tree(expr1, 0)
    expr2 = mute_3(expr1)
    plot_tree(expr2, 1)"""
    
    
    #Generamos la poblacion inicial
    n_ind = 40
    pop, objective = gen_pop(n_ind)
    #objective = objective_Li()
    
    
    n_iter = 30000
    samps = gen_samples(1000, 0.01, 1)
    f_obj = lambdify(x, objective, "math")
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
            scores[i]= fitness_mse(pop[i], obj_vals, samps)
                
        #Selection
        next_pop, sorted_scores = selection(pop, scores, 0.8)
        n_parents = int(len(pop) / 5)
        parents = next_pop[:n_parents]
        if (iter % 100 == 0):
            for j in range(3): 
                print(next_pop[j])
                print(sorted_scores[j])
            f = lambdify(x, next_pop[0], "numpy")
            evals = f(samps)
            #print(type(evals))
            ymax = np.amax(evals)
            ymin = np.amin(evals)
            best_ex = next_pop[0]
            p = plot(best_ex, objective, show=False)    
            p[0].line_color= 'blue'
            p[1].line_color= 'red'
            p.xlim=(-5,5)
            p.ylim=(ymin - 5, ymax + 5)
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

            if random.uniform(0, 1) < 0.1: #mutes the exponent of the functions
                next_pop[i] = mute_3(next_pop[i])
            elif random.uniform(0, 1) < 0.1: #mutes the coefficients of the monom
                next_pop[i] = mute_2(next_pop[i])
            elif random.uniform(0, 1) < 0.1: #changes one function of the monom
                next_pop[i] = mute_1(next_pop[i])            
        #Update the population
        pop = next_pop.copy()
    for i in range(len(pop)):
        scores[i]= fitness_complex(pop[i], obj_vals, samps)
    print("Poblacion final:")
    print(pop)
    print(scores)
