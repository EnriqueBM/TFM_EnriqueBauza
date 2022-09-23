# -*- coding: utf-8 -*-
from sympy import *
import numpy as np


def pop_gen(n, conf_esp, ran):
    population = []
    for i in range(n):
        ind = np.random.uniform(-0.5, 0.5, len(conf_esp))
        ind = ind * ran
        population.append(ind)
    objective = np.random.uniform(-0.5, 0.5, len(conf_esp))
    objective = objective * ran
    
    return population, objective

def gen_samples(n_samples, inf, sup):
    samples = []
    step = (sup - inf) / n_samples
    for i in range(n_samples):
        sample = inf + step * i
        samples.append(sample)
    samples = np.array(samples)
    
    return samples
        

def sampling(n_samples, inf, sup, conf_esp):
    """
    Computes a matrix with the values of the functions in conf_esp evualted 
    n_samples times in the interval [inf, sup). Each row correspond to a function,
     each column to a point in the intervfal.

    Parameters
    ----------
    n_samples : int
        Number of points to evaluate the functions.
    inf : int
        Lower boundary of the interval in which the functions is evaluated.
    sup : int
        Upper boundary of the interval in which the functions is evaluated.
.
    conf_esp : Sympy Matrix
        Configuration espace with the functions.

    Returns
    -------
    evals : np.array(dim=2)
        Matrix with the n_samples values of the functions of the configuration espace 
        evaluated in the interval [inf, sup).

    """
    samp = gen_samples(n_samples, inf, sup)
    evals = []
    for i in range(len(conf_esp)):
        if (conf_esp[i] == 1):
            pts = np.ones(len(samp))
        else:
            f = lambdify(x, conf_esp[i], "numpy")
            pts = f(samp)
        evals.append(pts)
    evals = np.vstack(evals)
    
    return evals

def evaluation(ind, obj, ev_mat):
    dif = obj - ind
    values = dif @ ev_mat
    fitness = np.linalg.norm(values)
    
    return fitness

def crossover_1(parent1, parent2):
    """
    This simply computes the child as the mean of both parents.

    Parameters
    ----------
    parent1 : np.array(float)
    
    parent2 : np.array(float)

    Returns
    -------
    child : np.array(float)

    """
    child = (parent1 + parent2) / 2
    
    return child

def crossover_2(parent1, parent2):
    """
    As a classical bitstring crossover algorithm combines part of each parent to
    generate the children.

    Parameters
    ----------
    parent1 : np.array(float)

    parent2 : np.array(float)


    Returns
    -------
    child1 : np.array(float)
    
    child2 : np.array(float)


    """
    
    cut = np.random.randint(0, len(parent1)) 
    
    child1[0:cut] = parent1[0:cut]
    child1[cut:] = parent2[cut:]
    
    child2[0:cut] = parent2[0:cut]
    child2[cut:] = parent1[cut:]
    
    return child1, child2

def crossover_3(parent1, parent2):
    """

    Parameters
    ----------
    parent1 : np.array(float)

    parent2 : np.array(float)


    Returns
    -------
    child1 : np.array(float)
    
    child2 : np.array(float)


    """
    for i in range(len(parent1)):
        
        rand= np.random.uniform()
        if (rand > 0.5):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        
    return child1, child2

def fitness(ind, obj, mat):
    dif = obj - ind
    fit_vec = dif @ mat
    fit = np.linalg.norm(fit_vec)
    
    return fit
    
def mutation(ind, rate):
       prob =  np.random.uniform()
       if (prob < rate):
           mut = np.random.uniform(-0.5, 0.5, len(ind))
           mut = mut * 10.0
           ind = ind + mut
           
def selection(population, scores):
    n_pop = len(population)
    scores_ind = np.argsort(-scores)
    ord_scores = -np.sort(-scores)
    next_pop = population[scores_ind]
    
            
            
        
if __name__ == "__main__":
    ## Definimos nuestro espacio de configuraciones (x, sen(x), 1, exp(x)):
    x = symbols('x')
    e1 = x 
    e2 = sin(x)
    e3 = exp(x)
    e4 = 1
    e5 = x * sin(x)
    e6 = x * exp(x)
    e7 = x ** 2
    e8 = sin(x) * exp(x)
    e9 = sin(x) ** 2
    e10 = exp(x) ** 2
    base = Matrix([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10])
    ev_mat = sampling(50, -5, 5, base)
    
    #Generamos la poblacion inicial
    n_ind = 10
    scale = 100
    pop, objective = pop_gen(n_ind, base, scale)
    
    n_iter = 100
    
    for iter in range(n_iter):
        mut_rate = 0.9 - 0.02 * iter
        if (mut_rate < 0.05):
            mut_rate = 0.05
        #evualte pop to select parents
        scores = np.zeros(len(pop))
        for i in range(len(pop)):
            scores[i]= fitness(pop[i], objective, ev_mat)
            if (iter == n_iter-1):
                print(scores[i])
        
        
