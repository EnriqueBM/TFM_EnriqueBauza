#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:52:35 2022

@author: quique
"""
import matplotlib.pyplot as plt
import numpy as np
import math




in_file = open("conv_stats.txt", 'r')
values_pix = []
cnt = 0
for line in in_file:
    cnt += 1
    line = line[0:-1]
    line = line.split(" ")
    #print(line)
    for t in line:
            values_pix.append(float(t))
in_file.close()

values_pix = np.array(values_pix)

values_acum_pix = [sum(values_pix[0:auxi+1]) for auxi in range(len(values_pix))]
#values_acum_pix = np.array(values_acum_pix)/1000



plt.figure(0,dpi=300)
#plt.title(Title)
plt.xlabel("Generation")
plt.ylabel("Fitness")
#plt.yscale("log")

plt.plot(range(len(values_pix)),values_pix,color="red")    
plt.grid()
#plt.yticks(np.arange(0, max(values_acum_seq), 50)) 
#plt.legend()

plt.savefig("conv_ss_2dim.pdf",dpi=300)