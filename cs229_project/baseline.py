#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:17:11 2018
"""

import math, random

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pacingenv import PacingEnv
from torch.autograd import Variable

dt = 5
now = 0L
index = 0
#generate simulated impressions
maxctr = 0.02
basecost = 10
baseIntensity = 2
"""
#initial state
hours = 24
imps = 0
clicks = 0
"""
dailyBudget = 100
ctrThres = 0
maxBid = 20 
cpcGoal = 100
env = PacingEnv(dt,dailyBudget,ctrThres,maxBid, cpcGoal, now,index,maxctr, basecost,baseIntensity)

class BaselineAlgo:
    def __init__(self):
        #short term sensitivity
        self.shortS = 1 #there could be better initial value but we'll make do with this   
        return

    def act(self,state, env, debugrec):
        #get states
        time = 288-env.state[1]
        budget = env.state[0]
        tod = (env.now/5)%288
        #cold start sensitivity range protection
        S = min(0.1,max(0.001,self.shortS)) if tod< 60 else self.shortS
        ref = budget/S/time
        intraday = budget/time
        debugrec.append((S,budget,time,intraday))
        res = min(1.0,max(0.00001,ref))
        
        #update sensitivity
        startTime = max(0,env.now - 3*12*env.dt)
        cost = 0
        pastpacing = 0.0
        for time in range(startTime,env.now,env.dt):
            cost = cost + env.cache[time][4]
            pastpacing = pastpacing+ env.cache[time][3]
        self.shortS = 1 if pastpacing==0 else cost/float(pastpacing)
        
        return res
    


BASELINE = True

baseline = BaselineAlgo()


#compare, naive bidding & pacing
total_reward = 0
actionrec = []
spendrec = []
timerec = []

env.dailyBudget = 100
state = env.reset()
print "data generated" + str(len(env.data))
debugrec = []
for idx in range(288):
    action = baseline.act(state, env,debugrec)
    actionrec.append(action)
    next_state, reward, done, _ = env.step(action)
    timerec.append(next_state[1])
    spendrec.append(next_state[0])
    state = next_state
    total_reward += reward

#plot pacing signals
y = actionrec
x = [i for i in range(len(actionrec))]
plt.plot(y)
plt.show()
#plot spendings

y = spendrec
x = timerec
plt.plot(y)
plt.show()


#plot debug signals
print "plot of S " + str(len(debugrec))

y = [debugrec[i][0] for i in range(len(debugrec))]
x = [i for i in range(len(debugrec))]
plt.plot(y)
plt.show()

print "plot of remaining budget"
y = [debugrec[i][1] for i in range(len(debugrec))]
x = [i for i in range(len(debugrec))]
plt.plot(y)
plt.show()

print "plot of ref signal"
y = [debugrec[i][3] for i in range(len(debugrec))]
x = [i for i in range(len(debugrec))]
plt.plot(y)
plt.show()

"""
print "plot of remaining budget ratio"
y = [debugrec[i][1] for i in range(len(debugrec))]
x = [i for i in range(len(debugrec))]
plt.plot(y)
plt.show()

print "plot of ref signal"
y = [debugrec[i][4] for i in range(len(debugrec))]
x = [i for i in range(len(debugrec))]
plt.plot(y)
plt.show()
"""