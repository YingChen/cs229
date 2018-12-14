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
        self.pacing = 1
        return

    def act(self,state, env, debugrec):
        #get states
        time = 288-env.state[1]-1
        budget = max(0,env.state[0])
        desired_ratio = budget/float(time)
        #print 'budget is: '+ str(budget)
        #cold start sensitivity range protection
        S = self.shortS
        adjust_multiplier = desired_ratio/float(S) if S!=0 else 10.0  #cap the multiplier at 10
        if S<1:
            print 'S is: '+ str(S) 
        #print 'multiplier is: '+ str(adjust_multiplier)
        debugrec.append((S,budget,time,adjust_multiplier))
        res = min(1.0,max(0.0,self.pacing*adjust_multiplier))
        self.pacing = res #record
        #print 'res is: '+ str(res)
        
        #update spending speed
        startTime = max(0,env.now - 12*env.dt) #1 hour past time
        cost = 0
        for time in range(startTime,env.now,env.dt):
            cost = cost + env.cache[time][4]
        self.shortS = max(0,cost/float(12))
        
        return res
    


BASELINE = True

baseline = BaselineAlgo()


#compare, naive bidding & pacing
total_reward = 0
actionrec = []
spendrec = []
timerec = []

state = env.reset()
env.dailyBudget = 40
debugrec = []
for idx in range(288):
    action = baseline.act(state, env,debugrec)
    #print "action generated" + str(action)
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
