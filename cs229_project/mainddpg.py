from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
from pacingenv import PacingEnv
import matplotlib.pyplot as plt

import trainddpg
import buffer

env = PacingEnv()
# env = gym.make('Pendulum-v0')

MAX_EPISODES = 5
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = trainddpg.Trainer(S_DIM, A_DIM, ram)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print 'EPISODE :- ', _ep
    for r in range(MAX_STEPS):
        state = np.float32(observation)
        #print "shape of state is: " + str(state.shape)
        action = trainer.get_exploration_action(state)
        # if _ep%5 == 0:
        #   # validate every 5th episode
        #   action = trainer.get_exploitation_action(state)
        # else:
        #   # get action based on observation, use exploration policy here
        #   action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        #   continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            ram.add(state, action, reward, new_state)

        observation = new_observation

        # perform optimization
        trainer.optimize()
        if done:
            break

    # check memory consumption and clear memory
    gc.collect()
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)

    if _ep%100 == 0:
        print _ep
        #trainer.save_models(_ep)


print 'Completed episodes'


#test
total_reward = 0
actionrec = []
spendrec = []
timerec = []
env.dailyBudget = 100
state = env.reset()
for idx in range(288):
    #print "count : "+str(idx)
    state = Variable(torch.from_numpy(state.astype(float))).float()
    action = trainer.actor.forward(state)
    actionrec.append(action) #convert to [0,1] range for record
    #print "action : "+str(action)
    next_state, reward, done, _ = env.step(action)
    timerec.append(next_state[1])
    spendrec.append(next_state[0])
    state = next_state
    print "state : "+str(state)
    #print "time : "+str(nz[0][2])+'---'+'budget: '+str(nz[0][3])
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