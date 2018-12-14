from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utilsddpg
import modelddpg

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

    def __init__(self, state_dim, action_dim, ram):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ram = ram
        self.iter = 0
        self.noise = utilsddpg.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = modelddpg.Actor(self.state_dim, self.action_dim)
        self.target_actor = modelddpg.Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        self.critic = modelddpg.Critic(self.state_dim, self.action_dim)
        self.target_critic = modelddpg.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

        utilsddpg.hard_update(self.target_actor, self.actor)
        utilsddpg.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        #print "state size: " + str(state.shape)
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + self.noise.sample()
        new_action = max(0.0,min(1.0,new_action))
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
        a1 = a1.reshape([a1.shape[0],1])
        #print "act: " + str(a1)
        #print "act size: " + str(a1.shape)
        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        #print "a2 size: " + str(a2.shape)
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA*next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        #print "preda1 size: " + str(a1.shape)
        loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utilsddpg.soft_update(self.target_actor, self.actor, TAU)
        utilsddpg.soft_update(self.target_critic, self.critic, TAU)

        # if self.iter % 100 == 0:
        #   print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
        #       ' Loss_critic :- ', loss_critic.data.numpy()
        # self.iter += 1

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print 'Models saved successfully'

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        utilsddpg.hard_update(self.target_actor, self.actor)
        utilsddpg.hard_update(self.target_critic, self.critic)
        print 'Models loaded succesfully'