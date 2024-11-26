#!/usr/bin/env python3
"""Implementing a full training"""


import numpy as np

def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix"""
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    act_prob = exp_z / np.sum(exp_z)
    action = np.random.choice(len(act_prob), p=act_prob)
    return action

def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix."""
    action = policy(state, weight)
    z = np.dot(state, weight)
    exp_z = np.exp(z - np.max(z))
    act_prob = exp_z / np.sum(exp_z)
    gradient = np.outer(state, act_prob)
    gradient[action, :] -= state
    return action, gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Function that implements a full training"""
    scores = []
    for ep in range(1, nb_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        ep_rewards = []
        ep_states = []
        ep_actions = []
        ep_gradients = []

        while not done:
            action = policy(state, env.weights)
            next_state, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            ep_states.append(state)
            ep_actions.append(action)
            total_reward += reward
            state = next_state

        returns = np.zeros_like(ep_rewards)
        discounted_sum = 0
        for t in reversed(range(len(ep_rewards))):
            discounted_sum = ep_rewards[t] + gamma * discounted_sum
            returns[t] = discounted_sum

        for t in range(len(ep_rewards)):
            action = ep_actions[t]
            state = ep_states[t]
            grad = policy_gradient(state, env.weights)[1]
            grad = grad[action, :]
            
            env.weights += alpha * grad * (returns[t] - np.mean(returns))

        scores.append(total_reward)


    return scores
