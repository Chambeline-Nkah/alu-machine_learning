#!/usr/bin/env python3
"""Performing SARSA(λ)"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, 
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    SARSA(λ) algorithm
    
    env is the openAI environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    
    Returns: Q, the updated Q table
    """
    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(Q.shape[1])
        return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy_policy(state, epsilon)
        eligibility_trace = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state, epsilon)

            td_error = reward + gamma * Q[next_state, next_action] * (not done) - Q[state, action]

            eligibility_trace[state, action] += 1

            Q += alpha * td_error * eligibility_trace

            eligibility_trace *= gamma * lambtha

            state, action = next_state, next_action
            if done:
                break

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q
