#!/usr/bin/env python3
"""Performing the Monte Carlo algorithm"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Monte Carlo algorithm

    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action to take
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate

    Returns: V, the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
