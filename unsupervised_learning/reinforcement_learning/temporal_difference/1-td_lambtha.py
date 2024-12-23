#!/usr/bin/env python3
"""Performing the TD(λ) algorithm"""


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    TD(λ) algorithm
    
    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate

    Returns: V, the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            
            td_error = reward + gamma * V[next_state] - V[state]
            
            eligibility_trace[state] += 1
            
            V += alpha * td_error * eligibility_trace
            
            eligibility_trace *= gamma * lambtha
            
            state = next_state
            if done:
                break

    return V
