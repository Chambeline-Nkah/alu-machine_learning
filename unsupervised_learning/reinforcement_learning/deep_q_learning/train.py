#!/usr/bin/env python3
"""Training an agent"""


import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Setting up the Breakout env
env = gym.make("Breakout-v0")
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)

env = FrameStack(env, num_stack=4)
nb_actions = env.action_space.n

# Deep Q-network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))  # Flattening the input
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Configuring the agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    policy=policy,
    enable_double_dqn=True,
    gamma=0.99,
)

# Compile
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])

# Training the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Saving the trained policy
model.save("policy.h5")
