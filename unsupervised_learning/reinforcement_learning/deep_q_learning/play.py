#!/usr/bin/env python3
"""Script that displays the game being played
"""


import gym
import tensorflow.keras as K
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory


AtariProcessor = __import__('train').AtariProcessor
create_CNN_model = __import__('train').create_CNN_model


def playing():
    """Displays a game"""
    env = gym.make('Breakout-v0')
    env.reset()
    nb_actions = env.action_space.n
    model, frames = create_CNN_model(nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=frames)
    processor = AtariProcessor()
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   processor=processor,
                   memory=memory)
    dqn.compile(K.optimizers.Adam(lr=0.00025),
                metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env,
             nb_episodes=10,
             visualize=True)


if __name__ == '__main__':
    playing()
