import warnings
import numpy as np
import gym
from dqn_keras import Agent
from utils import plotLearning
from gym import wrappers
import register_env as register_env
from rl_car_env import RlCarEnv

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

class Quit(Exception): pass

try:
    if __name__ == '__main__':
        env = gym.make("RlCar-v0") 
        lr = 0.0005
        n_episode = 2000
        agent = Agent(gamma=0.9, epsilon=0.0, alpha=lr, input_dims=5,
                    n_actions=3, mem_size=n_episode, batch_size=64, epsilon_end=0.0)
        
        agent.load_model()

        scores = []

        for i in range(n_episode):
            terminated = False
            score = 0
            state = env.reset()
            for step in range(10000):
                action = agent.choose_action(state)
                new_state, reward, terminated, _, _ = env.step(action)
                score += reward
                state = new_state
                
                quit = env.render()
                if quit:
                    raise Quit
                
                if terminated:
                    break

            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            print('episode: ', i,'score: %.2f' % score,
                ' average score %.2f' % avg_score)
            
            if i % 10 == 0 and i > 0:
                env.change_map()
except Quit:
    pass