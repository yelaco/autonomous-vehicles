import warnings
import numpy as np
import gym
import os
import time
from dqn_keras import Agent
from utils import plotLearning
from gym import wrappers
import register_env as register_env
from rl_car_env import RlCarEnv

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

class Quit(Exception): pass

if __name__ == '__main__':
    env = gym.make("RlCar-v0") 
    lr = 0.05
    n_episode = 4000
    agent = Agent(gamma=0.9, epsilon=0.95, alpha=lr, input_dims=5,
                  n_actions=3, mem_size=n_episode, batch_size=64, epsilon_end=0.0)

    if os.path.exists('dqn_model.keras'):
        print("Loaded model")
        agent.load_model()
        
    scores = []
    eps_history = []

    try:
        for i in range(n_episode):
            terminated = False
            score = 0
            state = env.reset()
            for step in range(10000):
                action = agent.choose_action(state)
                new_state, reward, terminated, _, _ = env.step(action)
                score += reward
                agent.remember(state, action, reward, new_state, int(terminated))
                state = new_state
                agent.learn()
                
                quit = env.render()
                if quit:
                    raise Quit
                    
                if terminated:
                    break
                
                agent.epsilon = min(1 / (lr * (i+1) * max(reward, 0.1)), 0.95)


            eps_history.append(agent.epsilon)
            scores.append(score)

            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            print('episode: ', i,'score: %.2f' % score,
                ' average score %.2f' % avg_score)

            if i % 10 == 0 and i > 0:
                env.change_map()
                agent.save_model()
    except Quit:
        agent.save_model()
    
    filename = 'obstacle_avoidance.png'

    x = [i+1 for i in range(len(eps_history))]
    plotLearning(x, scores, eps_history, filename)

    time.sleep(20)
    os.system('shutdown -h now')