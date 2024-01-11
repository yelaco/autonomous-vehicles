import numpy as np
import sys
import random
import gym
import warnings
import pickle
import register_env as register_env
from rl_car_env import RlCarEnv

actions = ["straight", "left", "right"]

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space + [action_space])) 
    return Qtable

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        action = np.argmax(Qtable[tuple(state)]) # exploit
    else:
        action = env.action_space.sample() # explore
    return action

def greedy_policy(Qtable, state):
    status = Qtable[tuple(state)]
    if all([s == 0 for s in status]):
        action = env.action_space.sample()
    else:
        action = np.argmax(Qtable[tuple(state)])
    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    state_changed = False
    total_crash = 0
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        #Reset the environment
        state = env.reset()
        terminated = False
        last_action = epsilon_greedy_policy(Qtable, state, epsilon)

        # repeat
        for step in range(max_steps):
            if state_changed: 
                action = epsilon_greedy_policy(Qtable, state, epsilon)
                last_action = action
            else:
                action = last_action
            new_state, reward, terminated, _, _ = env.step(action) # Terminated, Truncated, Info are not needed
            quit = env.render(f"Episode: {episode}        Step: {step}\nLZ: {state[1]}    RZ: {state[0]}    Reward: {'{:.2f}'.format(reward)}\nOL: {state[5]}    IL: {state[4]}    IR: {state[3]}    OR: {state[2]}\nAction:  {actions[action]}\nCrash: {total_crash}\n\nMode: Training\nMax episodes: {n_eval_episodes}\nMax steps: {max_steps}\nLearning rate: {learning_rate}\nGamma: {gamma}\nEpsilon: {'{:.2f}'.format(epsilon)}")
            if quit: return Qtable

            # custom indexing for state and action 
            state_action = tuple(np.append(state, action))
            new_state_action = tuple(np.append(new_state, action))

            Qtable[state_action] = Qtable[state_action] + learning_rate * (reward + gamma * np.max(Qtable[new_state_action]) - Qtable[state_action])
            
            if terminated:
                total_crash += 1
                break
            
            if any(new_state != state):
                state = new_state
                state_changed = True
            else:
                state_changed = False
                 
    env.close()
    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q):
    episode_rewards = []
    mean_reward = 0
    total_crash = 0
    for episode in range(n_eval_episodes):
        state = env.reset()
        total_rewards_ep = 0
    
        for step in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, _, info = env.step(action)
            total_rewards_ep += reward
            
            quit = env.render(f"Episode: {episode}        Step: {step}\nLZ: {state[0]}    RX: {state[1]}    Reward: {'{:.2f}'.format(reward)}\nOL: {state[5]}    IL: {state[4]}    IR: {state[3]}    OR: {state[2]}\nAction:  {actions[action]}\nCrash: {total_crash}\n\nMode: Evaluate\nMax episodes: {n_eval_episodes}\nMax steps: {max_steps}\nTotal reward: {'{:.2f}'.format(total_rewards_ep)}\nMean reward: {'{:.2f}'.format(mean_reward)}")
            if quit: return np.mean(episode_rewards),  np.std(episode_rewards)
            if terminated:
                total_crash += 1
                break
            
            state = new_state
        
        episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
    env.close()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

env = gym.make("RlCar-v0")

print("Observation Space:", env.observation_space)
print("Sample observation:", env.observation_space.sample())

print("Action Space Shape:", env.action_space.n)
print("Action Space Sample:", env.action_space.sample())

state_space = env.state_space
action_space = env.action_space.n

# Training parameters
n_training_episodes = 100
learning_rate = 0.5

# Evaluation parameters
n_eval_episodes = 100      

# Environment parameters
env_id = "RlCar-v0"   
max_steps = 800
gamma = 0.9               
eval_seed = []             

# Exploration parameters
max_epsilon = 0.95           
min_epsilon = 0.05           
decay_rate = 0.05

if len(sys.argv) == 1:
    print("**** Error ****[!]\nRun \'python3 q_learning.py train\' \nor \'python3 q_learning.py evaluate\'")
    sys.exit()
proc = sys.argv[1]

if proc == "train":
    try:    
        with open('q_table.pkl', 'rb') as f:
            Qtable_rlcar = pickle.load(f) 
    except FileNotFoundError:
        print("[!] No existing q-table found. Initializing a new one")
        Qtable_rlcar = initialize_q_table(state_space, action_space)
    
    # Start training
    Qtable_rlcar = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_rlcar)
    print(Qtable_rlcar)

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(Qtable_rlcar, f)
elif proc == "evaluate":
    with open('q_table.pkl', 'rb') as f:
        Qtable_rlcar = pickle.load(f)

    # Evaluate our Agent
    mean_reward, std_reward = evaluate_agent(env, max_steps + 400, n_eval_episodes, Qtable_rlcar)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
elif proc == "check":
    with open('q_table.pkl', 'rb') as f:
        Qtable_rlcar = pickle.load(f)
    
    state_space = env.state_space
    action_space = env.action_space.n
    
    # Print the Q-table in the specified format
    header = ["Straight", "Left", "Right", "RZ", "LZ", "ORS", "IRS", "ILS", "OLS"]
    header_str = "{:^8}" * env.action_space.n + " | " + "{:^4}" * len(state_space)
    print(header_str.format(*header))

    for i in range(state_space[0]):
        for j in range(state_space[1]):
            for k in range(state_space[2]):
                for l in range(state_space[3]):
                    for m in range(state_space[4]):
                        for n in range(state_space[5]):
                            row_values = Qtable_rlcar[i, j, k, l, m, n, :]
                            state_values = [i, j, k, l, m, n]
                            row_str = "{:^8.2f}" * len(row_values) + " | " + "{:^4}" * len(state_values)
                            print(row_str.format(*row_values, *state_values))
    
else:
    print("**** Error ****[!]\nRun \'python3 q_learning.py train\' \nor \'python3 q_learning.py evaluate\'")

