import numpy as np
import matplotlib.pyplot as plt

def decay_epsilon(min_epsilon, max_epsilon, decay_rate, num_episodes):
    episodes = np.arange(1, num_episodes + 1)
    epsilon_values = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episodes)
    return episodes, epsilon_values

def plot_epsilon_decay(min_epsilon, max_epsilon, decay_rate, num_episodes):
    episodes, epsilon_values = decay_epsilon(min_epsilon, max_epsilon, decay_rate, num_episodes)

    plt.plot(episodes, epsilon_values, label='Epsilon Decay')
    plt.title('Epsilon Decay Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

# Example values
min_epsilon = 0.05
max_epsilon = 0.95
decay_rate = 0.005
num_episodes = 300
save_path = 'epsilon_decay_plot.png'

plot_epsilon_decay(min_epsilon, max_epsilon, decay_rate, num_episodes)
