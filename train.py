import gym

env = gym.make('CustomEnv-v0')

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
