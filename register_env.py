import gym
from rl_car_env import RlCarEnv

gym.register(
    id='RlCar-v0',
    entry_point='rl_car_env:RlCarEnv',
)