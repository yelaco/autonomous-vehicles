from gym.envs.registration import register

register(
    id='RlCarEnv-v0',
    entry_point='custom_env:RlCarEnv',
)