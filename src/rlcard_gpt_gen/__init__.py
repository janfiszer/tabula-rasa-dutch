from rlcard.envs.registration import register

register(
    env_id='cambio',
    entry_point='src.rlcard_gpt_gen.envs.cambio:CambioEnv',
)
