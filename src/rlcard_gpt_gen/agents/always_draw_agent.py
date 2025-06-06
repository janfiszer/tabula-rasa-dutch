from src.rlcard_gpt_gen import config
import numpy as np


class AlwaysDrawAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, state):
        legal_actions = state['legal_actions']

        if "draw_deck" in legal_actions:
            return legal_actions.index("draw_deck")
        else:
            # Randomly select a legal action
            random_action = np.random.choice(legal_actions)
            return config.ACTIONS.index(random_action)

    def eval_step(self, state):
        return self.step(state), []