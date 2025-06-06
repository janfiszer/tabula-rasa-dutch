from src.rlcard_gpt_gen.games.game import CambioGame
from src.rlcard_gpt_gen import config as cambio_config

from rlcard.envs import Env
import numpy as np

class CambioEnv(Env):
    def __init__(self, config):
        self.name = 'cambio'
        self.game = CambioGame()
        self.action_list = cambio_config.ACTIONS
        
        # Define state shape: 
        # - obs_vector (max 4 cards * 13 values) = 52
        # - top_card (1 value) = 1
        # - discard_pile (max 52 cards) = 52
        # - player_discards (max 3 players * 4 cards) = 12
        # - drawn_card (1 value) = 1
        # - draw_phase (boolean) = 1
        # - called_cambio (boolean) = 1
        # - current_player (1 value) = 1
        self.state_shape = [120]
        
        super().__init__(config)

    def _extract_state(self, state):
        """Extract state information for the current player."""
        obs = state['obs']
        legal_actions = state['legal_actions']
        
        # Convert observation to numpy array
        obs_vector = np.array([x if x >= 0 else -1 for x in obs])
        
        # Extract public card information
        public_cards = state['public_cards']
        top_card = public_cards['top_card'] if public_cards['top_card'] is not None else -1
        discard_pile = public_cards['discard_pile']
        player_discards = public_cards['player_discards']
        
        return {
            'obs': obs_vector,
            'legal_actions': legal_actions,
            'public_cards': {
                'top_card': top_card,
                'discard_pile': discard_pile,
                'player_discards': player_discards
            },
            'drawn_card': state['drawn_card'],
            'draw_phase': state['draw_phase'],
            'called_cambio': state['called_cambio'],
            'current_player': state['current_player']
        }

    def _decode_action(self, action_id):
        return self.action_list[action_id]

    def _encode_action(self, action):
        return self.action_list.index(action)

    def _get_legal_actions(self):
        return self.game.get_legal_actions()

    def get_payoffs(self):
        return self.game.get_payoffs()