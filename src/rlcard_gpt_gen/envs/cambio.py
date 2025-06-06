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

    def _extract_json_state(self, state):
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

    def _extract_state(self, state):
        """Convert state dictionary into a vector format suitable for DQN."""
        # Convert observation to fixed-size vector (52 positions for cards)
        obs_vector = np.zeros(52)
        for i, card in enumerate(state['obs']):
            if card >= 0:  # valid card
                obs_vector[card] = 1
                
        # Extract and vectorize public card information
        public_cards = state['public_cards']
        
        # Top card (1 position)
        top_card_vector = np.zeros(1)
        top_card = public_cards['top_card']
        if top_card is not None:
            top_card_vector[0] = top_card + 1  # +1 to differentiate from no card (0)
            
        # Discard pile (52 positions)
        discard_vector = np.zeros(52)
        for card in public_cards['discard_pile']:
            if card >= 0:
                discard_vector[card] = 1
                
        # Player discards (12 positions - 3 players * 4 cards)
        player_discards_vector = np.zeros(12)
        for player_idx, discards in enumerate(public_cards['player_discards'].values()):
            for card_idx, card in enumerate(discards[:4]):  # Limit to 4 cards per player
                if card >= 0:
                    player_discards_vector[player_idx * 4 + card_idx] = card + 1

        # Game state flags
        drawn_card_vector = np.array([state['drawn_card'] if state['drawn_card'] is not None else -1])
        draw_phase_vector = np.array([1 if state['draw_phase'] else 0])
        called_cambio_vector = np.array([1 if state['called_cambio'] else 0])
        current_player_vector = np.array([state['current_player']])
        
        # Concatenate all vectors into final state vector
        state_vector = np.concatenate([
            obs_vector,                # 52
            top_card_vector,          # 1
            discard_vector,           # 52
            player_discards_vector,   # 12
            drawn_card_vector,        # 1
            draw_phase_vector,        # 1
            called_cambio_vector,     # 1
            current_player_vector     # 1
        ])
        
        return state_vector

    def get_state(self, player_id):
        """Get vectorized state representation for DQN."""
        raw_state = self._extract_state(self.game.get_state(player_id))
        vectorized_state = self._state_to_vector(raw_state)
        return {'obs': vectorized_state, 'legal_actions': raw_state['legal_actions']}