from src.rlcard_gpt_gen.games.game import CambioGame
from src.rlcard_gpt_gen import config as cambio_config
from collections import OrderedDict

from rlcard.envs import Env
import numpy as np

class CambioEnv(Env):
    def __init__(self, config):
        self.name = 'cambio'
        self.game = CambioGame()
        self.action_list = cambio_config.ACTIONS
        
        # State shape components:
        # - Player's hand (4 cards * 15 values) = 60 (14 card values + 1 for unknown)
        # - Discard pile (15 positions one-hot) = 15 (showing only top card)
        # - Per-player visible discards (15 values * 3 players) = 45
        # - Game state flags (draw_phase, called_cambio, current_player) = 3
        # Total: 123 dimensions
        self.state_shape = [[123] for _ in range(cambio_config.N_PLAYERS)]
        self.action_shape = [None for _ in range(cambio_config.N_PLAYERS)]
        
        super().__init__(config)

    def _extract_state(self, state):
        """Extract the state representation from state dictionary for agent"""
        extracted_state = {}
        
        # Convert legal actions to an OrderedDict as expected by DQN agent
        legal_actions = OrderedDict()
        for i, action in enumerate(self.action_list):
            if action in state['legal_actions']:
                legal_actions[i] = None
        extracted_state['legal_actions'] = legal_actions

        # Initialize observation vector
        obs = np.zeros(123)
        
        # Encode player's hand (first 60 positions = 4 cards * 15 possible values)
        # Value 14 represents unknown card, values 0-13 represent actual cards
        hand = state['obs']
        for card_idx, card in enumerate(hand):
            if card >= 0:  # known card
                obs[card_idx * 15 + card] = 1
            else:  # unknown card
                obs[card_idx * 15 + 14] = 1
        
        # Encode top card of discard pile (next 15 positions)
        public_cards = state['public_cards']
        top_card = public_cards['top_card']
        if top_card is not None and top_card >= 0:
            obs[60 + top_card] = 1
        
        # Encode per-player visible discards (next 45 positions = 15 * 3 players)
        for player_idx, discards in enumerate(public_cards['player_discards'].values()):
            if discards:  # Only encode the most recent discard for each player
                last_discard = discards[-1]
                if last_discard >= 0:
                    obs[75 + player_idx * 15 + last_discard] = 1
        
        # Encode game state flags (last 3 positions)
        offset = 120
        obs[offset] = 1 if state['draw_phase'] else 0
        obs[offset + 1] = 1 if state['called_cambio'] else 0
        obs[offset + 2] = state['current_player'] / cambio_config.N_PLAYERS  # Normalize player ID
        
        extracted_state['obs'] = obs
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = state['legal_actions']
        
        return extracted_state

    def _decode_action(self, action_id):
        return self.action_list[action_id]

    def _encode_action(self, action):
        return self.action_list.index(action)

    def _get_legal_actions(self):
        return self.game.get_legal_actions()

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_state(self, player_id):
        """Get state representation for current player"""
        return self._extract_state(self.game.get_state(player_id))

    def run(self, is_training=False):
        """Run a complete game and get trajectories and payoffs.
        Overriding the default run method to handle payoffs format differently for training vs evaluation.
        """
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop until the game is over
        while not self.is_over():
            # Save state
            trajectories[player_id].append(state)

            # Agent plays
            action = self.agents[player_id].step(state)

            # Save action
            trajectories[player_id].append(action)

            # Environment step
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            state = next_state
            player_id = next_player_id

        # Save final state
        trajectories[player_id].append(state)

        # Get payoffs
        payoffs = self.get_payoffs()
        
        return trajectories, payoffs