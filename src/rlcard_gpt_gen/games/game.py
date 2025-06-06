import numpy as np

from src.rlcard_gpt_gen.games.dealer import CambioDealer
from src.rlcard_gpt_gen.games.player import CambioPlayer
from src.rlcard_gpt_gen import config


class CambioGame:
    def __init__(self, num_players=config.N_PLAYERS):
        self.num_players = num_players
        self.player_discards = {i: [] for i in range(self.num_players)}  # Track discards per player

        self.dealer = CambioDealer()
        
        # Game state variables

        # Cambio-related variables
        self.called_cambio = False
        self.turns_after_cambio = 0
        self.terminal = False

        self.current_player = 0
        self.public_deck = []  # Cards that are visible to all players

        # during-move-phase relevant
        self.drawn_card = None  # Store the currently drawn card
        self.draw_phase = True  # Track if we're in drawing phase

        self._set_up_game()

    def _set_up_game(self):
        self.players = [CambioPlayer(i) for i in range(self.num_players)]

        self.dealer.shuffle()
        for player in self.players:
            player.receive_initial_cards(self.dealer.deal_four())

        self.current_player = 0
        self.called_cambio = False
        self.turns_after_cambio = 0

    def init_game(self):
        self._set_up_game()
        return self.get_state(self.current_player), self.current_player
    
    def step(self, action):
        # First phase: player must choose between draw actions or calling cambio
        if self.draw_phase:
            if action == 'call_cambio':
                self.called_cambio = True
                self.turns_after_cambio = len(self.players) - 1
                self.current_player = self.next_player()
                self.draw_phase = True  # TODO: not sure why True gen : This might need to be False if you want to end the turn immediately after calling cambi
                return self.get_state(self.current_player), self.current_player
            
            elif action == 'draw_deck':
                self.drawn_card = self.dealer.draw_card()
                self.draw_phase = False
                return self.get_state(self.current_player), self.current_player
            
            elif action == 'draw_pile':
                if len(self.public_deck) > 0:
                    self.drawn_card = self.public_deck.pop()
                    self.draw_phase = False
                    return self.get_state(self.current_player), self.current_player
                
        # Second phase: player must choose what to do with drawn card
        else:
            if action.startswith('swap_'):
                idx = int(action[-1])
                old_card = self.players[self.current_player].swap_card(idx, self.drawn_card)
                self.public_deck.append(old_card)
                self.player_discards[self.current_player].append(old_card)
                self.drawn_card = None
            else:  # discard
                self.public_deck.append(self.drawn_card)
                self.player_discards[self.current_player].append(self.drawn_card)
                self.drawn_card = None
            
            self.draw_phase = True
            self.current_player = self.next_player()

            if self.called_cambio:
                self.turns_after_cambio -= 1
                if self.turns_after_cambio <= 0:
                    self.terminal = True

        return self.get_state(self.current_player), self.current_player

    def get_legal_actions(self):
        if self.draw_phase:
            actions = ['draw_deck']
            if len(self.public_deck) > 0:
                actions.append('draw_pile')
            if not self.called_cambio:
                actions.append('call_cambio')
            return actions
        else:
            # When player has drawn a card, they can swap with any position or discard
            actions = ['discard']
            actions.extend([f'swap_{i}' for i in range(4)])
            return actions

    def get_state(self, player_id):
        """Get game state from the perspective of the given player."""
        player = self.players[player_id]

        # Get known cards for each player
        # all_known_cards = []
        # for p in self.players:
        #     # For each player, get their observable cards
        #     known_cards = []
        #     for card, known in zip(p.get_hand(), p.get_known_cards()):
        #         if known:  # Only include cards that have been revealed
        #             known_cards.append(card)
        #         else:
        #             known_cards.append(-1)  # -1 represents unknown cards
        #     all_known_cards.append(known_cards)

        return {
            'obs': player.get_obs(),  # Current player's hand
            'legal_actions': self.get_legal_actions(),
            'public_cards': {
                'top_card': self.public_deck[-1] if self.public_deck else None,
                'discard_pile': self.public_deck.copy(),  # All cards in discard pile
                # 'all_known_cards': all_known_cards,  # Known cards for each player
                'player_discards': self.player_discards.copy(),  # History of each player's discards
            },
            'drawn_card': self.drawn_card,
            'draw_phase': self.draw_phase,
            'called_cambio': self.called_cambio,
            'current_player': self.current_player,
        }

    def get_num_actions(self):
        return len(self.get_legal_actions())

    def get_payoffs(self):
        scores = [p.get_score() for p in self.players]
        min_score = min(scores)
        winner_idx = scores.index(min_score)
        payoffs = [-1] * len(self.players)
        payoffs[winner_idx] = 1
        return np.array(payoffs)  # Return list of lists when game is over

    def next_player(self):
        return (self.current_player + 1) % len(self.players)

    def get_num_players(self):
        return len(self.players)

    def is_over(self):
        if self.terminal:
            return True
        if self.dealer.deck_is_empty():
            return True
        return False
    
    def get_player_id(self):
        return self.current_player