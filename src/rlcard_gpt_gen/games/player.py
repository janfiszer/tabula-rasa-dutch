class CambioPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = [None] * 4
        self.known_cards = [False] * 4

    def receive_initial_cards(self, cards):
        self.hand = cards
        self.known_cards[0] = True
        self.known_cards[1] = True

    def swap_card(self, index, new_card):
        old_card = self.hand[index]
        self.hand[index] = new_card
        self.known_cards[index] = True

        return old_card

    def get_score(self):
        return sum(self._card_value(card) for card in self.hand)

    def _card_value(self, card):
        if card == 0: return 0  # Joker
        if card == 13: return -1  # Red King
        if card == 1: return 1  # Ace
        if card >= 11: return 10  # Face cards
        return card

    def get_obs(self):
        return [card if known else -1 for card, known in zip(self.hand, self.known_cards)]
