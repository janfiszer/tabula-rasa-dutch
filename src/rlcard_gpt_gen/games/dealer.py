import random

class CambioDealer:
    def __init__(self):
        self.deck = self._generate_deck()
        self.discard_pile = []

    def _generate_deck(self):
        deck = [i for i in range(1, 14)] * 4  # 1=Ace, 11-13 = J/Q/K  TODO: add jokers?
        deck += [0] * 2  # 2 Jokers
        return deck

    def shuffle(self):
        random.shuffle(self.deck)

    def draw_card(self):
        return self.deck.pop() if self.deck else None

    def deal_four(self):
        return [self.draw_card() for _ in range(4)]

    def deck_is_empty(self):
        return self.deck == []