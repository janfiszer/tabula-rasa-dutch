in: https://arxiv.org/pdf/1910.04376
page: 6

## State Representation
State is defined as all the information
that can be observed from the view of one player in a specific
timestep of the game. In the toolkit, each state is a dictio-
nary consisting of two values. The first value is a list of legal
actions. The second value is observation. There are various
ways to encode the observation. For Blackjack, we directly
use the player’s score and the dealer’s score as a representa-
tion. For other games in the toolkit, we encode the observed
cards into several card planes. For example, in Dou Dizhu,
the input of the policy is a matrix of 6 card planes, including
the current hand, the union of the other two players’ hands,
the recent three actions, and the union of all the played cards.

## Conclusions:
More to think about but, use plates to for state representation.

