import rlcard
from src import rlcard_gpt_gen
from src.rlcard_gpt_gen.agents.always_draw_agent import AlwaysDrawAgent
import sys


class HumanAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def step(self, state):
        print("\n===== Your Turn =====")

        if state['draw_phase']:
            self._print_game_state(state)
        else:
            # Player's hand
            print("\n=== Your Hand ===")
            print(f"Cards: {state['obs']}")
            print(f"Drawn card: {state['drawn_card']}")

        while True:
            print("\nAvailable actions:")
            for i, action in enumerate(rlcard_gpt_gen.config.ACTIONS):
                if action in state['legal_actions']:
                    print(f"{i}: {action}")
            
            action_index = int(input("\nEnter your action number: ").strip())
            action = rlcard_gpt_gen.config.ACTIONS[action_index]
            
            if action in state['legal_actions']:
                return action_index
            else:
                print("Invalid action. Try again.")

    def _print_game_state(self, state):
        """Print detailed game state information."""
        print("\n=== Game Information ===")
        print(f"Current phase: {'Draw Phase' if state['draw_phase'] else 'Action Phase'}")
        print(f"Cambio has been called: {state['called_cambio']}")
        
        # Player's hand
        print("\n=== Your Hand ===")
        print(f"Cards: {state['obs']}")
        if state['drawn_card'] is not None:
            print(f"Drawn card: {state['drawn_card']}")
        
        # Public information
        print("\n=== Public Information ===")
        public_cards = state['public_cards']
        
        print("Discard Pile:")
        if public_cards['top_card'] is not None:
            print(f"  Top card: {public_cards['top_card']}")
        print(f"  Full pile: {public_cards['discard_pile']}")
        
        # Player discards history
        print("\n=== Discarded Cards History ===")
        for player_id, discards in public_cards['player_discards'].items():
            if discards:  # Only show if player has discarded cards
                print(f"Player {player_id}: {discards}")
        
        print("\n=== Available Actions ===")
        print(f"Legal actions: {state['legal_actions']}")

    def eval_step(self, state):
        return self.step(state), []

def main():
    # Set up Cambio environment
    env = rlcard.make('cambio')

    # Set agents: Human vs Random
    human_agent = HumanAgent(num_actions=env.num_actions)
    random_agent = AlwaysDrawAgent(num_actions=env.num_actions)
    env.set_agents([human_agent, random_agent, random_agent])

    # Start game
    state, player_id = env.reset()
    
    while not env.is_over():
        if state['draw_phase']:
            print("\n" + "="*50)
            print(f"Player {player_id}'s turn")
        action = env.agents[player_id].step(state)
        state, player_id = env.step(action)
        
    # Show final results with more detail
    print("\n" + "="*50)
    print("\n===== Game Over =====")
    print("\nFinal Hands:")
    for i, player in enumerate(env.game.players):
        print(f"Player {i}: {player.hand} → Score: {player.get_score()}")
    print("\nFinal Payoffs:", env.get_payoffs())

if __name__ == '__main__':
    main()
