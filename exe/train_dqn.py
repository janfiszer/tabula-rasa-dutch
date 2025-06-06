import os
import numpy as np
import torch

from src import rlcard_gpt_gen
from src.rlcard_gpt_gen import config
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):
    # Check if CUDA is available
    device = get_device()
    
    # Seed numpy, torch, and random
    set_seed(args['seed'])

    # Make the environment with seed
    env = rlcard.make(
        'cambio',
        config={
            'seed': args['seed'],
        }
    )

    # Initialize the DQN agent and opponents
    agent = DQNAgent(
        num_actions=config.NUM_ACTIONS,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device,
    )
    
    # Create random agents for other players
    opponent_agents = [DQNAgent(
        num_actions=config.NUM_ACTIONS,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device,
    ) for _ in range(2)]  # 2 opponents for 3 total players
    
    # Set the agents in the environment
    env.set_agents([agent] + opponent_agents)
    
    # Mean rewards for logging
    rewards = []
    
    # Create a Logger instance
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = Logger(log_dir='logs/cambio_dqn')

    # Start training
    for episode in range(args['num_episodes']):
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        
        # Reorganize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        
        # Feed transitions into agent memory, and train the agent
        # Here, we assume that training on a single episode is enough
        for ts in trajectories[0]:
            agent.feed(ts)
            
        # Evaluate the performance
        if episode % args['evaluate_every'] == 0:            # Set up evaluation environment
            eval_env = rlcard.make('cambio', config={'seed': args['seed']})
            eval_env.set_agents([agent] + opponent_agents)
            eval_rewards = tournament(eval_env, args['num_eval_games'])
            # tournament returns average payoffs, so we can directly use the first player's average
            rewards.append(eval_rewards[0])
            
            # Add point to logger
            logger.log_point(x=episode, y=np.mean(rewards[-100:]), z=agent.total_loss)
            
            # Print out results
            print(f'\nEpisode {episode}: Average reward is {np.mean(rewards[-100:])}')
            
            # Save model
            if args['save_path'] and np.mean(rewards[-100:]) > args['save_threshold']:
                agent.save(args['save_path'])
                
    # Plot rewards
    plot_curve(rewards, args['figure_path'], 'DQN on Cambio')

    # Save final model
    if args['save_path']:
        agent.save(args['save_path'])

def evaluate(agent, eval_env, num_games=100):
    rewards = []
    for _ in range(num_games):
        _, payoffs = eval_env.run(is_training=False)
        rewards.append(payoffs[0])  # Get player 1's payoff
    return np.mean(rewards)

if __name__ == '__main__':
    # Set the arguments
    args = {
        'seed': 42,
        'num_episodes': 5000,
        'num_eval_games': 100,
        'evaluate_every': 100,
        'save_path': 'models/cambio_dqn',
        'figure_path': 'figures/cambio_dqn.png',
        'save_threshold': 0.1,  # Only save if mean reward over last 100 episodes exceeds this
    }

    # Create directories if not exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Train the agent
    train(args)
