import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
# BUG 1 FIX: Use correct class name ScarceGridEnv (not ScarseGridEnv)
from src.environment import ScarceGridEnv
# BUG 2 FIX: Import QLearningAgent (now exists in agents.py)
from src.agents import QLearningAgent, MultiAgentQLearning
from src.game_theory import GameTheoryAnalyzer, RewardShapingStrategy
from src.utils import (
    Logger, 
    MetricsTracker, 
    ConfigManager, 
    RandomSeedManager, 
    visualize_grid
)

class ScarceGridTrainer:
    def __init__(self, config=None):
        """
        Initialize training environment and configuration
        
        Args:
            config (dict, optional): Custom configuration
        """
        # Initialize logging and configuration
        self.logger = Logger()
        self.config_manager = ConfigManager()
        self.config = config or self.config_manager.config
        
        # Set random seed for reproducibility
        RandomSeedManager.set_seed(
            self.config['training'].get('seed', 42)
        )
        
        # Initialize environment
        self.env = ScarceGridEnv(
            grid_size=self.config['environment']['grid_size']
        )
        
        # Initialize metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Game Theory Components
        self.game_theory_analyzer = GameTheoryAnalyzer()
        self.reward_shaper = RewardShapingStrategy({
            'base_reward': 1.0,
            'cooperation_bonus': 1.5
        })
        
        # Multi-Agent Learning Setup
        self.multi_agent_learning = MultiAgentQLearning(
            env=self.env, 
            num_agents=self.config['training'].get('num_agents', 2)
        )
    
    def train(self):
        """
        Main training loop for multi-agent reinforcement learning
        """
        try:
            # Training configuration
            num_episodes = self.config['training']['episodes']
            max_steps = self.config['training']['max_steps_per_episode']
            
            # Logging training start
            self.logger.info(f"Starting training for {num_episodes} episodes")
            
            # Experiment tracking
            experiment_results = {
                'total_rewards': [],
                'episode_lengths': [],
                'game_theory_metrics': []
            }
            
            # Main training loop
            for episode in range(num_episodes):
                # BUG 4 FIX: Properly unpack reset() return value
                state, _ = self.env.reset()
                
                # Episode tracking
                episode_reward = 0
                done = False
                steps = 0
                
                while not done and steps < max_steps:
                    # BUG 18 FIX: Both agents see the same grid state
                    # BUG 3 FIX: Use choose_action() method (exists on QLearningAgent)
                    actions = [
                        agent.choose_action(state) 
                        for agent in self.multi_agent_learning.agents
                    ]
                    
                    # Environment step
                    next_state, rewards, done, _, info = self.env.step(actions)
                    
                    # Reward shaping
                    shaped_rewards = [
                        self.reward_shaper.shape_reward([state, state], reward) 
                        for reward in rewards
                    ]
                    
                    # BUG 3 FIX: Use update() method (exists on QLearningAgent)
                    # BUG 18 FIX: Both agents see the same grid state
                    for i, agent in enumerate(self.multi_agent_learning.agents):
                        agent.update(
                            state, 
                            actions[i], 
                            shaped_rewards[i], 
                            next_state, 
                            done
                        )
                    
                    # Update states and rewards
                    state = next_state
                    episode_reward += sum(rewards)
                    steps += 1
                
                # Decay exploration
                for agent in self.multi_agent_learning.agents:
                    agent.decay_exploration()
                
                # Metrics tracking
                self.metrics_tracker.update(
                    reward=episode_reward, 
                    exploration_rate=self.multi_agent_learning.agents[0].get_exploration_rate(),
                    episode_length=steps
                )
                
                # Experiment result logging
                experiment_results['total_rewards'].append(episode_reward)
                experiment_results['episode_lengths'].append(steps)
                
                # Periodic logging and visualization
                if episode % 50 == 0:
                    self.logger.info(
                        f"Episode {episode}: "
                        f"Reward = {episode_reward}, "
                        f"Steps = {steps}"
                    )
            
            # Final analysis and logging
            self._analyze_results(experiment_results)
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _analyze_results(self, experiment_results):
        """
        Comprehensive results analysis
        
        Args:
            experiment_results (dict): Training experiment results
        """
        # Summary statistics
        avg_reward = np.mean(experiment_results['total_rewards'])
        avg_episode_length = np.mean(experiment_results['episode_lengths'])
        
        # Game theory analysis
        strategies = ['cooperative', 'competitive', 'mixed']
        payoff_matrix = self.game_theory_analyzer.create_payoff_matrix(strategies)
        nash_equilibria = self.game_theory_analyzer.nash_equilibrium(payoff_matrix)
        
        # Logging results
        self.logger.info("\n--- Training Summary ---")
        self.logger.info(f"Average Episode Reward: {avg_reward}")
        self.logger.info(f"Average Episode Length: {avg_episode_length}")
        self.logger.info(f"Nash Equilibria: {nash_equilibria}")
        
        # Visualization
        self.metrics_tracker.plot_metrics()
        
        # Optional: Save results
        self._save_results(experiment_results)
    
    def _save_results(self, experiment_results):
        """
        Save experiment results
        
        Args:
            experiment_results (dict): Training experiment results
        """
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Timestamp for unique file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save reward data
        np.save(
            os.path.join(results_dir, f'rewards_{timestamp}.npy'), 
            experiment_results['total_rewards']
        )
        
        # Plot and save reward progression
        plt.figure(figsize=(10, 5))
        plt.plot(experiment_results['total_rewards'])
        plt.title('Reward Progression')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(os.path.join(results_dir, f'reward_plot_{timestamp}.png'))
        plt.close()

def main():
    """
    Main execution point for training
    """
    try:
        # Initialize trainer
        trainer = ScarceGridTrainer()
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()