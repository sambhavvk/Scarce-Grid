import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from src.environment import ScarceGridEnv
from src.agents import QLearningAgent, MultiAgentQLearning
from src.game_theory import GameTheoryAnalyzer
from src.utils import (
    Logger, 
    ConfigManager, 
    RandomSeedManager, 
    visualize_grid
)

class ModelEvaluator:
    def __init__(self, config=None, model_path=None):
        """
        Initialize model evaluation environment
        
        Args:
            config (dict, optional): Evaluation configuration
            model_path (str, optional): Path to saved model
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
        
        # Game Theory Analyzer
        self.game_theory_analyzer = GameTheoryAnalyzer()
        
        # Load or initialize multi-agent system
        self.multi_agent_learning = self._load_or_create_agents(model_path)
    
    def _load_or_create_agents(self, model_path=None):
        """
        Load pre-trained agents or create new ones
        
        Args:
            model_path (str, optional): Path to saved model
        
        Returns:
            MultiAgentQLearning: Configured multi-agent system
        """
        multi_agent_system = MultiAgentQLearning(
            env=self.env, 
            num_agents=self.config['training'].get('num_agents', 2)
        )
        
        if model_path and os.path.exists(model_path):
            try:
                # Load agent Q-tables
                for i, agent in enumerate(multi_agent_system.agents):
                    agent_file = os.path.join(model_path, f'agent_{i}_qtable.npy')
                    agent.q_table = np.load(agent_file, allow_pickle=True).item()
                self.logger.info(f"Loaded pre-trained agents from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")
        
        return multi_agent_system
    
    def evaluate_performance(self, num_episodes=100):
        """
        Comprehensive performance evaluation
        
        Args:
            num_episodes (int): Number of evaluation episodes
        
        Returns:
            dict: Performance metrics
        """
        performance_metrics = {
            'total_rewards': [],
            'episode_lengths': [],
            'reward_distributions': [],
            'agent_strategies': []
        }
        
        for episode in range(num_episodes):
            # Reset environment
            states = self.env.reset()
            done = False
            steps = 0
            episode_rewards = [0, 0]  # Rewards for each agent
            
            while not done:
                # Get deterministic actions (no exploration)
                actions = [
                    np.argmax(agent.q_table.get(
                        tuple(state.flatten()), 
                        np.zeros(self.env.action_space.n)
                    )) 
                    for agent, state in zip(
                        self.multi_agent_learning.agents, 
                        states
                    )
                ]
                
                # Environment step
                next_states, rewards, done, _, info = self.env.step(actions)
                
                # Track rewards
                for i in range(len(rewards)):
                    episode_rewards[i] += rewards[i]
                
                states = next_states
                steps += 1
                
                # Prevent infinite loops
                if steps >= self.config['training']['max_steps_per_episode']:
                    break
            
            # Store performance metrics
            performance_metrics['total_rewards'].append(sum(episode_rewards))
            performance_metrics['episode_lengths'].append(steps)
            performance_metrics['reward_distributions'].append(episode_rewards)
        
        return performance_metrics
    
    def analyze_agent_interactions(self, performance_metrics):
        """
        Detailed analysis of agent interactions
        
        Args:
            performance_metrics (dict): Performance metrics
        
        Returns:
            dict: Interaction analysis results
        """
        interaction_analysis = {
            'average_total_reward': np.mean(performance_metrics['total_rewards']),
            'reward_variance': np.var(performance_metrics['total_rewards']),
            'episode_length_stats': {
                'mean': np.mean(performance_metrics['episode_lengths']),
                'std': np.std(performance_metrics['episode_lengths'])
            },
            'reward_distribution_analysis': self._analyze_reward_distribution(
                performance_metrics['reward_distributions']
            )
        }
        
        return interaction_analysis
    
    def _analyze_reward_distribution(self, reward_distributions):
        """
        Analyze reward distribution between agents
        
        Args:
            reward_distributions (list): List of reward pairs
        
        Returns:
            dict: Reward distribution analysis
        """
        agent_rewards = list(zip(*reward_distributions))
        
        return {
            'agent_1_rewards': {
                'mean': np.mean(agent_rewards[0]),
                'std': np.std(agent_rewards[0])
            },
            'agent_2_rewards': {
                'mean': np.mean(agent_rewards[1]),
                'std': np.std(agent_rewards[1])
            },
            'reward_ratio': np.mean(agent_rewards[0]) / np.mean(agent_rewards[1])
        }
    
    def visualize_results(self, performance_metrics):
        """
        Create comprehensive visualizations
        
        Args:
            performance_metrics (dict): Performance metrics
        """
        plt.figure(figsize=(15, 10))
        
        # Reward Distribution
        plt.subplot(2, 2, 1)
        plt.title('Total Reward Distribution')
        plt.hist(performance_metrics['total_rewards'], bins=20)
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        
        # Episode Length Distribution
        plt.subplot(2, 2, 2)
        plt.title('Episode Length Distribution')
        plt.hist(performance_metrics['episode_lengths'], bins=20)
        plt.xlabel('Episode Length')
        plt.ylabel('Frequency')
        
        # Agent Reward Comparison
        plt.subplot(2, 2, 3)
        agent_rewards = list(zip(*performance_metrics['reward_distributions']))
        plt.boxplot(agent_rewards, labels=['Agent 1', 'Agent 2'])
        plt.title('Agent Reward Comparison')
        plt.ylabel('Rewards')
        
        # Scatter plot of agent rewards
        plt.subplot(2, 2, 4)
        plt.scatter(
            agent_rewards[0], 
            agent_rewards[1], 
            alpha=0.5
        )
        plt.title('Agent Reward Correlation')
        plt.xlabel('Agent 1 Rewards')
        plt.ylabel('Agent 2 Rewards')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, performance_metrics):
        """
        Generate comprehensive evaluation report
        
        Args:
            performance_metrics (dict): Performance metrics
        """
        interaction_analysis = self.analyze_agent_interactions(performance_metrics)
        
        report = f"""
        === ScarseGrid Multi-Agent Evaluation Report ===
        
        Performance Metrics:
        - Average Total Reward: {interaction_analysis['average_total_reward']:.2f}
        - Reward Variance: {interaction_analysis['reward_variance']:.2f}
        
        Episode Length:
        - Mean: {interaction_analysis['episode_length_stats']['mean']:.2f}
        - Standard Deviation: {interaction_analysis['episode_length_stats']['std']:.2f}
        
        Reward Distribution:
        Agent 1:
        - Mean Reward: {interaction_analysis['reward_distribution_analysis']['agent_1_rewards']['mean']:.2f}
        - Reward Std Dev: {interaction_analysis['reward_distribution_analysis']['agent_1_rewards']['std']:.2f}
        
        Agent 2:
        - Mean Reward: {interaction_analysis['reward_distribution_analysis']['agent_2_rewards']['mean']:.2f}
        - Reward Std Dev: {interaction_analysis['reward_distribution_analysis']['agent_2_rewards']['std']:.2f}
        
        Reward Ratio (Agent 1 / Agent 2): {interaction_analysis['reward_distribution_analysis']['reward_ratio']:.2f}
        """
        
        print(report)
        
        # Optional: Save report to file
        with open(os.path.join(project_root, 'results', 'evaluation_report.txt'), 'w') as f:
            f.write(report)

def main():
    """
    Main evaluation execution
    """
    try:
        # Path to saved model (modify as needed)
        model_path = os.path.join(project_root, 'saved_models')
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path=model_path)
        
        # Evaluate performance
        performance_metrics = evaluator.evaluate_performance(num_episodes=100)
        
        # Visualize results
        evaluator.visualize_results(performance_metrics)
        
        # Generate report
        evaluator.generate_report(performance_metrics)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()