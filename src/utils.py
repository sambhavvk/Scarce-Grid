import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
import random
from datetime import datetime

class Logger:
    """
    Advanced logging utility with multiple output modes
    """
    def __init__(self, project_name='ScarceGrid'):
        """
        Initialize logging configuration
        
        Args:
            project_name (str): Name of the project
        """
        # Create logs directory if not exists
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # BUG 22 FIX: Use a dedicated logger instance instead of root logger
        # to avoid issues with multiple Logger instances and handler accumulation
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if multiple Logger instances are created
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(console_handler)
            
            # File handler
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f'{project_name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """Log informational message"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

class MetricsTracker:
    """
    Comprehensive metrics tracking and analysis
    """
    def __init__(self):
        """Initialize metrics storage"""
        self.metrics = {
            'rewards': [],
            'exploration_rates': [],
            'episode_lengths': [],
            'cumulative_rewards': []
        }
    
    def update(self, reward, exploration_rate, episode_length):
        """
        Update metrics for each episode
        
        Args:
            reward (float): Episode reward
            exploration_rate (float): Current exploration rate
            episode_length (int): Length of the episode
        """
        self.metrics['rewards'].append(reward)
        self.metrics['exploration_rates'].append(exploration_rate)
        self.metrics['episode_lengths'].append(episode_length)
        
        # Cumulative reward tracking
        cumulative_reward = (
            self.metrics['cumulative_rewards'][-1] + reward 
            if self.metrics['cumulative_rewards'] 
            else reward
        )
        self.metrics['cumulative_rewards'].append(cumulative_reward)
    
    def get_summary_statistics(self):
        """
        BUG 23 FIX: Calculate summary statistics safely for empty metrics
        
        Returns:
            dict: Summary statistics
        """
        if not self.metrics['rewards']:
            return {
                'mean_reward': 0,
                'std_reward': 0,
                'max_reward': 0,
                'min_reward': 0,
                'mean_episode_length': 0
            }
        
        return {
            'mean_reward': np.mean(self.metrics['rewards']),
            'std_reward': np.std(self.metrics['rewards']),
            'max_reward': np.max(self.metrics['rewards']),
            'min_reward': np.min(self.metrics['rewards']),
            'mean_episode_length': np.mean(self.metrics['episode_lengths'])
        }
    
    def plot_metrics(self):
        """
        Create comprehensive metrics visualization
        """
        if not self.metrics['rewards']:
            print("No metrics to plot")
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ScarceGrid Learning Metrics')
        
        # Reward progression
        axs[0, 0].plot(self.metrics['rewards'])
        axs[0, 0].set_title('Rewards per Episode')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        
        # Cumulative Reward
        axs[0, 1].plot(self.metrics['cumulative_rewards'])
        axs[0, 1].set_title('Cumulative Reward')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Cumulative Reward')
        
        # Exploration Rate
        axs[1, 0].plot(self.metrics['exploration_rates'])
        axs[1, 0].set_title('Exploration Rate')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Exploration Probability')
        
        # Episode Length
        axs[1, 1].plot(self.metrics['episode_lengths'])
        axs[1, 1].set_title('Episode Lengths')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.show()

class ConfigManager:
    """
    Configuration management and experiment tracking
    """
    def __init__(self, config_path='config.json'):
        """
        Initialize configuration management
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """
        Load configuration from JSON file
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_config()
    
    def create_default_config(self):
        """
        Create default configuration
        
        Returns:
            dict: Default configuration
        """
        default_config = {
            'environment': {
                'grid_size': 5,
                'reward_count': 4
            },
            'agent': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'initial_exploration_rate': 1.0,
                'min_exploration_rate': 0.01
            },
            'training': {
                'episodes': 1000,
                'max_steps_per_episode': 200
            }
        }
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    
    def update_config(self, updates):
        """
        Update configuration
        
        Args:
            updates (dict): Configuration updates
        """
        self.config.update(updates)
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

class RandomSeedManager:
    """
    Seed management for reproducibility
    """
    @staticmethod
    def set_seed(seed=42):
        """
        BUG 24 FIX: Set consistent random seeds for all relevant libraries
        
        Args:
            seed (int): Seed value
        """
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # torch not available

def visualize_grid(grid, title='Grid Visualization'):
    """
    Visualize grid state
    
    Args:
        grid (np.ndarray): Grid to visualize
        title (str): Visualization title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title(title)
    plt.show()

# Example usage demonstration
def main():
    # Logger demonstration
    logger = Logger()
    logger.info("ScarceGrid project initialized")
    
    # Metrics tracking
    metrics_tracker = MetricsTracker()
    
    # BUG 23 FIX: Safe to call on empty metrics now
    print("Summary (empty):", metrics_tracker.get_summary_statistics())
    
    # Configuration management
    config_manager = ConfigManager()
    print(config_manager.config)
    
    # Random seed setting
    RandomSeedManager.set_seed(42)

if __name__ == "__main__":
    main()