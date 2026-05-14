import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RewardShapingStrategy:
    def __init__(self, base_reward_structure=None):
        """
        Advanced Reward Shaping Mechanism
        
        Args:
            base_reward_structure (dict): Base reward configuration
        """
        # Default configuration if not provided
        self.base_rewards = base_reward_structure or {
            'base_reward': 1.0,
            'cooperation_bonus': 1.5,
            'competition_penalty': 0.5
        }
        
        # Dynamic reward modifiers
        self.dynamic_modifiers = {
            'cooperation_bonus': self.base_rewards.get('cooperation_bonus', 1.5),
            'competition_penalty': self.base_rewards.get('competition_penalty', 0.5)
        }
        
        # Tracking reward history
        self.reward_history = []
    
    def shape_reward(self, agents_states, reward):
        """
        Dynamically modify rewards based on agent interactions
        
        Args:
            agents_states (list): Current states of agents
            reward (float): Base reward
        
        Returns:
            float: Shaped reward
        """
        # Analyze interaction type
        interaction_type = self._analyze_interaction(agents_states)
        
        # Apply reward shaping based on interaction
        if interaction_type == 'cooperative':
            shaped_reward = reward * self.dynamic_modifiers['cooperation_bonus']
        elif interaction_type == 'competitive':
            shaped_reward = reward * self.dynamic_modifiers['competition_penalty']
        else:
            shaped_reward = reward
        
        # Store reward history
        self.reward_history.append(shaped_reward)
        
        return shaped_reward
    
    def _analyze_interaction(self, agents_states):
        """
        Determine interaction type between agents
        
        Args:
            agents_states (list): Current states of agents
        
        Returns:
            str: Interaction type
        """
        # Simple interaction analysis
        # You can expand this with more complex logic
        if len(agents_states) < 2:
            return 'neutral'
        
        # Check proximity or shared resources
        state_similarity = np.sum(np.abs(agents_states[0] - agents_states[1]))
        
        if state_similarity < 0.1:  # Agents are very close
            return 'competitive'
        elif state_similarity > 0.5:  # Agents are somewhat aligned
            return 'cooperative'
        else:
            return 'neutral'
    
    def get_reward_statistics(self):
        """
        Compute reward statistics
        
        Returns:
            dict: Reward statistics
        """
        return {
            'mean_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'max_reward': np.max(self.reward_history) if self.reward_history else 0,
            'min_reward': np.min(self.reward_history) if self.reward_history else 0,
            'reward_variance': np.var(self.reward_history) if self.reward_history else 0
        }
    
    def visualize_reward_distribution(self):
        """
        Visualize reward distribution
        """
        if not self.reward_history:
            print("No reward history to visualize")
            return
        
        plt.figure(figsize=(10, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.reward_history, bins=20, edgecolor='black')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        
        # Cumulative Reward
        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(self.reward_history))
        plt.title('Cumulative Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        
        plt.tight_layout()
        plt.show()

class GameTheoryAnalyzer:
    def __init__(self, num_agents=2):
        """
        Game Theory Analysis Toolkit
        
        Args:
            num_agents (int): Number of agents in the game
        """
        self.num_agents = num_agents
    
    def create_payoff_matrix(self, strategies):
        """
        Generate a payoff matrix for different strategy combinations
        
        Args:
            strategies (list): List of possible strategies
        
        Returns:
            np.ndarray: Payoff matrix
        """
        strategy_combinations = list(itertools.product(strategies, repeat=self.num_agents))
        payoff_matrix = np.zeros((len(strategy_combinations), self.num_agents))
        
        for i, combination in enumerate(strategy_combinations):
            payoff_matrix[i] = self._calculate_payoffs(combination)
        
        return payoff_matrix
    
    def _calculate_payoffs(self, strategies):
        """
        Calculate payoffs for a specific strategy combination
        
        Args:
            strategies (tuple): Strategies of each agent
        
        Returns:
            np.ndarray: Payoffs for each agent
        """
        # More comprehensive payoff calculation
        if len(set(strategies)) == 1:
            # Cooperative scenario
            return np.full(self.num_agents, 5.0)
        elif len(set(strategies)) == self.num_agents:
            # Competitive scenario
            return np.array([3.0, 1.0]) if self.num_agents == 2 else np.zeros(self.num_agents)
        else:
            # Mixed strategy scenario
            return np.array([2.0, 4.0]) if self.num_agents == 2 else np.zeros(self.num_agents)
    
    def nash_equilibrium(self, payoff_matrix):
        """
        Identify Nash Equilibrium in the game
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix
        
        Returns:
            list: Nash Equilibrium strategies
        """
        nash_equilibria = []
        
        # Ensure payoff_matrix is 2D
        if payoff_matrix.ndim == 1:
            payoff_matrix = payoff_matrix.reshape(1, -1)
        
        # Generate all possible strategy combinations
        strategy_combinations = list(itertools.product(
            range(payoff_matrix.shape[0]), 
            repeat=self.num_agents
        ))
        
        for combination in strategy_combinations:
            try:
                is_nash = self._is_nash_equilibrium(payoff_matrix, combination)
                if is_nash:
                    nash_equilibria.append(combination)
            except Exception as e:
                print(f"Error checking combination {combination}: {e}")
        
        return nash_equilibria
    
    def _is_nash_equilibrium(self, payoff_matrix, strategy_combination):
        """
        Check if a strategy combination is a Nash Equilibrium
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix
            strategy_combination (tuple): Current strategy combination
        
        Returns:
            bool: Whether the combination is a Nash Equilibrium
        """
        # Ensure strategy_combination is valid
        if len(strategy_combination) != self.num_agents:
            return False
        
        # Get current payoffs
        try:
            current_payoffs = payoff_matrix[strategy_combination]
        except IndexError:
            return False
        
        # Ensure current_payoffs is a numpy array
        current_payoffs = np.atleast_1d(current_payoffs)
        
        # Check if any agent can unilaterally improve their payoff
        for agent in range(self.num_agents):
            for alternative_strategy in range(payoff_matrix.shape[0]):
                # Create alternative combination
                alternative_combination = list(strategy_combination)
                alternative_combination[agent] = alternative_strategy
                
                try:
                    alternative_payoffs = payoff_matrix[tuple(alternative_combination)]
                    alternative_payoffs = np.atleast_1d(alternative_payoffs)
                    
                    # Compare payoffs
                    if alternative_payoffs[agent] > current_payoffs[agent]:
                        return False
                except IndexError:
                    continue
        
        return True
    
    def cooperation_potential(self, payoff_matrix):
        """
        Analyze potential for cooperation
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix
        
        Returns:
            float: Cooperation potential score
        """
        # Ensure payoff_matrix is 2D
        if payoff_matrix.ndim == 1:
            payoff_matrix = payoff_matrix.reshape(1, -1)
        
        # Calculate cooperative potential
        cooperative_payoffs = np.max(payoff_matrix, axis=0)
        total_cooperative_payoff = np.sum(cooperative_payoffs)
        total_potential_payoff = np.sum(payoff_matrix)
        
        return total_cooperative_payoff / total_potential_payoff if total_potential_payoff != 0 else 0

class GameTheoryVisualization:
    @staticmethod
    def plot_payoff_matrix(payoff_matrix):
        """
        Visualize payoff matrix using heatmap
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix to visualize
        """
        # Ensure payoff_matrix is 2D
        if payoff_matrix.ndim == 1:
            payoff_matrix = payoff_matrix.reshape(1, -1)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            payoff_matrix, 
            annot=True, 
            cmap='YlGnBu', 
            fmt='.2f'
        )
        plt.title('Game Payoff Matrix')
        plt.xlabel('Agent Strategies')
        plt.ylabel('Payoff Values')
        plt.show()
    
    @staticmethod
    def plot_strategy_evolution(strategy_history):
        """
        Plot strategy evolution over time
        
        Args:
            strategy_history (list): History of agent strategies
        """
        plt.figure(figsize=(12, 6))
        for agent, strategies in enumerate(strategy_history):
            plt.plot(strategies, label=f'Agent {agent+1}')
        
        plt.title('Agent Strategy Evolution')
        plt.xlabel('Iterations')
        plt.ylabel('Strategy')
        plt.legend()
        plt.show()

def run_game_theory_analysis():
    """
    Comprehensive Game Theory Analysis Runner
    """
    # Example strategies: Cooperative, Competitive, Mixed
    strategies = ['cooperative', 'competitive', 'mixed']
    
    # Initialize analyzers
    game_analyzer = GameTheoryAnalyzer(num_agents=2)
    
    # Create payoff matrix
    payoff_matrix = game_analyzer.create_payoff_matrix(strategies)
    
    # Visualize payoff matrix
    GameTheoryVisualization.plot_payoff_matrix(payoff_matrix)
    
    # Analyze Nash Equilibrium
    nash_equilibria = game_analyzer.nash_equilibrium(payoff_matrix)
    print("Nash Equilibria:", nash_equilibria)
    
    # Cooperation potential
    cooperation_score = game_analyzer.cooperation_potential(payoff_matrix)
    print("Cooperation Potential Score:", cooperation_score)

def run_reward_shaping_experiment():
    """
    Demonstrate reward shaping experiment
    """
    # Create reward shaping strategy with custom configuration
    reward_shaper = RewardShapingStrategy({
        'base_reward': 1.0,
        'cooperation_bonus': 1.5,
        'competition_penalty': 0.5
    })
    
    # Simulate some interactions
    # This is a mock simulation - replace with your actual environment logic
    def mock_agent_interaction():
        # Simulate different agent states and rewards
        states = [
            np.random.rand(5, 5),  # Agent 1 state
            np.random.rand(5, 5)   # Agent 2 state
        ]
        base_reward = np.random.uniform(0, 10)
        return states, base_reward
    
    # Run multiple interactions
    for _ in range(100):
        states, base_reward = mock_agent_interaction()
        shaped_reward = reward_shaper.shape_reward(states, base_reward)
    
    # Visualize results
    reward_shaper.visualize_reward_distribution()
    
    # Print reward statistics
    print(reward_shaper.get_reward_statistics())

# Example usage
if __name__ == "__main__":
    run_reward_shaping_experiment()

# Example usage and execution
if __name__ == "__main__":
    run_game_theory_analysis()