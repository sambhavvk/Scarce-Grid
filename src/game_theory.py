import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
# BUG 26 FIX: Removed unused pandas import

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
        if len(agents_states) < 2:
            return 'neutral'
        
        # BUG FIX: Renamed variable for clarity (this is a distance, not similarity)
        state_distance = np.sum(np.abs(agents_states[0] - agents_states[1]))
        
        if state_distance < 0.1:  # Agents are very close (competing for same resources)
            return 'competitive'
        elif state_distance > 0.5:  # Agents are spread out (cooperating)
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
        self.strategies = None
        self.strategy_combinations = None
        self.combination_to_index = None
    
    def create_payoff_matrix(self, strategies):
        """
        Generate a payoff matrix for different strategy combinations
        
        Args:
            strategies (list): List of possible strategies
        
        Returns:
            np.ndarray: Payoff matrix
        """
        # Store strategies for Nash equilibrium calculation
        self.strategies = strategies
        self.strategy_combinations = list(itertools.product(strategies, repeat=self.num_agents))
        self.combination_to_index = {combo: i for i, combo in enumerate(self.strategy_combinations)}
        
        payoff_matrix = np.zeros((len(self.strategy_combinations), self.num_agents))
        
        for i, combination in enumerate(self.strategy_combinations):
            payoff_matrix[i] = self._calculate_payoffs(combination)
        
        return payoff_matrix
    
    def _calculate_payoffs(self, strategies):
        """
        BUG 15 FIX: Calculate payoffs based on actual strategy types,
        not just whether strategies are the same or different.
        Implements a Prisoner's Dilemma-style payoff structure.
        
        Args:
            strategies (tuple): Strategies of each agent
        
        Returns:
            np.ndarray: Payoffs for each agent
        """
        if self.num_agents == 2:
            # Explicit payoff lookup for 2-agent games
            payoff_table = {
                ('cooperative', 'cooperative'): (5.0, 5.0),
                ('cooperative', 'competitive'): (1.0, 6.0),
                ('cooperative', 'mixed'): (3.0, 4.0),
                ('competitive', 'cooperative'): (6.0, 1.0),
                ('competitive', 'competitive'): (2.0, 2.0),
                ('competitive', 'mixed'): (4.0, 3.0),
                ('mixed', 'cooperative'): (4.0, 3.0),
                ('mixed', 'competitive'): (3.0, 4.0),
                ('mixed', 'mixed'): (3.5, 3.5),
            }
            return np.array(payoff_table.get(tuple(strategies), (0.0, 0.0)))
        else:
            # For n-agent games: cooperative bonus when all cooperate,
            # reduced payoffs otherwise
            if len(set(strategies)) == 1 and strategies[0] == 'cooperative':
                return np.full(self.num_agents, 5.0)
            elif all(s == 'competitive' for s in strategies):
                return np.full(self.num_agents, 2.0)
            else:
                return np.full(self.num_agents, 3.0)
    
    def nash_equilibrium(self, payoff_matrix):
        """
        BUG 16 FIX: Correctly identify Nash Equilibrium in the game.
        Uses stored strategy combinations and proper index mapping.
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix
        
        Returns:
            list: Nash Equilibrium strategy combinations
        """
        if self.strategy_combinations is None or self.combination_to_index is None:
            raise ValueError("Must call create_payoff_matrix() before nash_equilibrium()")
        
        nash_equilibria = []
        
        # Ensure payoff_matrix is 2D
        if payoff_matrix.ndim == 1:
            payoff_matrix = payoff_matrix.reshape(1, -1)
        
        num_combinations = len(self.strategy_combinations)
        
        for row_idx in range(num_combinations):
            combination = self.strategy_combinations[row_idx]
            is_nash = True
            
            for agent in range(self.num_agents):
                current_payoff = payoff_matrix[row_idx, agent]
                current_strategy = combination[agent]
                
                # Check all possible alternative strategies for this agent
                for alt_strategy in self.strategies:
                    if alt_strategy == current_strategy:
                        continue
                    
                    # Create alternative combination
                    alt_combination = list(combination)
                    alt_combination[agent] = alt_strategy
                    alt_combo_tuple = tuple(alt_combination)
                    
                    # Look up the row index for the alternative combination
                    if alt_combo_tuple not in self.combination_to_index:
                        continue
                    
                    alt_row = self.combination_to_index[alt_combo_tuple]
                    alt_payoff = payoff_matrix[alt_row, agent]
                    
                    # If agent can improve by deviating, this is not Nash Equilibrium
                    if alt_payoff > current_payoff:
                        is_nash = False
                        break
                
                if not is_nash:
                    break
            
            if is_nash:
                nash_equilibria.append(combination)
        
        return nash_equilibria
    
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
    def mock_agent_interaction():
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

# BUG 20 FIX: Single __main__ block instead of two separate ones
if __name__ == "__main__":
    run_reward_shaping_experiment()
    run_game_theory_analysis()