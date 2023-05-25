import matplotlib.pyplot as plt
import numpy as np


class BanditProblem:
    def __init__(self, n_arms: int, arm_values: np.ndarray):
        self.n_arms = n_arms
        self.true_arm_values = arm_values
        self.estimated_arm_values = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        self.reward_history = []
        self.label = "Bandit"

    def pull(self, arm: int) -> float:
        """Samples the reward for a given arm. (abstract fn)

        Args:
            arm (int): Arm pulled

        Returns:
            float: Reward for pulling the arm
        """        
        pass

    def update_estimates(self, action: int, reward: int) -> None:
        """Updates the estimated value of all arms. (abstract fn)

        Args:
            action (int): Arm pulled
            reward (int): Reward for pulling the arm
        """        
        pass

    def select_action(self) -> int:
        """Selects an arm to pull. (abstract fn)

        Returns:
            int: Arm to pull
        """        
        pass

    def run_iter(self) -> None:
        """Runs one iteration of the bandit problem
        """        
        action = self.select_action()
        reward = self.pull(action)
        self.update_estimates(action, reward)
        self.reward_history.append(reward)

    def plot_reward_history(self, show=True) -> None:
        """Plots the reward history
        """
        history = np.array(self.reward_history)
        adjusted_history = np.cumsum(history) / np.arange(1, len(history) + 1)
        plt.plot(adjusted_history, label=self.label)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        if show:
            plt.legend()
            plt.show()


class StationaryBanditProblem(BanditProblem):
    def __init__(self, n_arms: int, arm_values: np.ndarray):
        super().__init__(n_arms, arm_values)
    
    def pull(self, arm: int) -> float:
        """Samples the reward for a given arm.

        Args:
            arm (int): Arm pulled

        Returns:
            float: Reward for pulling the arm
        """        
        return np.random.randn() + self.true_arm_values[arm]
    

class StationaryGreedyBandit(StationaryBanditProblem):
    def __init__(self, n_arms: int, arm_values: np.ndarray, optimistic: bool=False):
        super().__init__(n_arms, arm_values)
        if optimistic:
            self.estimated_arm_values[:] = 10
        self.label = ("Optimistic" if optimistic else "") + "Greedy Bandit"
    
    def update_estimates(self, action: int, reward: int) -> None:
        """Updates the estimated value of all arms.

        Args:
            action (int): Arm pulled
            reward (int): Reward for pulling the arm
        """
        self.n_pulls[action] += 1
        # The expected value of any action is given by the mean of all past rewards for that arm
        self.estimated_arm_values[action] += (reward - self.estimated_arm_values[action]) / self.n_pulls[action]
        super().update_estimates(action, reward)
    
    def select_action(self) -> int:
        """Selects an arm to pull greedily

        Returns:
            int: Selected arm
        """
        return self.estimated_arm_values.argmax()

class StationaryEpsilonGreedyBandit(StationaryGreedyBandit):
    def __init__(self, n_arms: int, arm_values: np.ndarray, epsilon: float, optimistic: bool=False):
        super().__init__(n_arms, arm_values)
        self.epsilon = epsilon
        if optimistic:
            self.estimated_arm_values[:] = 10
        self.label = ("Optimistic" if optimistic else "") + f"Epsilon Greedy Bandit (eps={epsilon})"

    def select_action(self) -> int:
        """Selects greedily with probability 1 - eps, otherwise selects at random uniformly
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        return super().select_action()


def simulate_stationary():
    num_iters = 2000
    num_arms = 10
    eps = [0.01, 0.05, 0.1, 0.2]

    arm_means = np.random.randn(num_arms) * 5
    arm_stds = np.ones(num_arms) * 2
    arm_values = np.random.randn(num_arms) * arm_stds + arm_means

    bandits = []
    bandits.append(StationaryGreedyBandit(num_arms, arm_values))
    bandits.append(StationaryGreedyBandit(num_arms, arm_values, optimistic=True))
    for epsilon in eps:
        bandits.append(StationaryEpsilonGreedyBandit(num_arms, arm_values, epsilon))
        bandits.append(StationaryEpsilonGreedyBandit(num_arms, arm_values, epsilon, optimistic=True))

    for _ in range(num_iters):
        for bandit in bandits:
            bandit.run_iter()
    
    for bandit in bandits[:-1]:
        bandit.plot_reward_history(show=False)
    bandits[-1].plot_reward_history(show=True)


if __name__ == '__main__':
    simulate_stationary()

