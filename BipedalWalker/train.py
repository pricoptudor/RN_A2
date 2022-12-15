import gym
import numpy as np

### Solving means gettting over 300 points in 100 consecutive trials

### only for simple version, hardcore cannot be trained with ARS!

class BipedalWalker:
	def __init__(self, render_mode='rgb_array'):
		self.env = gym.make("BipedalWalker-v3", render_mode=render_mode)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space

	def reset(self):
		return self.env.reset()[0]

	def step(self, action):
		obs, reward, done, truncated, info = self.env.step(action)
		return obs, reward, done 	

	def set_state(self, state):
		return self.env.set_state(state)

	def get_state(self):
		return self.env.get_state(self.env)

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

### Hyperparameters ###
class HP():
	def __init__(self):
		self.v = 0.03			# Noise in delta
		self.N = 16				# No of perturbations
		self.b = 8				# No of top performing directions (<= self.N)
		self.lr = 0.02			# Learning rate
		self.normalizer = True	# Use normalizer

		self.iterations = 3_000
		self.episode_length = 5000
		self.test_iterations = 100

class Agent:
	def __init__(self):
		self.hp = HP()

	def policy(self, state, weights):
		# print('Weights: ', weights.shape) 			# weights : (4, 24)
		# print('State: ', state.shape)  				# state : (1, 24)
		return np.matmul(weights, state.reshape(-1,1))	# action : (4, 1)

	def test_env(self, env, policy, weights, normalizer=None, test_only=False):
		## policy is a function that takes (weights,state) as input and returns actions
		state = env.reset()
		done = False
		total_reward = 0.0
		steps = 0

		while not done and steps<self.hp.episode_length:  # 5000 is episode length
			# env.render()	## DON'T RENDER ENVIRONMENT AT TRAINING BRO
			if normalizer:
				if not test_only: normalizer.observe(state) # obs space contains inf so we compute only what we observe
				state = normalizer.normalize(state)
			action = policy(state, weights)
			next_state, reward, done = env.step(action)

			# Avoid being stuck (next_state[2] == velocity on x axis)
			if abs(next_state[2]) < 0.001:
				reward = -100
				done = True

			total_reward += reward
			steps += 1
			state = next_state
		
		return float(total_reward)


### Normalizing the states ###
## Ensures that policies put equal weight on every component of state 
##		-> otherwise dramatic changes in small ranges will only barely affect computation
class Normalizer():
	def __init__(self, nb_inputs):
		self.n = np.zeros(nb_inputs)
		self.mean = np.zeros(nb_inputs)
		self.mean_diff = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def observe(self, x):
		self.n += 1.
		last_mean = self.mean.copy()
		self.mean += (x - self.mean) / self.n
		self.mean_diff += (x - last_mean) * (x - self.mean)
		self.var = (self.mean_diff / self.n).clip(min=1e-2)

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std ## here (2nd axis of enhancement over BRS)

	def store(self):
		np.savetxt('NormalizerMean.txt', self.mean)
		np.savetxt('NormalizerVar.txt', self.var)

### ARS Algorithm ###
## it explores policy spaces, other algorithms explore action spaces
## (instead of analyzing the rewards it gets after each action,
##		it analyzes the reward after a series of actions)

## uses perceptron instead of deep neural net
## adds tiny values to the weights along with the negative of that value to figure out if they help the agent get a bigger reward
## the bigger the reward from a weight configuration, the bigger its influence on the adjustment of the weights
class ARS:
	def __init__(self):
		self.hp = HP()
		self.env = BipedalWalker()
		self.agent = Agent()
		self.best_score = -1000
		self.desired_score = 300
		self.size = [self.env.action_space.shape[0], self.env.observation_space.shape[0]] # (4, 24)
		self.weights = np.zeros(self.size)		# 𝜃 parameters: shape (output_size, input_size)
		if self.hp.normalizer: 
			self.hp.normalizer = Normalizer([1,self.size[1]]) # (1, 24) -> normalizing the observation space
		else: 
			self.hp.normalizer=None
		self.plot = ModelPlot()

	# Sort rewards in descending order based on (r(𝜃+𝛎𝜹), r(𝜃-𝛎𝜹)) and use only top b rewards with their perturbations 𝜹
	#	(for each iteration the small rewards will push down the average of collected rewards r(𝜃+𝛎𝜹), r(𝜃-𝛎𝜹))
	#	(3rd axis of enhancement over BRS) - (when b==N the algorithm is the same as the one without this enhancement)
	#   ex: mean of 100 iterations = 300: b==N ~ 1000 steps /|\  b==N/2 ~ 400 steps
	def sort_directions(self, reward_p, reward_n):
		reward_max = [max(rp, rn) for rp, rn in zip(reward_p, reward_n)]
		idx = np.argsort(reward_max)[::-1]	# Sort rewards in descending order and get indices.
		return idx

	def update_weights(self, reward_p, reward_n, delta):
		idx = self.sort_directions(reward_p, reward_n)

		step = np.zeros(self.weights.shape)
		for i in range(self.hp.b):
			step += [reward_p[idx[i]] - reward_n[idx[i]]]*delta[idx[i]] # (reward_p, reward_n, delta) == rollouts
		
		# np.array(reward)[idx][:self.hp.b] -> permutation based on sorted idx and take first 'b'
		sigmaR = np.std(np.array(reward_p)[idx][:self.hp.b] + np.array(reward_n)[idx][:self.hp.b])
		# divide by standard deviation of the collected rewards (otherwise the variations can be too big): (1st axis of enhancement over BRS)
		self.weights += self.hp.lr / (self.hp.b*sigmaR) * step

	def sample_delta(self, size):
		return [np.random.randn(*size) for _ in range(self.hp.N)]

	def save_policy(self):
		# np.savetxt('BipedalWalkerModel.txt', self.weights)
		# self.hp.normalizer.store()
		pass ##TODO: here is comment only to test things without messing good weigths

	def train_epoch(self):
		delta = self.sample_delta(self.size)

		reward_p = [self.agent.test_env(self.env, self.agent.policy, self.weights + self.hp.v*x, normalizer=self.hp.normalizer) for x in delta]
		reward_n = [self.agent.test_env(self.env, self.agent.policy, self.weights - self.hp.v*x, normalizer=self.hp.normalizer) for x in delta]
		
		self.update_weights(reward_p, reward_n, delta)

	def train(self):
		print('Training started...')

		for counter in range(self.hp.iterations):
			self.train_epoch()

			test_reward = self.agent.test_env(self.env, self.agent.policy, self.weights, self.hp.normalizer, test_only=True)

			self.plot.rewards += [test_reward]
			self.plot.rewards_means += [np.mean(self.plot.rewards[-self.hp.test_iterations:])]

			if np.mean(self.plot.rewards[-self.hp.test_iterations:]) > self.desired_score and test_reward > self.best_score:
				self.best_score = test_reward
				self.save_policy()
			print(f'Iteration: {counter} -> Reward: {test_reward} || Average: {np.mean(self.plot.rewards[-self.hp.test_iterations:])}')
			
		print('Training ended!')
		self.plot.plot_convergence()

### Plotting convergence ###
class ModelPlot():
	def __init__(self):
		self.steps = [i for i in range(HP().iterations)]
		self.rewards = []
		self.rewards_means = []

	def plot_convergence(self):
		import matplotlib.pyplot as plt
		plt.plot(self.steps, self.rewards_means)
		plt.title('Bipedal Walker -> reinforcement learning with ARS')
		plt.xlabel('Steps')
		plt.ylabel('Reward')
		plt.show()
		plt.savefig('BipedalWalkerConvergence.png')


# Profiling app (to find bottlenecks) -> found in env.render()
def profile_app():
	import cProfile
	import pstats

	ars = ARS()
	profile = cProfile.Profile()
	profile.runcall(ars.train)
	ps = pstats.Stats(profile)
	ps.sort_stats('calls','cumtime')
	ps.print_stats()

if __name__ == '__main__':
	ars = ARS()
	ars.train()

	# profile_app()

	
'''
Model-free reinforcement learning (RL) aims to offer off-the-shelf solutions for controlling dynamical systems without requiring models of the system dynamics
Research papers showed that complicated neural network policies are not needed to solve these continuous control problems
In the quest to find methods that are sample efficient (i.e. methods that need little data) the general trend has been to develop increasingly complicated methods.
Aims to optimize reward by directly optimizing over the policy parameters θ. We consider methods which explore in the parameter space rather than the action space
We demonstrate that a simple random search method can match or exceed state-of-the-art sample efficiency on benchmarks. Moreover, our method is at least 15 times more computationally efficient than Evolution Strategies (ES), the fastest competing method.
State-of-the-art performance is still uniformly achieved. ARS found policies that achieve significantly higher rewards than any other results we encountered in the literature
ARS is not highly sensitive to the choice of hyperparameters because its success rate when varying hyperarameters is similar to its success rate when performing independent trials with a “good” choice of hyperparameters.
'''
