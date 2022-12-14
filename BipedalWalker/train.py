import gym
import numpy as np

### Solving means gettting over 300 points in 100 consecutive trials

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
		self.b = 16				# No of top performing directions (<= self.N)
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

	def test_env(self, env, policy, weights, normalizer=None, eval_policy=False):
		## policy is a function that takes (weights,state) as input and returns actions
		state = env.reset()
		done = False
		total_reward = 0.0
		steps = 0

		while not done and steps<self.hp.episode_length:  # 5000 is episode length
			# env.render()	## DON'T RENDER ENVIRONMENT AT TRAINING BRO
			if normalizer:
				if not eval_policy: normalizer.observe(state)
				state = normalizer.normalize(state)
			action = policy(state, weights)
			next_state, reward, done = env.step(action)

			# Avoid local optimum (next_state[2] == velocity on x axis)
			if abs(next_state[2]) < 0.001:
				reward = -100
				done = True

			total_reward += reward
			steps += 1
			state = next_state
		if eval_policy: return float(total_reward), steps # Return also the number of steps for evaluating the policy in plots
		else: return float(total_reward)


### Normalizing the states ###
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
		return (inputs - obs_mean) / obs_std

	def store(self):
		np.savetxt('NormalizerMean.txt', self.mean)
		np.savetxt('NormalizerVar.txt', self.var)

### ARS Algorithm ###
class ARS:
	def __init__(self):
		self.hp = HP()
		self.env = BipedalWalker()
		self.agent = Agent()
		self.best_score = -1000
		self.desired_score = 300
		self.size = [self.env.action_space.shape[0], self.env.observation_space.shape[0]] # (4, 24)
		self.weights = np.zeros(self.size)		# 𝜃 parameters
		if self.hp.normalizer: 
			self.hp.normalizer = Normalizer([1,self.size[1]]) # (1, 24) -> normalizing the observation space
		else: 
			self.hp.normalizer=None
		self.plot = ModelPlot()

	def sort_directions(self, reward_p, reward_n):
		reward_max = [max(rp, rn) for rp, rn in zip(reward_p, reward_n)]

		idx = np.argsort(reward_max)[::-1]	# Sort rewards in descending order and get indices.

		return idx

	def update_weights(self, reward_p, reward_n, delta):
		idx = self.sort_directions(reward_p, reward_n)

		step = np.zeros(self.weights.shape)
		for i in range(self.hp.b):
			step += [reward_p[idx[i]] - reward_n[idx[i]]]*delta[idx[i]]

		sigmaR = np.std(np.array(reward_p)[idx][:self.hp.b] + np.array(reward_n)[idx][:self.hp.b])
		self.weights += self.hp.lr / (self.hp.b*sigmaR) * step

	def sample_delta(self, size):
		return [np.random.randn(*size) for _ in range(self.hp.N)]

	def save_policy(self):
		np.savetxt('BipedalWalkerModel.txt', self.weights)
		self.hp.normalizer.store()

	def train_one_epoch(self):
		delta = self.sample_delta(self.size)

		reward_p = [self.agent.test_env(self.env, self.agent.policy, self.weights + self.hp.v*x, normalizer=self.hp.normalizer) for x in delta]
		reward_n = [self.agent.test_env(self.env, self.agent.policy, self.weights - self.hp.v*x, normalizer=self.hp.normalizer) for x in delta]
		
		self.update_weights(reward_p, reward_n, delta)

	def train(self):
		print('Training started...')

		for counter in range(self.hp.iterations):
			self.train_one_epoch()

			test_reward, steps = self.agent.test_env(self.env, self.agent.policy, self.weights, self.hp.normalizer, eval_policy=True)

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

	
