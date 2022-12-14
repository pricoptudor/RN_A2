import gym
import numpy as np


class BipedalWalker:
	def __init__(self, render_mode='human'):
		self.env = gym.make("BipedalWalker-v3", render_mode=render_mode)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space

	def reset(self):
		return self.env.reset()[0]

	def step(self, action):
		obs, rew, done, truncated, info = self.env.step(action)
		return obs, rew, done

	def set_state(self, state):
		return self.env.set_state(state)

	def get_state(self):
		return self.env.get_state(self.env)

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


def policy(state, weights):
	# print('Weights: ', weights.shape) # weights : (4, 24)
	# print('State: ', state.shape)  	# state : (24, 1)
	return np.matmul(weights, state) 	# action : (4, 1)


def test(env, policy, weights, episodes=100, normalizer=None):
	## policy is a function that takes (weights,state) as input and returns actions
	rewards = []
	for ep in range(episodes):
		state = env.reset()
		done = False
		total_reward = 0.0
		steps = 0

		while not done and steps<5000: ## 5000 is episode_length
			env.render()
			if normalizer:
				state = normalizer.normalize(state)
			action = policy(state, weights)
			next_state, reward, done = env.step(action)

			total_reward+=reward
			steps+=1
			state = next_state
		
		print(f'Episode: {ep}/{episodes}, Score: {total_reward}, Steps: {steps}')
		rewards += [total_reward]

	return rewards


#################### Normalizing the states #################### 
class Normalizer():
	def __init__(self, nb_inputs):
		self.mean = np.zeros(nb_inputs)
		self.var = np.zeros(nb_inputs)

	def restore(self):
		self.mean = np.loadtxt('NormalizerMean.txt')
		self.var = np.loadtxt('NormalizerVar.txt')

	def normalize(self, inputs):
		obs_mean = self.mean
		obs_std = np.sqrt(self.var)
		return (inputs - obs_mean) / obs_std

def plot_results(rewards, episodes=100):
	import matplotlib.pyplot as plt
	print(f'Mean of {episodes} episodes: {np.mean(rewards)}')
	plt.plot(rewards[0])
	plt.title('BipedalWalker with ARS')
	plt.xlabel('Episodes')
	plt.ylabel('Reward per episode')
	plt.tick_params()
	plt.axhline(y=np.mean(rewards))
	plt.show()


if __name__ == '__main__':
	weights = np.loadtxt('BipedalWalkerModel.txt')
	env = BipedalWalker()
	normalizer = Normalizer([1, env.observation_space.shape[0]]) ## env.observation_spave.shape == (24,)
	normalizer.restore()

	rewards = test(env, policy, weights, normalizer=normalizer)
	
	plot_results(np.array(rewards).reshape(1,-1))