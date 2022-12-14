import numpy as np

b=16
idx = np.argsort([5, 2, 8, 1, 10, 12, 3, 4])[::-1]

reward_p = [-5, 6, -7, 8, -9, 10]
reward_n = [5, -6, 7, -8, 9, -10]

reward_max = [max(rp, rn) for rp, rn in zip(reward_p, reward_n)]

sigmaR = np.std(np.array(reward_p)[idx][:b] + np.array(reward_n)[idx][:b])

print(np.array(reward_p)[idx])
print(np.array(reward_n)[idx])