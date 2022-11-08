from Data.dataset import Dataset
from NN.MLNN import MultiLayerNeuralNetwork as MLNN
from Functions.functions import *

dataset = Dataset()

neural_network = MLNN((784, 100, 10), 0.01)

neural_network.train(dataset, 10, 1000)
neural_network.test(dataset.test_set)

print("------------------------------")
print("------------------------------")