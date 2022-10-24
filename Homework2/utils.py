import pickle, gzip
import numpy as np
import sys

### MNIST dataset: ( [example1, example2, example3, ... ],
###                  [label1,   label2,   label3,   ... ] )
### example : [] of 784 elements : 0 white ; 1 black ; 0..1 gray
### label   : 0 / 1 / 2 / ... / 9 == which digit is represented

def read_data_sets():
    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
        np.set_printoptions(threshold=sys.maxsize)
        # print(train_set)
        # print(valid_set[0][0])
        # print(valid_set[1])
        # print(test_set)
        return (train_set, valid_set, test_set)

### compute input as (bias = 1, atr.1, atr.2, ..., atr.n)
def add_bias_to_inputs(inputs):
    input_list = np.insert(np.copy(inputs), 0, 1, axis=1)
    # for input in input_list:
    #     input = np.insert(input, 0, 1)
    return input_list

def transform_labels_for_digit(labels, digit):
    neuron_labels = []
    for label in labels:
        if label == digit:
            neuron_labels += [1]
        else:
            neuron_labels += [0]
    return neuron_labels

### activation function:
def predict(input, weights):
    return (np.dot(input,weights) >= 0)

def predict_likeliness(input, weights):
    return np.dot(input, weights)

def accuracy(inputs, weights, labels):
    correct = 0
    preds = []
    for i in range(len(inputs)):
        preds += [predict(inputs[i],weights)]
        if predict(inputs[i],weights) == labels[i]:
            correct += 1
    # print("\nPredictions: ", preds)
    return correct/len(inputs)

## epoch : pass through all dataset
## iteration : pass through batch
def train(inputs, labels, weights, epochs=10, learn_rate=0.15):
    prev_acc = -1
    for epoch in range(epochs):
        acc = accuracy(inputs,weights,labels)
        print("\nEpoch %d" % epoch)
        # print("\nWeights: ", weights)
        print("\nAccuracy: ", acc)

        if acc == 1.00 or acc == prev_acc:
            break
        prev_acc = acc

        # MINIBATCH TRAINING:
        minibatch_size = 5
        for i in range(minibatch_size):
            weights = process_mini_batch(inputs[i::minibatch_size], labels[i::minibatch_size],\
                weights, learn_rate)

        # ONLINE TRAINING:
        # for i in range(len(inputs)):
        #     pred = predict(inputs[i],weights)
        #     error = labels[i] - pred

        #     weights = weights + error*learn_rate*inputs[i]
        #     # extremely slow: 
        #     # for j in range(len(weights)):
        #     #     weights[j] = weights[j] + (learn_rate*error*inputs[i][j])

    return weights

def process_mini_batch(batch_inputs, batch_labels, weights, learn_rate):
    delta = np.array([0] * len(weights))
    for i in range(len(batch_inputs)):
        pred = predict(batch_inputs[i],weights)
        error = batch_labels[i] - pred
        delta = delta + error*learn_rate*batch_inputs[i]
    return weights + delta

## init weights for 784 inputs and 1 bias:
def init_weights():
    weights = 2*np.random.random((10,785)) - 1
    return weights

def train_digits(train_inputs, train_labels, epochs, learn_rate):
    weights = init_weights()
    for i in range(10):
        digit_labels = transform_labels_for_digit(train_labels,i)
        weights[i] = \
            train(train_inputs, digit_labels, weights[i], epochs, learn_rate)
    return weights

def predict_digit(input, weights):
    predictions = []
    for i in range(10):
        predictions += [predict_likeliness(input, weights[i])]
    return predictions.index(max(predictions))

def test_model(test_inputs, test_labels, weights):
    correct = 0
    for i in range(len(test_inputs)):
        if predict_digit(test_inputs[i], weights) == test_labels[i]:
            correct += 1
    return correct/len(test_inputs)

def print_model(weights, epochs, learn_rate):
    with open('digit_model', 'w') as f:
        f.write(str(weights))

    with open('digit_hyperparams', 'w') as g:
        g.write(str(epochs))
        g.write(" ")
        g.write(str(learn_rate))

def tune_hyperparameters(train_inputs, train_labels, valid_inputs, valid_labels, weights, epochs=10, learn_rate=0.15):
    while test_model(valid_inputs, valid_labels, weights) < 0.85:
        ## adjust hyperparams:
        epochs += 5
        learn_rate = (learn_rate * 3)/ 4
        ## retrain model:
        weights = train_digits(train_inputs, train_labels, epochs, learn_rate)
    print_model(weights, epochs, learn_rate)
    return (weights, epochs, learn_rate)
        

def main():
    ###!!!! i forgot about bias in weights? compare input and wheight lengths
    ###!!!! add validation test: modify learn_rate and epochs as follows:
    ###     1.5 10 => 0.75 15 => fie 1.1 15 / fie 0.35 20 => ... based on accuracy
    ###     print result (weights and hyperparams) to file
    epochs = 10 # default:30
    learn_rate = 0.15 # default:0.1
    (train_set, valid_set, test_set) = read_data_sets()

    train_inputs = add_bias_to_inputs(train_set[0])
    train_labels = train_set[1]

    # weights = init_weights()
    # weights = train(train_inputs,transform_labels_for_digit(train_labels,0),weights,epochs,learn_rate)
    # print("Is 0?", predict(add_bias_to_inputs(valid_set[0])[18], weights))

    model_weights = train_digits(train_inputs, train_labels, epochs, learn_rate)
    # for i in range(10):
    #     print("Is ", i , "?")
    #     print("Answer: ", predict(add_bias_to_inputs(valid_set[0])[18], model_weights[i]))


    ## Tuning phase with validation test:
    valid_inputs = add_bias_to_inputs(valid_set[0])
    valid_labels = valid_set[1]
    (model_weights, epochs, learn_rate) = tune_hyperparameters(train_inputs, train_labels, valid_inputs, valid_labels, model_weights,\
                                            epochs, learn_rate)

    ## Testin phase for accuracy:
    test_inputs = add_bias_to_inputs(test_set[0])
    test_labels = test_set[1]
    print("Model accuracy: ", test_model(test_inputs, test_labels, model_weights))

    ## a ramas sa fac wheights pentru fiecare digit si sa antrenez 
    ##      si dupa sa folosesc setul de validare pentru ajustarea hiperparametrilor
    ##      iar la final sa folosesc test setul pentru verificare
    ## eventual sa ma folosesc si de mini batches pentru bonus


### Validation set: tune hyperparameters: learn_rate, number of iterations, initial weights

### scale learning rate by 1/sqrt(n) where n is batch size

main()



