import Functions.functions as fct
import numpy as np

class MultiLayerNeuralNetwork:

    def __init__(self, layers, learn_rate):
        self.weights = []
        self.biases = []
        self.learn_rate = learn_rate

        for i in range(len(layers) - 1):
            n = layers[i+1]
            m = layers[i]

            self.weights.append(fct.smart_init_weights(n,m))
            self.biases.append(fct.smart_init_weights(n,1))
        
        # fct.sigmoid = np.vectorize(fct.sigmoid)
        # np.seterr(all="ignore")

    def feed_forward(self, input):
        curr_output = np.reshape(input, (len(input),1))

        final_layer = False
        for i in range(len(self.weights)): 
            if i == len(self.weights) - 1:
                final_layer = True
            curr_output = self.activate(np.dot(self.weights[i], curr_output) + self.biases[i], final_layer)

        return curr_output

    def activate(self, output, final_layer):
        for i in range(output.shape[0]):
            if final_layer == True:
                output[i] = fct.softmax(output, i)
            else:
                output[i] = fct.sigmoid(output, i)
        return output

    def get_outputs(self, inp):
        input = np.reshape(inp, (len(inp), 1))
        curr_input = input

        outputs = [curr_input]
        final_layer = False
        for i in range(len(self.weights)):
            if i == len(self.weights) - 1:
                final_layer = True
            curr_input = self.activate(np.dot(self.weights[i], curr_input) + self.biases[i], final_layer)
            outputs.append(curr_input)

        return outputs

    def get_errors(self, err, outputs):
        curr_error = err
        errors = [curr_error]

        for i in range(len(self.weights)-1, -1, -1):
            trans = np.transpose(self.weights[i])
            ## derivata la softmax... last layer
            ## derivata sigmoid ... altfel
            curr_error = curr_error * outputs[i+1] * (1-outputs[i+1])
            curr_error = np.dot(trans, curr_error)
            errors.insert(0, curr_error)

        return errors

    def train(self, dataset, epochs_len, batches_len):
        print("MLNN training...")

        for e in range(epochs_len):
            epoch_error = 0

            grad_squared_weights = []
            grad_squared_biases = []
            for i in range(len(self.weights)):
                grad_squared_weights.append(np.zeros(self.weights[i].shape))
                grad_squared_biases.append(np.zeros(self.biases[i].shape))

            dataset.shuffle()
            batches = dataset.get_batches(batches_len)

            batch_no = 0
            for batch in batches:
                batch_no += 1
                batch_error = 0

                delta_w = []
                delta_b = []
                for i in range(len(self.weights)):
                    delta_w.append(np.zeros(self.weights[i].shape))
                    delta_b.append(np.zeros(self.biases[i].shape))

                for instance in batch:
                    input = instance[0]
                    target = instance[1]

                    outputs = self.get_outputs(input)
                    errors = self.get_errors(outputs[-1]-target, outputs)

                    err = self.check_prediction(target, outputs[-1])
                    batch_error += err

                    for i in range(len(self.weights)-1,-1,-1):
                #         # print(len(errors), len(errors[0]),len(errors[0][0]))
                #         # print(len(outputs),len(outputs[0]),len(outputs[0][0]))
                        # if i != len(self.weights)-1:
                        #     der = fct.quadratic_derivative(errors[i+1], outputs[i+1],self.weights[i].shape,outputs[i])
                        #     der_bias = fct.quadratic_derivative(errors[i+1],outputs[i+1],self.biases[i].shape)
                        # else:
                        der = fct.cross_entropy_derivative(errors[i+1],self.weights[i].shape)
                        der_bias = fct.cross_entropy_derivative(errors[i+1],self.biases[i].shape)
                            
                        delta_w[i] += der/len(batch)
                        delta_b[i] += der_bias/len(batch)

                # delta_w /= len(batch)
                # delta_b /= len(batch)

                # print(len(delta_w), len(delta_w[0]),len(delta_w[0][0]))
                # print(len(delta_b), len(delta_b[0]),len(delta_b[0][0]))
                # print(len(grad_squared), len(grad_squared[0]),len(grad_squared[0][0]))
                # RMSProp:
                for i in range(len(self.weights)):
                    grad_squared_weights[i] = 0.9 * grad_squared_weights[i] + 0.1 * delta_w[i] * delta_w[i]
                    self.weights[i] = self.weights[i] - (self.learn_rate / np.sqrt(1e-6 + grad_squared_weights[i])) * delta_w[i]
                    grad_squared_biases[i] = 0.9 * grad_squared_biases[i] + 0.1 * delta_b[i] * delta_b[i]
                    self.biases[i] = self.biases[i] - (self.learn_rate / np.sqrt(1e-6 + grad_squared_biases[i])) * delta_b[i]
                # print(len(delta_w), len(delta_w[0]),len(delta_w[0][0]))
                # print(len(delta_b), len(delta_b[0]),len(delta_b[0][0]))
                # print(len(grad_squared), len(grad_squared[0]),len(grad_squared[0][0]))

                batch_error /= len(batch)
                batch_accuracy = (1-batch_error) * 100
                print("Batch ",batch_no,"accuracy: ", round(batch_accuracy,2), "%")
            
            for instance in dataset.train_set:
                epoch_error += self.check_prediction(instance[1], self.feed_forward(instance[0]))
            epoch_error /= len(dataset.train_set)
            epoch_accuracy = (1-epoch_error) * 100
            print("Epoch ", e, " accuracy: " , round(epoch_accuracy,2), "%")
            
    

    def test(self, dataset):
        print("---------- Verify model on test set --------------")
        errors = 0
        for instance in dataset:
            output = self.feed_forward(instance[0])
            # print(output)
            errors += self.check_prediction(instance[1], output)

        accuracy = (1-errors/len(dataset)) * 100
        print("Tested accuracy: ", round(accuracy, 2), "%")

    def check_prediction(self, target, output):
        if target.argmax() != output.argmax():
            return 1
        return 0