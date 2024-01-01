import numpy as np

# the random numbers will be predictable
np.random.seed(93)
class Layer:
    def __init__(self,n_input,n_neurons,activation_function='sigmoid'):
        self.activation_function = activation_function
        self.weights = np.random.randn(n_input,n_neurons)
        self.bias = np.random.randn(n_neurons)
        #the activation after applying the sum and the activation functions ...
        self.last_activation = None
        #for backpropagation we need the delta weight change in the previous iteration
        self.previous_weight_change = 0
        #measure the error - not the MSE but "target - activation"
        self.error = 0
        #delta parameter
        self.delta = 0

    def activate(self,x):
        net_input = np.dot(x,self.weights) + self.bias
        self.last_activation = self.apply_activation_function(net_input)
        return self.last_activation

    def apply_activation_function(self, x):
        if self.activation_function == 'sigmoid':
            return 1/(1+np.exp(-x))

    def apply_activation_function_derivative(self, o):
        if self.activation_function == 'sigmoid':
            return o*(1-o)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self,layer):
        self.layers.append(layer)

    #forward propagation function
    def forward_propagation(self,x):
        for layer in self.layers:
            #calculate the activation function
            x = layer.activate(x)
        return x
    def predict(self, inputs):
        return self.forward_propagation(inputs)

    def back_propagation(self, inputs, labels, learning_rate=0.3, momentum=0.6):
        #the back propagation operation first requires a forward propagation
        # to develop deltas that can be used to adjust the weights.
        output=self.forward_propagation(inputs)

        #proceed through the layers in reverse order.
        #we used "reversed" to proceed in reverse order of the layers
        for i in reversed (range(len(self.layers))):
            actual_layer = self.layers[i]
            #if this is the output layer, then calculate the delta
            if actual_layer == self.layers[-1]:
                actual_layer.error = labels - output
                actual_layer.delta = actual_layer.error * actual_layer.apply_activation_function_derivative(output)
            else:
                # i+ 1 but in the reversed order (so it is the previous layer - input)
                next_layer = self.layers[i+1]
                actual_layer.delta = np.dot(next_layer.weights, next_layer.delta) * actual_layer.apply_activation_function_derivative(actual_layer.last_activation)
        #we have the delta  values we just have to update the weights
        for i in range(len(self.layers)):
            actual_layer = self.layers[i]
            # the input is either the previous layer's output
            #or the inputs themselves (if this is the hidden layer)
            input_to_use = np.atleast_2d(inputs if i==0 else self.layers[i-1].last_activation)
            actual_layer.dw = actual_layer.delta * input_to_use.T * learning_rate + momentum * actual_layer.previous_weight_change
            actual_layer.previous_weight_change = actual_layer.dw

    def train(self, features, labels, learning_rate, max_epochs):
        mse = []
        for epoch_counter in range(max_epochs):
            for j in range(len(features)):
                self.back_propagation(features[j], labels[j], learning_rate)
                mean_squared_error = np.mean(np.square(labels-self.predict(features)))
                mse.append(mean_squared_error)
                print("Training epoch %s and the MSE: %s " % (epoch_counter, float(mean_squared_error)))


if __name__ == "__main__":

    network = NeuralNetwork()
    network.add_layer(Layer(2,3,'sigmoid'))
    network.add_layer(Layer(3,1,'sigmoid'))

    #data-set for the XOR logical operator
    x = [[0,0],[0,1],[1,0],[1,1]]
    y = [[0],[1],[1],[0]]

    #now train the network for 1000, 10 000, 1000 000 iterations
    network.train(x,y,0.3,1000)
    print("The predicted output: " + str(network.predict(x)))