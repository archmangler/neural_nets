import numpy as np

#activation function is a heaviside (step function around 0 x axis)
def activation_function(x):
    if x<1:
        return 0
    return 1

class PerceptronNetwork:
    def __init__(self, inputs, labels):
        self.input_features = inputs
        self.input_labels = labels
        self.num_weights = len(inputs[0])
        self.weights = np.random.rand(self.num_weights)

    def train(self, learning_rate=0.3):
        total_error = 1
        while total_error != 0:
            total_error = 0
            for i, input_feature in enumerate(self.input_features):
                calculate_output = self.calculate_output(input_feature)
                error = self.input_labels[i] - calculate_output

                total_error = total_error + error

                for j, weight in enumerate(self.weights):
                   self.weights[j] = weight + learning_rate * error * input_feature[j]
                   print("Training the neural network ....")

    def calculate_output(self, activations):
        net_input = activations.dot(self.weights)
        return activation_function(net_input)

if __name__ == "__main__":

    #Learning and Behaving like an AND Operator:
    input_features = np.array([[0, 0], [0, 1], [1,0], [1,1]])
    input_labels = np.array([[0],[0],[0],[1]])
    network = PerceptronNetwork(input_features, input_labels)
    network.train()
    print("Training the neural network to be an AND Operator is complete ...")
    print(network.calculate_output(np.array([0, 0])))
    print(network.calculate_output(np.array([0, 1])))
    print(network.calculate_output(np.array([1, 0])))
    print(network.calculate_output(np.array([1, 1])))

    #Attempt to learn and behave like an XOR Operator:
    input_features = np.array([[0, 0], [0, 1], [1,0], [1,1]])
    input_labels = np.array([[0],[1],[1],[0]])
    network = PerceptronNetwork(input_features, input_labels)
    network.train()
    print("Training the neural network to be an XOR Operator is complete ...")
    print(network.calculate_output(np.array([0, 0])))
    print(network.calculate_output(np.array([0, 1])))
    print(network.calculate_output(np.array([1, 0])))
    print(network.calculate_output(np.array([1, 1])))
