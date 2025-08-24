import numpy as np
import matplotlib.pyplot as plt

def initialize_network_parameters( 
        inputSize=2, 
        hiddenSize=2, 
        outputSize=1, 
        learning_rate=0.40, 
        epochs=200000
    ):
    
    # Network parameters
    # inputSize = 2      # Number of input neurons (x1, x2)
    # hiddenSize = 2     # Number of hidden neurons
    # outputSize = 1     # Number of output neurons
    # lr = 0.1           # Learning rate
    # epochs = 180000    # Number of training epochs

    # Initialize weights and biases randomly within the range [-1, 1]
    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # Weights from input to hidden layer
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # Bias for hidden layer
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1 # Weights from hidden to output layer
    b2 = np.random.rand(outputSize, 1) * 2 - 1          # Bias for output layer

    return w1, b1, w2, b2, learning_rate, epochs

def weighted_sum( weigth, X ,bias):
    return np.dot(weigth, X) + bias

def sigmoid( z ):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4 matrix, each column is a training example
d = np.array([0, 1, 1, 0])  # Expected output for XOR
w1, b1, w2, b2, lr, epochs = initialize_network_parameters( learning_rate= 4.1)

error_list = []
for epoch in range(epochs):
    
    # First hidden layer (only hidden layer in this example)
    z1 = weighted_sum( w1, X, b1)
    a1 = sigmoid( z1 )

    # Output layer (for this example)
    z2 = weighted_sum( w2, a1, b2)
    a2 = sigmoid( z2 )

    # in this example, a2 is the result of the neural network
    
    # Calculate the error and backpropagation
    error = d-a2
    da2 = error * ( a2 * (1-a2)) #derivative for putput layer
    dz2 = da2 #gradiend for output layer

    # Propagate the error to the hidden layers
    da1 = np.dot( w2.T, dz2) #gradient for hidden layer
    dz1 = da1 * (a1 * ( 1 - a1 )) #derivation for hidden layer

    # update weigth and biases

    w2 += lr * np.dot( dz2, a1.T ) #update weigth from hidden to output layer
    b2 += lr * np.sum( dz2, axis=1, keepdims=True ) #update bias for output layer

    w1 += lr * np.dot( dz1, X.T ) # Update weigth from input to hidden layer
    b1 += lr * np.sum( dz1, axis=1, keepdims=True ) #update bias for hidden layer

    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))


    