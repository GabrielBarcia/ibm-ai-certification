import numpy as np # import the Numpy library

'''
Logic:
Set the parameters:
Inputs - relevant only for the first run
m - array with number of nodes in each hidden layer
num hidden layers = the len of m, because M contains the number of nodes
                    per hidden layer, hence also contains the hidden layers

num_output_layers = this is there just to sume the +1 in the range. Probably
                    there is a better solution

num_output nodes = the output nodes. Above is output layers. layers != nodes.

Logic:
create the name of the layer by appening the number
if it is the last element of the range, name it as output

for each element, add the node name.
in each node add the weigth and bias

'''



inputs = 2
m = [2,2] #nodes in each hidden layer
num_hidden_layers = len(m)

num_output_layers = 1
num_output_nodes = 1


num_nodes_previous_layer = inputs
neural_network = {}  #Initialize a empty neural network

# for layer in range( num_hidden_layers + num_output_layers ):

#     #define the name of the layer
#     if layer == num_hidden_layers:
#         layer_name = 'output'
#         num_nodes = num_output_nodes
#     else:
#         layer_name = 'layer_{}'.format(layer+1)
#         num_nodes = m[layer]

#     #initialize empty disctionary entry for the layer_n or output
#     neural_network[layer_name] = {} 
#     for node in range(num_nodes):
#         node_name = "node_{}".format(node+1)
#         neural_network[layer_name][node_name] = {
#             'weights': np.around(
#                 #uniform = list of random items
#                 np.random.uniform(size= num_nodes_previous_layer), decimals=2 
#             ),
#             'bias': np.around(
#                 np.random.uniform(size=1), decimals=2
#             )
#         }
    
#     num_nodes_previous_layer = num_nodes

# print(neural_network)

inputs = 2
m = [2,2] #nodes in each hidden layer
num_hidden_layers = len(m)

num_output_layers = 1
num_output_nodes = 1


num_nodes_previous_layer = inputs
neural_network = {}  #Initialize a empty neural network

for layer in range(1, num_hidden_layers+2):
    if layer == num_hidden_layers +1:
        layer_name = 'output'
        num_nodes = 1
    else:
        layer_name = 'layer_{}'.format(layer)
        num_nodes = m[layer-1]

    #Initialize weight and bias for each node
    neural_network[layer_name] = {}
    for node in range(1, num_nodes+1):
        node_name = 'node_{}'.format(node)
        neural_network[layer_name][node_name] = {
            'weight': np.around(
                np.random.uniform(size=num_nodes_previous_layer),
                decimals= 2
            ),
            'bias': np.around(
                np.random.uniform(size=1),
                decimals=2
            )
            
        }
    num_nodes_previous_layer = num_nodes

print(neural_network)





