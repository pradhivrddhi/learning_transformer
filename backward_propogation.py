import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


def show_0(matrix_data, title=''):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Normalize data for color mapping
    norm = Normalize(vmin=min(matrix_data), vmax=max(matrix_data))
    # Loop through each index and column to draw lines
    for i in range(len(matrix_data)):
        # Define start and end points for the line
        x = [-1, 1]
        y = [i, i]
        
        # Define color based on the value at the index-column pair
        color = plt.cm.coolwarm(norm(matrix_data[i]))
        
        # Create a line collection with a single line
        lines = [[(x[0], y[0] - len(matrix_data)/2.0), (x[1], y[1] - len(matrix_data) / 2.0)]]
        lc = LineCollection(lines, colors=color, linewidths=2)
        
        # Add the line collection to the plot
        ax.add_collection(lc)

    # Set labels and title
    ax.set_xlabel('Columns')
    ax.set_ylabel('Indices')
    ax.set_title('Index to Column Mapping with Color Intensity')

    # Set x-axis limits
    ax.set_xlim(-1, 1)

    # Set y-axis limits based on number of indices
    ax.set_ylim((-len(matrix_data) -1) / 2.0, (+len(matrix_data) +1) / 2.0)

    # Set y-ticks and labels for indices
    ax.set_yticks(np.arange((-len(matrix_data) -1) / 2.0, (+len(matrix_data) +1) / 2.0))

    # Set x-ticks and labels for columns
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Row', 'Row'])

    # Show plot
    plt.tight_layout()
    plt.title(title)
    plt.show()


def show_1(matrix_data, title=''):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Normalize data for color mapping
    norm = Normalize(vmin=matrix_data.min(), vmax=matrix_data.max())
    # Loop through each index and column to draw lines
    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[1]):
            # Define start and end points for the line
            x = [-1, 1]
            y = [i, j]
            
            # Define color based on the value at the index-column pair
            color = plt.cm.coolwarm(norm(matrix_data[i, j]))
            
            # Create a line collection with a single line
            lines = [[(x[0], y[0] - matrix_data.shape[0]/2.0), (x[1], y[1] - matrix_data.shape[1] / 2.0)]]
            lc = LineCollection(lines, colors=color, linewidths=2)
            
            # Add the line collection to the plot
            ax.add_collection(lc)

    # Set labels and title
    ax.set_xlabel('Columns')
    ax.set_ylabel('Indices')
    ax.set_title('Index to Column Mapping with Color Intensity')

    # Set x-axis limits
    ax.set_xlim(-1, 1)

    # Set y-axis limits based on number of indices
    ax.set_ylim(-max(matrix_data.shape[0], matrix_data.shape[1]) / 2.0, max(matrix_data.shape[0], matrix_data.shape[1]) / 2)

    # Set y-ticks and labels for indices
    ax.set_yticks(np.arange(max(-int(matrix_data.shape[0]/2), int(matrix_data.shape[0]/2))))

    # Set x-ticks and labels for columns
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['', 'Columns'])

    # Show plot
    plt.tight_layout()
    plt.title(title)
    plt.show()

def show_2(matrix_data, title=''):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Normalize data for color mapping
    norm = Normalize(vmin=matrix_data.min(), vmax=matrix_data.max())
    # Loop through each index and column to draw lines
    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[1]):
            for k in range(matrix_data.shape[2]):
                # Define start and end points for the line
                x = [-1, k]
                y = [i, j]
                
                # Define color based on the value at the index-column pair
                color = plt.cm.coolwarm(norm(matrix_data[i, j, k]))
                
                # Create a line collection with a single line
                lines = [[(x[0], y[0] - matrix_data.shape[0]/2.0), (x[1], y[1] - matrix_data.shape[1] / 2.0 + 1 - float(x[1]) / matrix_data.shape[2])]]
                lc = LineCollection(lines, colors=color, linewidths=2)
                
                # Add the line collection to the plot
                ax.add_collection(lc)

    # Set labels and title
    ax.set_xlabel('Columns')
    ax.set_ylabel('Indices')
    ax.set_title('Index to Column Mapping with Color Intensity')

    # Set x-axis limits
    ax.set_xlim(-1, matrix_data.shape[2])

    # Set y-axis limits based on number of indices
    ax.set_ylim(-max(matrix_data.shape[0], matrix_data.shape[1]) / 2, max(matrix_data.shape[0], matrix_data.shape[1]) / 2)

    # Set y-ticks and labels for indices
    ax.set_yticks(np.arange(max(-int(matrix_data.shape[0]/2), int(matrix_data.shape[0]/2))))

    # Show plot
    plt.tight_layout()
    plt.title(title)
    plt.show()


def show(show, X, title=''):
    if not show:
        return
    try:
        show_0(X, title)
    except:
        try:
            show_1(X, title)
        except:
            try:
                show_2(X, title)
            except:
                print(title, X)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize random weights
def initialize_parameters(input_dim, hidden_dim, show1):
    np.random.seed(0)
    W1 = np.random.randn(input_dim, hidden_dim)
    show(show1, W1, 'IP:W1')
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1)
    show(show1, W2, 'IP:W2')
    b2 = np.zeros((1, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# Forward pass
def forward_propagation(X, parameters, show1):
    Z1 = np.dot(X, parameters['W1']) + parameters['b1']
    show(show1, Z1, 'FP:Z1')
    A1 = sigmoid(Z1)
    show(show1, A1, 'FP:A1')
    Z2 = np.dot(A1, parameters['W2']) + parameters['b2']
    show(show1, Z2, 'FP:Z2')
    A2 = sigmoid(Z2)
    show(show1, A2, 'FP:A2')
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

# Backward pass
def backward_propagation(X, Y, parameters, cache, show1):
    m = X.shape[0]
    dZ2 = cache['A2'] - Y
    show(show1, dZ2, 'BP:dZ2')
    dW2 = np.dot(cache['A1'].T, dZ2) / m
    show(show1, dW2, 'BP:dW2')
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    show(show1, db2, 'BP:db2')
    dZ1 = np.dot(dZ2, parameters['W2'].T) * sigmoid_derivative(cache['A1'])
    show(show1, dZ1, 'BP:dZ1')
    dW1 = np.dot(X.T, dZ1) / m
    show(show1, dW1, 'BP:dW1')
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    show(show1, db1, 'BP:db1')
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

# Update parameters
def update_parameters(parameters, grads, learning_rate, show1):
    parameters['W1'] -= learning_rate * grads['dW1']
    show(show1, parameters['W1'], 'UP:W1')
    parameters['b1'] -= learning_rate * grads['db1']
    show(show1, parameters['b1'], 'UP:b1')
    parameters['W2'] -= learning_rate * grads['dW2']
    show(show1, parameters['W2'], 'UP:W2')
    parameters['b2'] -= learning_rate * grads['db2']
    show(show1, parameters['b1'], 'UP:b1')
    return parameters

# Main function to train the neural network
def train_neural_network(X, Y, input_dim, hidden_dim, num_iterations, learning_rate):
    parameters = initialize_parameters(input_dim, hidden_dim, show1=True)
    for i in range(num_iterations):
        cache = forward_propagation(X, parameters, show1=i % 100 == 0)
        grads = backward_propagation(X, Y, parameters, cache, show1=i % 100 == 0)
        parameters = update_parameters(parameters, grads, learning_rate, show1=i % 100 == 0)
        if i % 100 == 0:
            cost = -np.mean(Y * np.log(cache['A2']) + (1 - Y) * np.log(1 - cache['A2']))
            print(f"Iteration {i}, Cost: {cost}")
    return parameters

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
show1 = True
show(show1, X, 'X')
show(show1, Y, 'Y')

input_dim = X.shape[1]
hidden_dim = 4
num_iterations = 1500
learning_rate = 0.1

trained_parameters = train_neural_network(X, Y, input_dim, hidden_dim, num_iterations, learning_rate)
print("Trained Parameters:")
print(trained_parameters)
