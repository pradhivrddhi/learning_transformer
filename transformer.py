import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


def show(matrix_data, title=''):
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
            lines = [[(x[0], y[0] - matrix_data.shape[0]/2), (x[1], y[1] - matrix_data.shape[1] / 2)]]
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
    ax.set_ylim(-max(matrix_data.shape[0], matrix_data.shape[1]) / 2, max(matrix_data.shape[0], matrix_data.shape[1]) / 2)

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

    # Set x-ticks and labels for columns
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['', 'Columns'])

    # Show plot
    plt.tight_layout()
    plt.title(title)
    plt.show()


# Let's pretend this is our small book
book_text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'
"""

# Tokenization: Splitting the text into words
tokens = book_text.split()

# Vocabulary: Create a set of unique words
vocab = set(tokens)

# Word to index mapping
word_to_index = {word: i for i, word in enumerate(vocab)}

# Index to word mapping
index_to_word = {i: word for word, i in word_to_index.items()}

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    z = np.max(x, axis=-1, keepdims=True)
    y = x - z
    exp_x = np.exp(y)
    d = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / d

def show_1(mat, title=''):
    sns.heatmap(mat)
    plt.title(title)
    plt.show()

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = np.random.randn(embed_dim, embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim)

        show(self.W_q, 'Query Weights')
        show(self.W_k, 'Key Weights')
        show(self.W_v, 'Value Weights')
        show(self.W_o, 'Oputput Weights')

    def forward(self, query, key, value):
        # Linear transformation
        Q = np.dot(query, self.W_q)
        show(Q, 'Transformed Query by weights')
        Q = np.reshape(Q, (-1, self.num_heads, self.head_dim))
        show_2(Q, 'Reshaped Query by weights')
        for q in np.swapaxes(Q, 0, 1):
            show(q, 'Query Head')

        K = np.dot(key, self.W_k)
        show(K, 'Transformed Key by weights')
        K = np.reshape(K, (-1, self.num_heads, self.head_dim))
        show_2(K, 'Reshaped Key by weights')
        for k in np.swapaxes(K, 0, 1):
            show(k, 'Key Head')


        V = np.dot(value, self.W_v)
        show(V, 'Transformed Value by weights')
        V = np.reshape(V, (-1, self.num_heads, self.head_dim))
        show_2(V, 'Reshaped Value by weights')
        for v in np.swapaxes(V, 0, 1):
            show(v, 'Value Head')

        # Scaled dot-product attention
        attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        show_2(attention_scores, 'Attention Scores')
        attention_probs = softmax(attention_scores)
        show_2(attention_probs, 'Attention Probabilities')
        attention_output = np.matmul(attention_probs, V)
        show_2(attention_output, 'Attention Output')

        # Concatenate heads and linear transformation
        attention_output = np.reshape(attention_output, (-1, self.embed_dim))
        show_2(attention_output, 'Attention Output Reshaped')
        output = np.dot(attention_output, self.W_o)
        show(output, 'Output')

        return output

# Define feed-forward network layer
class FeedForwardNetwork:
    def __init__(self, embed_dim, hidden_dim):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Initialize parameters
        self.W_1 = np.random.randn(embed_dim, hidden_dim)

        
        print('Left Hidden')
        show(self.W_1)
        self.W_2 = np.random.randn(hidden_dim, embed_dim)

        print('Right Hidden')
        show(self.W_2)
    def forward(self, x):
        # Linear transformation and activation function (e.g., ReLU)
        hidden_output = relu(np.dot(x, self.W_1))
        output = np.dot(hidden_output, self.W_2)

        return output

# Define transformer model
class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Initialize layers
        self.embedding_weights = np.random.randn(vocab_size, embed_dim)

        print('Embedding Weights')
        show(self.embedding_weights)

        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward_network = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, input_sequence):
        # Embedding layer
        embedded_input = self.embedding_weights[input_sequence]

        # Multi-head attention layer
        attention_output = self.multihead_attention.forward(embedded_input, embedded_input, embedded_input)

        # Residual connection and layer normalization
        attention_output += embedded_input
        attention_output /= np.sqrt(self.embed_dim)  # Layer normalization

        # Feed-forward network layer
        ff_output = self.feedforward_network.forward(attention_output)

        # Residual connection and layer normalization
        output = ff_output + attention_output
        output /= np.sqrt(self.embed_dim)  # Layer normalization

        return output

