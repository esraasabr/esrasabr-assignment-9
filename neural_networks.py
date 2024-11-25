import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Activation Functions
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) 

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = {
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        }[activation]
        self.activation_derivative = {
            "tanh": lambda x: 1 - np.tanh(x) ** 2,
            "relu": lambda x: (x > 0).astype(float),
            "sigmoid": lambda x: self.activation_fn(x) * (1 - self.activation_fn(x)),
        }[activation]

        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

        # Storage for visualization
        self.hidden_output = None
        self.gradients = None

    
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.hidden_output = self.activation_fn(self.z1)
        self.z2 = self.hidden_output @ self.W2 + self.b2
        self.output = sigmoid(self.z2)  # Output layer uses sigmoid for binary classification
        # TODO: store activations for visualization
        return self.output

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        output_error = self.output - y
        dW2 = self.hidden_output.T @ output_error
        db2 = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = output_error @ self.W2.T * self.activation_derivative(self.z1)
        dW1 = X.T @ hidden_error
        db1 = np.sum(hidden_error, axis=0, keepdims=True)

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        self.gradients = (np.abs(dW1).mean(), np.abs(dW2).mean())


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    batch_size = 10
    for _ in range(1):  # Gradual updates for smoother transitions
        batch_indices = np.random.permutation(len(X))[:batch_size]
        mlp.forward(X[batch_indices])
        mlp.backward(X[batch_indices], y[batch_indices])
        
    # TODO: Plot hidden features
    hidden_features = mlp.activation_fn(X @ mlp.W1 + mlp.b1)  # Compute for all data
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y[:hidden_features.shape[0]].ravel(),  # Ensure shape consistency
        cmap='bwr', alpha=0.7, edgecolor='k'
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.grid(True)

    # TODO: Hyperplane visualization in the hidden space
     # Visualize hyperplanes for each hidden neuron
    x_vals = np.linspace(-2, 2, 10)
    y_vals = np.linspace(-2, 2, 10)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

    for i in range(mlp.W1.shape[1]):  # One plane per hidden neuron
        weights = mlp.W1[:, i]
        bias = mlp.b1[0, i]
        z_mesh = (-weights[0] * x_mesh - weights[1] * y_mesh - bias) / max(1e-5, np.linalg.norm(weights))  # Normalize weights
        ax_hidden.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.2, label=f"Neuron {i+1} Plane")
        
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.set_xlim([-2, 2])
    ax_hidden.set_ylim([-2, 2])
    ax_hidden.set_zlim([-2, 2])
    ax_hidden.grid(True)

    # TODO: Distorted input space transformed by the hidden layer
    transformed_space = mlp.activation_fn(X @ mlp.W1 + mlp.b1)
    ax_input.scatter(transformed_space[:, 0], transformed_space[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title("Distorted Input Space Transformed by Hidden Layer")

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)

    # Apply smoothing
    if 'prev_predictions' in globals():
        alpha = 0.9  # Smoothing factor
        predictions = alpha * predictions + (1 - alpha) * prev_predictions
    prev_predictions = predictions

    if 'prev_hidden_features' in globals():
        alpha = 0.9  # Smoothing factor
        hidden_features = alpha * hidden_features + (1 - alpha) * prev_hidden_features
    prev_hidden_features = hidden_features

    ax_input.cla()
    ax_input.contourf(xx, yy, predictions, alpha=0.7, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title(f"Input Space at Step {frame}")
    ax_input.grid(True)

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    features = ['x1', 'x2', 'h1', 'h2', 'h3', 'y']
    positions = {
        0: (0, 1),  # x1
        1: (0, 0),  # x2
        2: (1, 1.5),  # h1
        3: (1, 0.5),  # h2
        4: (1, -0.5),  # h3
        5: (2, 0.5),  # y
    }  # Define positions for nodes in 2D space
    edge_labels = [
        (0, 2), (0, 3), (0, 4),  # x1 -> h1, h2, h3
        (1, 2), (1, 3), (1, 4),  # x2 -> h1, h2, h3
        (2, 5), (3, 5), (4, 5),  # h1, h2, h3 -> y
    ]
    gradient_magnitudes = [
        np.abs(mlp.W1[0, 0]), np.abs(mlp.W1[0, 1]), np.abs(mlp.W1[0, 2]),  # x1 -> h1, h2, h3
        np.abs(mlp.W1[1, 0]), np.abs(mlp.W1[1, 1]), np.abs(mlp.W1[1, 2]),  # x2 -> h1, h2, h3
        np.abs(mlp.W2[0, 0]), np.abs(mlp.W2[1, 0]), np.abs(mlp.W2[2, 0]),  # h1, h2, h3 -> y
    ]

      # Normalize gradient magnitudes for consistent scaling
    max_grad = max(gradient_magnitudes) if gradient_magnitudes else 1  # Avoid division by zero
    gradient_magnitudes = [min(grad, 1.0) for grad in gradient_magnitudes]  # Cap at 1.0

    # Draw nodes (circles)
    for i, feature in enumerate(features):
        x, y = positions[i]
        circle = Circle((x, y), 0.1, color='blue', alpha=0.8)  # Node as a circle
        ax_gradient.add_patch(circle)
        ax_gradient.text(x, y + 0.2, feature, fontsize=10, ha='center')  # Label above the node

    # Draw edges (lines) with thickness based on gradient magnitude
    for (start, end), grad in zip(edge_labels, gradient_magnitudes):
        x_start, y_start = positions[start]
        x_end, y_end = positions[end]
        thickness = min(5, 1 + grad * 5)  # Cap thickness at 5
        ax_gradient.plot(
            [x_start, x_end], [y_start, y_end],
            color='purple', alpha=0.7, linewidth=thickness
        )

   # Set plot limits and labels         
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-1, 2)
    ax_gradient.set_title(f"Gradients at Step {frame}")
    ax_gradient.set_ylabel("Node Position")
    ax_gradient.grid(True)  # Add grid for better readability

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    # ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)
    from functools import partial

    # Pass the required arguments using partial
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num,
        interval=3000,  # Slow down frame rate
        repeat=False
    )




    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)