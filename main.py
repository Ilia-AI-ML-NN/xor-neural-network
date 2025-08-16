
import numpy as np
import matplotlib.pyplot as plt

# =============== Activation functions ===============
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# =============== Forward pass ===============
def forward_pass(X, W1, b1, W2, b2):
    hidden_input = X.dot(W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = hidden_output.dot(W2) + b2
    final_output = sigmoid(final_input)
    return hidden_output, final_output

# =============== Backward pass ===============
def backward_pass(y, hidden_output, final_output, W2):
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    return d_output, d_hidden, error

# =============== Training ===============
def train(
    X, y, *, input_size=2, hidden_size=2, output_size=1,
    lr=0.1, epochs=10000, seed=42
):
    np.random.seed(seed)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    loss_history = []
    acc_history = []

    for _ in range(epochs):
        h, out = forward_pass(X, W1, b1, W2, b2)
        d_out, d_hid, err = backward_pass(y, h, out, W2)

        # gradient step
        W2 += h.T.dot(d_out) * lr
        b2 += d_out.sum(axis=0, keepdims=True) * lr
        W1 += X.T.dot(d_hid) * lr
        b1 += d_hid.sum(axis=0, keepdims=True) * lr

        # metrics
        loss = np.mean(err ** 2)
        preds = (out > 0.5).astype(int)
        acc = np.mean(preds == y)
        loss_history.append(loss)
        acc_history.append(acc)

    return W1, b1, W2, b2, loss_history, acc_history

# =============== Prediction ===============
def predict(X, W1, b1, W2, b2, threshold=0.5):
    _, out = forward_pass(X, W1, b1, W2, b2)
    return (out > threshold).astype(int), out

# =============== Plot helpers ===============
def plot_training_curves(loss_history, acc_history):
    # Plot loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(acc_history)
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, W1, b1, W2, b2, grid_points=200):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points),
        np.linspace(y_min, y_max, grid_points)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z, _ = predict(grid, W1, b1, W2, b2)
    Z = Z.reshape(xx.shape)

    plt.figure()
    # default colormap/colors (no manual colors)
    plt.contourf(xx, yy, Z, alpha=0.7)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors="k")
    plt.title("Decision Boundary for XOR")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()

# =============== CLI entry ===============
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Hyperparameters (change to experiment)
    HIDDEN_SIZE = 2
    LR = 0.1
    EPOCHS = 10000

    # Train
    W1, b1, W2, b2, loss_hist, acc_hist = train(
        X, y, hidden_size=HIDDEN_SIZE, lr=LR, epochs=EPOCHS
    )

    # Report
    preds_cls, preds_prob = predict(X, W1, b1, W2, b2)
    print("Final probabilities:\n", np.round(preds_prob, 3))
    print("Final predictions:\n", preds_cls)
    print("Final Accuracy:", np.mean(preds_cls == y))

    # Visualizations
    plot_training_curves(loss_hist, acc_hist)
    plot_decision_boundary(X, y, W1, b1, W2, b2)

    # Save model
    np.savez("xor_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print("Model saved to xor_model.npz")
