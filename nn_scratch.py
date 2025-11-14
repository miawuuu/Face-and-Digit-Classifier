import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 learning_rate=1e-1, seed=0):
        np.random.seed(seed)
        
        # Set weights and biases for each layer
        self.W1 = self._xavier_init(input_size, hidden_size1)
        self.b1 = np.zeros(hidden_size1)
        self.W2 = self._xavier_init(hidden_size1, hidden_size2)
        self.b2 = np.zeros(hidden_size2)
        self.W3 = self._xavier_init(hidden_size2, output_size)
        self.b3 = np.zeros(output_size)
        
        # Learning rate
        self.lr = learning_rate

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        # Convert label vector y to one-hot matrix
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def _xavier_init(self, fan_in, fan_out):
        # Xavier/Glorot uniform initialization
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    def forward(self, X):
        # Forward pass: compute activations layer by layer
        
        # Layer 1: pre-activation and ReLU activation
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        
        # Layer 2: pre-activation and ReLU activation
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        scores = a2 @ self.W3 + self.b3
        probs = self._softmax(scores)
        return probs, (X, z1, a1, z2, a2, scores, probs)

    def backward(self, cache, y):
        # Backward pass: compute gradients of loss w.r.t. params
        X, z1, a1, z2, a2, scores, probs = cache
        N = X.shape[0]
        y_onehot = self._one_hot(y, probs.shape[1])
        dscores = (probs - y_onehot) / N

        # Gradients for W3 and b3
        dW3 = a2.T @ dscores
        db3 = np.sum(dscores, axis=0)

        # Backprop into layer 2
        da2 = dscores @ self.W3.T
        dz2 = da2 * self._relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Backprop into layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def update_params(self, grads):
        # Gradient descent parameter update
        self.W1 -= self.lr * grads['W1']
        self.b1 -= self.lr * grads['b1']
        self.W2 -= self.lr * grads['W2']
        self.b2 -= self.lr * grads['b2']
        self.W3 -= self.lr * grads['W3']
        self.b3 -= self.lr * grads['b3']

    def train(self, X, y, X_val, y_val, epochs=10, batch_size=64):
        num_train = X.shape[0]
        best_val_acc = 0
        best_weights = None

        # Shuffle training data at each epoch
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_train)
            X, y = X[indices], y[indices]

            for i in range(0, num_train, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass, backwaard pass, and update
                probs, cache = self.forward(X_batch)
                grads = self.backward(cache, y_batch)
                self.update_params(grads)

            # Validation accuracy
            val_preds = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {
                    'W1': self.W1.copy(), 'b1': self.b1.copy(),
                    'W2': self.W2.copy(), 'b2': self.b2.copy(),
                    'W3': self.W3.copy(), 'b3': self.b3.copy()
                }

        # Restore best model
        if best_weights:
            self.W1 = best_weights['W1']
            self.b1 = best_weights['b1']
            self.W2 = best_weights['W2']
            self.b2 = best_weights['b2']
            self.W3 = best_weights['W3']
            self.b3 = best_weights['b3']

    def predict(self, X):
        # Compute class predictions for inputs X
        probs, _ = self.forward(X)
         # Return class with highest probability
        return np.argmax(probs, axis=1)
