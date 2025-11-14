import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden1=128, hidden2=64, output_size=10):
        super(ThreeLayerNN, self).__init__()
        
        #Layer definitions
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        # Layer 1: Linear transformation + ReLU activation
        x = F.relu(self.fc1(x))
        
        # Layer 2: Linear transformation + ReLU activation
        x = F.relu(self.fc2(x))
        
        # Layer 3: Linear transformation (output layer)
        x = self.fc3(x)
        return x

def train_model_with_val(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Convert NumPy arrays into a PyTorch Dataset and DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            # Zero out any previously computed gradients
            optimizer.zero_grad()
            
            # Forward pass: compute logits for the batch
            pred = model(xb)
            
            # Compute cross-entropy loss comparing logits to true labels
            loss = criterion(pred, yb)
            
            # Backpropagate to compute gradients
            loss.backward()
            
            # Update model parameters using the optimizer
            optimizer.step()

        # Evaluate on validation set after each epoch
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32))
            val_labels = torch.argmax(val_preds, dim=1).numpy()
            val_acc = accuracy_score(y_val, val_labels)

            #print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")

            # If this epoch gave a better validation accuracy, save the model weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()

    return best_model_state
