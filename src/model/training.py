import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.model.model import PARModel  # Assuming the model is defined in model.py

# Define the focal loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, 88, 5, T) - model output
            targets: (B, 88, T) - ground truth note states
        Returns:
            loss: scalar loss value
        """
        B, C, T = targets.shape
        inputs = inputs.permute(0, 3, 1, 2)  # (B, T, 88, 5)
        inputs = inputs.reshape(-1, 5)  # (B*T*88, 5)
        targets = targets.reshape(-1)  # (B*T*88,)

        # Compute softmax over the last dimension
        p = torch.softmax(inputs, dim=-1)

        # Extract the probability of the true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()

# Define a placeholder dataset class
class PianoTranscriptionDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=100):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Placeholder: return random data with correct shapes
        mel_spec = torch.randn(1, 700, self.seq_length)
        state_labels = torch.randint(0, 5, (88, self.seq_length))
        context = torch.randn(88, 3)  # [state, duration, velocity] for each pitch
        return mel_spec, state_labels, context

# Hyperparameters
batch_size = 12
learning_rate = 1e-3
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_dataset = PianoTranscriptionDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = PARModel().to(device)
criterion = FocalLoss(alpha=1.0, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (mel_spec, state_labels, context) in enumerate(train_loader):
            mel_spec = mel_spec.to(device)
            state_labels = state_labels.to(device)
            context = context.to(device)

            # Forward pass
            state_probs = model(mel_spec, context)

            # Compute loss
            loss = criterion(state_probs, state_labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Start training
train_model()
