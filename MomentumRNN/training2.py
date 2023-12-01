# Assuming you have the model, loss function, optimizer, and data loaders set up
import torch
import torch.nn as nn
from momentumRNN import MomentumLSTMCell
from model import MomentumLSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Generate synthetic time series data
def generate_synthetic_data(num_samples, sequence_length, num_features):
    data = torch.randn(num_samples, sequence_length, num_features)
    return data

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_length=1):
        self.data = data
        self.target_length = target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = self.data[idx, :-self.target_length]
        target_seq = self.data[idx, self.target_length:]
        return input_seq, target_seq

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data
num_samples = 100
sequence_length = 20
num_features = 1
data = generate_synthetic_data(num_samples, sequence_length, num_features)

# Create dataset and DataLoader
target_length = 1  # Number of steps to predict into the future
dataset = TimeSeriesDataset(data, target_length=target_length)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = 1
hidden_size = 20
output_size = 1
sequence_length = 20
batch_size = 32
#num_layers = 2

model = MomentumLSTM(input_size, hidden_size, output_size)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 10

for epoch in range(epochs):
    for input_seq, target_seq in data_loader:
        #print('input_seq: ', input_seq.size())
        # Initial hidden state and momentum state
        initial_hidden = (
            #torch.zeros(batch_size, hidden_size),
            #torch.zeros(batch_size, hidden_size)
            torch.zeros(input_size, hidden_size),
            torch.zeros(input_size, hidden_size)
        )
        #initial_momentum = torch.zeros(batch_size, hidden_size)
        initial_momentum = torch.zeros(input_size, hidden_size)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output_seq, _, _ = model(input_seq, initial_hidden, initial_momentum)

        print(output_seq)
        print(target_seq)

        # Compute the loss
        loss = loss_function(output_seq, target_seq)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


