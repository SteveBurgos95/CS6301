import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Load the data from block_0.csv
df = pd.read_csv('./Input/daily_dataset/daily_dataset/block_0.csv')

# Assuming you want to predict the 'value' column, replace it with the actual target column
target_column = 'energy_median'

# Convert the 'timestamp' column to datetime format
df['timestamp'] = pd.to_datetime(df['day'])

# Sort the DataFrame by timestamp
df.sort_values(by='timestamp', inplace=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Function to generate time series data
def generate_time_series_data(df, sequence_length):
    data = []

    for i in range(len(df) - sequence_length + 1):
        sequence = df[target_column].iloc[i:i + sequence_length].values
        data.append(sequence)

    return torch.tensor(data, dtype=torch.float32)

# Specify the sequence length
sequence_length = 5  # You can adjust this based on the desired length

# Generate time series data
time_series_data = generate_time_series_data(df, sequence_length)

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = self.data[idx, :-1]  # Input sequence
        target_seq = self.data[idx, -1]   # Target sequence (next element in the time series)
        return input_seq, target_seq

# Create dataset and DataLoader
dataset = TimeSeriesDataset(time_series_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example: Print the first batch
for input_seq, target_seq in data_loader:
    print("Input Sequence:")
    print(input_seq)
    print("Target Sequence:")
    print(target_seq)
    break
