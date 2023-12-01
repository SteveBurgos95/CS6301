import torch
import torch.nn as nn
from momentumRNN import MomentumLSTMCell

class MomentumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mu=0.6, s=0.6):
        super(PredictiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = MomentumLSTMCell(input_size, hidden_size, mu=mu, s=s)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden, v):
        debugging = False
        
        if debugging == True: 
            print('From model: hx size: ', hidden[0].size())
            print('From model: ', v.size())
        outputs = []
        for x in input_seq:
            #hidden, _, v = self.lstm_cell(x, hidden, v)
            x, hidden_tuple, v = self.lstm_cell(x, hidden, v)

            if debugging == True:
                print("from model: x size:", x.size())
                print("from model: cy size from hidden_tuple:", hidden_tuple[1].size())
            output = self.fc(x)
            outputs.append(output)
        return torch.stack(outputs), hidden_tuple, v

'''
# Example usage:
input_size = 10
hidden_size = 20
output_size = 1
seq_length = 5
batch_size = 32

# Create an instance of the PredictiveLSTM model
model = PredictiveLSTM(input_size, hidden_size, output_size)

# Dummy input sequence
input_seq = torch.randn(seq_length, batch_size, input_size)

# Initial hidden state and momentum state
initial_hidden = (
    torch.zeros(batch_size, hidden_size),
    torch.zeros(batch_size, hidden_size)
)
initial_momentum = torch.zeros(batch_size, hidden_size)

# Forward pass
output_seq, final_hidden, final_momentum = model(input_seq, initial_hidden, initial_momentum)

print("Output sequence shape:", output_seq.shape)
print("Final hidden state shape:", final_hidden[0].shape)
print("Final momentum state shape:", final_momentum.shape)
'''