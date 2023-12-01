import torch
import torch.nn as nn
from torch.nn import functional as F

class MomentumLSTMCell(nn.Module):
    
    """
    An implementation of MomentumLSTM Cell
    Args :
    input_size: The number of expected features in the input 'x' 
    hidden_size: The number of features in the hidden state 'h'
    mu: momentum coefficient in MomentumLSTM Cell
    s : step size in MomentumLSTM Cell
    bias: If ''False'', then the layer does not use bias weights '
    b_ih ' and 'b_hh '. Default : ''True ''
    Inputs: input, hidden0=(h_0, c_0), v0
    - input of shape '(batch , input_size) ': tensor containing input
    features
    - h_0 of shape '(batch , hidden_size) ': tensor containing the
    initial hidden state for each element in the batch.
    - c_0 of shape '(batch , hidden_size) ': tensor containing the
    initial cell state for each element in the batch.
    - v0 of shape '(batch , hidden_size) ': tensor containing the
    initial momentum state for each element in the batch
    Outputs: h1, (h_1, c_1), v1
    - h_1 of shape '(batch , hidden_size) ': tensor containing the next
    hidden state for each element in the batch
    - c_1 of shape '(batch , hidden_size) ': tensor containing the next
    cell state for each element in the batch
    - v_1 of shape '(batch , hidden_size) ': tensor containing the next
    momentum state for each element in the batch
    """

    def __init__(self, input_size, hidden_size, mu = 0.6, s = 0.6, bias = True):
        super(MomentumLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias = bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias = bias)

        # for momentum
        self.mu = mu
        self.s = s

    def reset_parameters(self, hidden_size):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.eye_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[hidden_size:(2 * hidden_size)].fill_(1.0)
        nn.init.zeros_(self.h2h.bias)
        self.h2h.bias.data[hidden_size:(2 * hidden_size)].fill(1.0)

    def forward(self, x, hidden, v):
        debugging = False
        
        print('x: ', x)

        hx, cx = hidden
        x = x.view(-1, x.size(1))
        v = v.view(-1, v.size(1))

        if debugging == True:
            print("hx size:", hx.size())
            print("cx size:", cx.size())

            print("x size:", x.size())
            print("v size:", v.size())
            print("x2h size:", self.x2h(x).size())
            print("h2h size:", self.h2h(hx).size())
            
            print('Comparing ', v.size(dim=1), 'to ', self.x2h(x).size(dim=1))

        if v.size(dim=1) != self.x2h(x).size(dim=1):
            print('Modifed size')
            v = self.h2h(v)

        if debugging == True: 
            print('v size: ', v.size())
        #vy = self.mu * v + self.s * self.x2h(x)
        vy = self.mu * v + self.s * self.x2h(x) # Modifed

        if debugging == True: 
            print("vy size:", vy.size())

        gates = vy + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        #print('cx: ', cx[0])
        cy = torch.mul(cx[0], forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        if debugging == True: 
            print('hy: ', hy.size())
            print('cy: ', cy.size())

        return hy, (hy, cy), vy

