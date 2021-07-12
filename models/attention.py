import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):    
    def __init__(self, hidden_size, key_size, query_size=None):
        """
        query-size = size of hidden_dim of eventLSTM
        key_size = size of feature representation of player

        """
        super(Attention, self).__init__()
    
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, query, keys, mask = None):
        """
        query : (B,1,Q)
        keys : (B,P,K) #(batch_size, #players, #key_size)
        mask : (B,P)
        
        """

        # [B,1,Q] -> [B,1,H]
        proj_query = self.query_layer(query)

        # [B,P,K] -> [B,P,H]
        proj_keys = self.key_layer(keys)

        # [B,1,H] + [B,P,H] = [B,P,H]
        energies = torch.tanh(proj_query + proj_keys)
        
        # Calculate energies.
        # [B,P,H] -> [B,P]
        energies = self.energy_layer(energies).squeeze(2)
        
        # Mask out invalid positions.
        if mask is not None:
            energies.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        alphas = F.softmax(energies, dim=1)   
        
        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(alphas.unsqueeze(1), keys)
        context.squeeze_(1)
        
        # context shape: [B, K]
        return context