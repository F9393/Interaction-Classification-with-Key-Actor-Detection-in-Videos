import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention1(nn.Module):    
    def __init__(self, hidden_size, key_size):
        """
        hidden-size = size of hidden_dim of eventLSTM
        key_size = size of feature representation of player

        """
        super(Attention1, self).__init__()
    
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, query, keys, mask = None):
        """
        query : (B,1,H) (prev LSTM hidden state)
        keys : (B,P,K) (batch_size, #players, #key_size) (key_size = number of keypoints)
        mask : (B,P)
        
        """

        # [B,1,H] -> [B,1,H]
        proj_query = self.query_layer(query)

        # [B,P,K] -> [B,P,H]
        proj_keys = self.key_layer(keys)

        # [B,1,H] + [B,P,H] = [B,P,H]
        energies = torch.tanh(proj_query + proj_keys)
        
        # Calculate energies.
        # [B,P,H] -> [B,P]
        energies = self.energy_layer(energies).squeeze(2)
        
        # Mask invalid positions.
        if mask is not None:
            energies.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        alphas = F.softmax(energies, dim=1)   
        
        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(alphas.unsqueeze(1), keys)

        # embeddings_shape: [B,1,K], alphas shape: [B,P]
        return context, alphas

class Attention2(nn.Module):    
    def __init__(self, hidden_size, key_size):
        """
        query-size = size of hidden_dim of eventLSTM
        key_size = size of feature representation of player

        """
        super(Attention2, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(key_size + hidden_size, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64,1,bias=False)
        )
        
    def forward(self, query, keys, mask = None):
        """
        query : (B,1,H) (prev LSTM hidden state)
        keys : (B,P,K) (batch_size, #players, #key_size) (key_size = number of keypoints)
        mask : (B,P)
        
        """
        
        # [B,1,H] -> [B,P,H]
        repeat_query = query.repeat(1,keys.size(1),1)

        # cat([B,P,K],[B,P,H]) -> [B,P,K+H]
        concatenated = torch.cat([keys,repeat_query],2)

        # [B,P,K+H] -> [B,P]
        energies = self.mlp(concatenated).squeeze(2)

        # Mask invalid positions.
        if mask is not None:
            energies.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        alphas = F.softmax(energies, dim=1)   

        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(alphas.unsqueeze(1), keys)
        
        # context shape: [B,1,K], alphas shape: [B,P]
        # context is also the input to current time step of LSTM, so returned first
        return context, alphas


if __name__ == "__main__":
    print('attention1 test')
    query = torch.randn((8,1,128))
    keys = torch.randn((8,5,30))
    attn = Attention1(128,30)
    out = attn(query,keys)
    print(out[0].shape, out[1].shape)

    print('attention2 test')
    query = torch.randn((8,1,128))
    keys = torch.randn((8,5,30))
    attn = Attention2(128,30)
    out = attn(query,keys)
    print(out[0].shape, out[1].shape)