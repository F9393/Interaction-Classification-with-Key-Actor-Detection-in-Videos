import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention1(nn.Module):    
    """
    attention for phase 3 model.
    """

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
        query : (B,1,H) (prev eventLSTM hidden state)
        keys : (B,P,K) (batch_size, #players, #key_size) (key_size = number of keypoints)
        mask : (B,P)
        
        """

        # [B,1,H] -> [B,1,H]
        proj_query = self.query_layer(query)

        # [B,P,K] -> [B,P,H]
        proj_keys = self.key_layer(keys)

        # [B,1,H] + [B,P,H] = [B,P,H]
        temp_sum = torch.tanh(proj_query + proj_keys)
        
        # Calculate raw_scores.
        # [B,P,H] -> [B,P]
        raw_scores = self.energy_layer(temp_sum).squeeze(2)

        # Mask invalid positions.
        if mask is not None:
            raw_scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        weights = F.softmax(raw_scores, dim=1)   

        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(weights.unsqueeze(1), keys)

        # embeddings_shape: [B,1,K], weights shape: [B,P]
        return context, weights

class Attention2(nn.Module):   
    """
    attention for phase 3 model.
    """

    def __init__(self, hidden_size, key_size):
        """
        hidden-size = size of hidden_dim of eventLSTM
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
        query : (B,1,H) (prev eventLSTM hidden state)
        keys : (B,P,K) (batch_size, #players, #key_size) (key_size = number of keypoints)
        mask : (B,P)
        
        """
        
        # [B,1,H] -> [B,P,H]
        repeat_query = query.repeat(1,keys.size(1),1)

        # cat([B,P,K],[B,P,H]) -> [B,P,K+H]
        concatenated = torch.cat([keys,repeat_query],2)

        # [B,P,K+H] -> [B,P]
        raw_weights = self.mlp(concatenated).squeeze(2)

        # Mask invalid positions.
        if mask is not None:
            raw_weights.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        weights = F.softmax(raw_weights, dim=1)   

        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(weights.unsqueeze(1), keys)
        
        # context shape: [B,1,K], weights shape: [B,P]
        # context is also the input to current time step of LSTM, so returned first
        return context, weights

class Attention3(nn.Module):  
    """
    attention for phase 4 model
    """  

    def __init__(self, key_size, eventLSTM_h_dim, frameLSTM_h_dim):
        """
        key_size = size of feature representation of player
        eventLSTM_h_dim = hidden dim of eventLSTM
        frameLSTM_h_dim = hidden dim of frameLSTM

        """
        super(Attention3, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(key_size + eventLSTM_h_dim + 2 * frameLSTM_h_dim, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64,1,bias=False)
        )
        
    def forward(self, query, keys, frameLSTM_h, mask = None):
        """
        query : (B,1,H_e) (prev eventLSTM hidden state)
        keys : (B,P,K) (batch_size, #players, #key_size) (key_size = number of keypoints)
        frameLSTM_h : (B,1,2*H_f) (batch_size, 1, 2 * hidden dim of frameLSTM)
        mask : (B,P)
        
        """
        
        # [B,1,H_e] -> [B,P,H_e]
        repeat_query = query.repeat(1,keys.size(1),1)

        # [B,1,H_f] -> [B,P,H_f]
        repeat_frameLSTM_h = frameLSTM_h.repeat(1,keys.size(1),1)

        # cat([B,P,K],[B,P,H_e], [B,P,H_f]) -> [B,P,K+H_e+H_f]
        concatenated = torch.cat([keys,repeat_query, repeat_frameLSTM_h],2)

        # [B,P,K+H_e+H_f] -> [B,P]
        raw_scores = self.mlp(concatenated).squeeze(2)

        # Mask invalid positions.
        if mask is not None:
            raw_scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        # [B,P] -> [B,P]
        weights = F.softmax(raw_scores, dim=1)   

        # same as softmax but with numerical stability. 
        # b,_ = torch.max(raw_scores, dim=1, keepdim=True)
        # raw_scores = torch.exp(raw_scores - b)
        # c = torch.sum(raw_scores,1,keepdim=True) + 1.2e-38
        # weights = raw_scores / c


        # The context vector is the weighted sum of the values.
        # [B,P] -> [B,1,P]
        # [B,1,P] * [B,P,K] -> [B,1,K]
        context = torch.bmm(weights.unsqueeze(1), keys)

        # uncomment below return stmt. when we want to input only weighted pose vector to the eventLSTM.(another change has to be made in models.py for this to work)
        # return context,weights

        # cat([B,1,K],[B,1,2*H_f]) -> [B,1,K+2*H_f]
        embeddings = torch.cat([context, frameLSTM_h], 2)
        
        # embeddings shape: [B,1,K+2*H_f], weights shape: [B,P]
        return embeddings, weights

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

    print('attention3 test')
    query = torch.randn((8,1,256))
    keys = torch.randn((8,5,30))
    frameLSTM_h = torch.randn((8,1,512))
    attn = Attention3(30,256,256)
    out = attn(query,keys,frameLSTM_h)
    print(out[0].shape, out[1].shape)