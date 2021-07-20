import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import FrameLSTM, EventLSTM
from .attention import Attention1, Attention2

class Model1(nn.Module):
    def __init__(self, frameLSTM, eventLSTM, CNN_embed_dim, num_classes, **kwargs):
        """
        Phase-1 Model

        Parameters
        ----------
        frameLSTM: dictionary 
            dictionary containing parameters of frame-level LSTM. 
            Must contain 'hidden_size' key representing size of hidden_dim of LSTM. 
            'winit' and 'forget_gate_bias' keys are optional.
        eventLSTM: dictionary 
            dictionary containing parameters of event-level LSTM. 
            Must contain 'hidden_size' key representing size of hidden_dim of LSTM. 
            'winit' and 'forget_gate_bias' keys are optional.
        CNN_embed_dim  : int
            size of embedding layer in encoder
        num_classes : int
            number of output classes of model

        """
        super(Model1, self).__init__()

        self.encoder = Encoder(CNN_embed_dim = CNN_embed_dim)
        self.frameLSTM = FrameLSTM(input_size = CNN_embed_dim, **frameLSTM, **kwargs)
        self.eventLSTM = EventLSTM(input_size = 2 * frameLSTM['hidden_size'], **eventLSTM, **kwargs)

        self.fc = nn.Linear(in_features = eventLSTM['hidden_size'], out_features = num_classes)

        # initialize FC layer
        # nn.init.constant_(self.fc.bias, 0.0)
        # nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        """
        x : shape (B,T,I)
        out : shape (B,O)

        """

        out = self.encoder(x)

        f_out, (f_h_n, f_h_c) = self.frameLSTM(out)  

        e_out, (e_h_n, e_h_c) = self.eventLSTM(f_out)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

class Model2(nn.Module):
    def __init__(self, eventLSTM, input_size, num_classes, **kwargs):
        """
        Phase-2 Model

        Parameters
        ----------
        eventLSTM: dictionary 
            dictionary containing parameters of event-level LSTM. 
            Must contain 'hidden_size' key representing size of hidden_dim of LSTM. 
            'winit' and 'forget_gate_bias' keys are optional.
        input_size  : int
            size of input to eventLSTM
        num_classes : int
            number of output classes of model
        
        """

        super(Model2, self).__init__()

        self.eventLSTM = EventLSTM(input_size = input_size, **eventLSTM)

        self.fc = nn.Linear(in_features = eventLSTM['hidden_size'], out_features = num_classes)

    def forward(self, x):
        """
        x : shape (B,T,I)
        out : shape (B,O)
        
        """

        e_out, (e_h_n, e_h_c) = self.eventLSTM(x)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

class Model3(nn.Module):
    def __init__(self, eventLSTM, input_size, num_classes, attention_type, **kwargs):
        """
        Phase-3 Model

        Parameters
        ----------
        eventLSTM: dictionary 
            dictionary containing parameters of event-level LSTM. 
            Must contain 'hidden_size' key representing size of hidden_dim of LSTM. 
            'winit' and 'forget_gate_bias' keys are optional.
        input_size  : int
            size of input to eventLSTM
        num_classes : int
            number of output classes of model
        attention_type: 1 or 2
        
        """

        super(Model3, self).__init__()

        self.hidden_size = eventLSTM['hidden_size']

        if attention_type == 1 :
            self.attention = Attention1(self.hidden_size, input_size)
        elif attention_type == 2:
            self.attention = Attention2(self.hidden_size, input_size)
        else:
            raise Exception("invalid attention type! Must be either 1 or 2.")
            
        self.eventLSTM = EventLSTM(input_size = input_size, **eventLSTM)
        self.fc = nn.Linear(in_features = eventLSTM['hidden_size'], out_features = num_classes)

    def forward(self, x):
        """
        x : shape (B,T,P,I) (batch_size, #frames, #person, person feature size)
        out : shape (B,O)
        
        """
        out = torch.zeros(x.size(0), 1, self.hidden_size, device=torch.device("cuda"))
        hidden = torch.zeros(1, x.size(0), self.hidden_size, device=torch.device("cuda"))
        cell = torch.zeros(1, x.size(0), self.hidden_size, device=torch.device("cuda"))

        for t in range(x.size(1)):
            embeddings,_ = self.attention(out, x[:,t,:,:])
            
            out, (hidden,cell) = self.eventLSTM(embeddings, (hidden,cell))
        
        # [B,1,H] -> [B,1,O]
        out = self.fc(out) 

        return out[:,-1,:] 

if __name__ == "__main__":
    import torch

    print(f'Model 1 Test')
    inp = torch.randn((2,5,3,224,224)) #(batch, time_step, channels, img_h, img_w)
    m1 = Model1(frameLSTM={"hidden_size":128}, eventLSTM={"hidden_size":128}, CNN_embed_dim=32, num_classes=8)
    out = m1(inp)
    print(out.shape)

    print()
    print(f'Model 2 Test')
    inp = torch.randn((2,10,30)) #(B,T,I)
    m2 = Model2(eventLSTM={"hidden_size":128}, input_size=30, num_classes=8)
    out = m2(inp)
    print(out.shape)

    print()
    print(f'Model 3 Test')
    inp = torch.randn((2,10,5,30),device='cuda') #(B,T,P,I)
    m3 = Model3(eventLSTM={"hidden_size":128}, input_size=30, num_classes=8, attention_type=2).cuda()
    out = m3(inp)
    print(out.shape)
