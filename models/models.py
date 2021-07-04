import torch.nn as nn
from .encoder import Encoder
from .decoder import FrameLSTM, EventLSTM

class Model1(nn.Module):
    def __init__(self, CNN_embed_dim = 256, h_frameLSTM = 256, h_eventLSTM = 256, num_classes = 8):
        super(Model1, self).__init__()

        self.encoder = Encoder(CNN_embed_dim = CNN_embed_dim)
        self.frameLSTM = FrameLSTM(input_size = CNN_embed_dim, h_frameLSTM = h_frameLSTM)
        self.eventLSTM = EventLSTM(input_size = 2 * h_frameLSTM, h_eventLSTM = h_eventLSTM)

        self.fc = nn.Linear(in_features = h_eventLSTM, out_features = num_classes)


        # initialize FC layer
        # nn.init.constant_(self.fc.bias, 0.0)
        # nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):

        out = self.encoder(x)

        f_out, (f_h_n, f_h_c) = self.frameLSTM(out)  

        e_out, (e_h_n, e_h_c) = self.eventLSTM(f_out)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out.reshape(-1,H)).view(B,T,-1) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

class Model2(nn.Module):
    def __init__(self, input_size = 30, h_eventLSTM = 256, num_classes = 8):
        super(Model2, self).__init__()

        self.eventLSTM = EventLSTM(input_size = input_size, h_eventLSTM = h_eventLSTM)

        self.fc = nn.Linear(in_features = h_eventLSTM, out_features = num_classes)

    def forward(self, x):
        """
        x.shape : (B,T,30) 
        """

        e_out, (e_h_n, e_h_c) = self.eventLSTM(x)

        B,T,H = e_out.shape # batch,time_step,hidden_size
        
        out = self.fc(e_out.reshape(-1,H)).view(B,T,-1) # pass output hidden states of all time steps to the same FC layer (will be needed for hinge loss)

        return out[:,-1,:] # return last time step for now

if __name__ == "__main__":
    import torch
    inp = torch.randn((2,5,3,224,224)) # (batch, time_step, channels, img_h, img_w)

    print(f'Model 1 Test')
    inp = torch.randn((2,5,3,224,224))
    m1 = Model1(CNN_embed_dim = 10, h_frameLSTM = 4, h_eventLSTM = 6, num_classes = 8)
    out = m1(inp)
    print(out.shape)

    print()
    print(f'Model 2 Test')
    inp = torch.randn((2,10,30))
    m2 = Model2(input_size=30, h_eventLSTM=256, num_classes=8)
    out = m2(inp)
    print(out.shape)

    